"""模块四：Schema-Evidence Graph 构建 (论文核心创新一/二 + 图谱构建.pdf)。

严格对照《图谱构建.pdf》实现"物理共现知识图 + 文章结构导航图"耦合：

纵向·结构层级图 (§五/§九)：
    Paper(文献根) → Title(标题) / Section(逐级目录) → Chunk → Point
    Section → Figure/Table/Equation；Figure/Table → Caption
横向·物理共现图 (§四)：
    块内点对建边，语义类型 = f(端点标签) (§三)，方向规范化，
    cooccur_scope ∈ same_sentence/same_paragraph/same_chunk/same_caption/...
跨模态 (§十)：
    (1) 正文 chunk 显式引用 Fig./Table/Eq. → references 边
    (2) Figure/Table/Equation 结构归属其 Section
    (3) 图注/表注抽出的概念 与 正文同名概念 → aligns_with 对齐 (同对象跨模态证据)
跨文献 (§八)：
    同 canonical_id 的 Concept/Attribute 跨文献 → same_as

流程：evidence_blocks.jsonl → 结构图[无LLM] → 文本/图注 LLM 点抽取 → 共现边 →
      跨模态引用边 → 跨模态对齐边 → 跨文献聚合边 → nodes/edges.jsonl/graphml
"""

from __future__ import annotations

import dataclasses
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from autoqag.common.graph_store import GraphStore
from autoqag.common.io import read_jsonl_list, write_jsonl
from autoqag.ops.base import BaseStage, PipelineContext
from autoqag.ops.m4_graph.edge_rules import LABEL_TO_NODE_TYPE, resolve_edge
from autoqag.ops.m4_graph.extractor import build_prompt, parse_extraction
from autoqag.ops.m4_graph.quality import (
    is_meaningful_attribute,
    is_valid_point,
)
from autoqag.ops.m4_graph.table_parser import (
    is_numeric_cell,
    parse_table_html,
    unit_from_header,
)
from autoqag.registry import STAGES
from autoqag.schema import (
    Address,
    Edge,
    EvidenceBlock,
    Modality,
    NodeType,
    PointNode,
)


def _slug(text: str, n: int = 24) -> str:
    s = re.sub(r"\s+", "_", text.strip().lower())
    s = re.sub(r"[^0-9a-z一-鿿_]+", "", s)
    return s[:n] or "x"


def _canonical(node_type: str, content: str) -> str:
    return f"{node_type}:{_slug(content, 40)}"


def _locate_sentence(content: str, sentences: List[str]) -> Optional[str]:
    """返回 content 所在句子的序号 (字符串)，找不到返回 None。"""
    head = content.strip()[:20]
    if not head:
        return None
    for idx, s in enumerate(sentences):
        if head in s:
            return str(idx)
    return None


@STAGES.register_module("graph")
class GraphStage(BaseStage):
    declared_inputs = ["evidence_blocks.jsonl"]
    declared_outputs = ["nodes.jsonl", "edges.jsonl", "graph.graphml"]

    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        rows = read_jsonl_list(ctx.path("evidence_blocks.jsonl"))
        if not rows:
            self.log("evidence_blocks.jsonl 为空，先运行 normalize")
            return {"nodes": 0, "edges": 0}

        blocks = [EvidenceBlock.from_dict(r) for r in rows]
        extract_points = self.params.get("extract_points", True)
        lang = self.params.get("lang", "en")
        cross_paper = self.params.get("cross_paper", True)
        paper_titles = self._load_paper_titles(ctx)

        store = GraphStore()

        # 1) 纵向结构图：Paper 根 + Title + Section + Chunk/Figure/Table/Equation/Caption
        self._build_structure(store, blocks, paper_titles)

        # 1b) 表格行列点与共现边 (§四 same_table_row / same_table_column)，无需 LLM
        n_cells = 0
        if self.params.get("parse_table_cells", True):
            n_cells = self._build_table_cells(store, blocks)

        # 2) 横向共现图：文本块 + 图注块 LLM 点抽取 + 块内共现边
        n_points = 0
        if extract_points:
            n_points = self._extract_and_link(ctx, store, blocks, lang)
        else:
            self.log("extract_points=false，跳过 LLM 点抽取，仅建结构图")

        # 3) 跨模态引用边 (§十.1)：正文 → 图/表/公式
        n_cross = self._cross_modal_edges(store, blocks)

        # 4) 跨模态对象对齐 (§十.3)：图注概念 ↔ 正文同名概念
        n_align = self._intra_paper_alignment(store) if extract_points else 0

        # 5) 跨文献同类点聚合 (§八)
        n_xpaper = self._cross_paper_edges(store) if cross_paper else 0

        n_nodes, n_edges = store.export_jsonl(
            ctx.path("nodes.jsonl"), ctx.path("edges.jsonl")
        )
        try:
            store.save_graphml(ctx.path("graph.graphml"))
        except Exception as exc:  # pragma: no cover
            self.log("graphml 导出失败 (忽略): %s", exc)

        self._mark_graph_status(ctx)
        self.log(
            "图谱: nodes=%d edges=%d (points=%d table_cells=%d cross_modal=%d align=%d cross_paper=%d)",
            n_nodes, n_edges, n_points, n_cells, n_cross, n_align, n_xpaper,
        )
        return {
            "nodes": n_nodes,
            "edges": n_edges,
            "points": n_points,
            "table_cells": n_cells,
            "cross_modal_edges": n_cross,
            "alignment_edges": n_align,
            "cross_paper_edges": n_xpaper,
        }

    # ---------------- 辅助 ----------------
    @staticmethod
    def _load_paper_titles(ctx: PipelineContext) -> Dict[str, str]:
        titles: Dict[str, str] = {}
        if ctx.artifact_exists("documents.jsonl"):
            for d in read_jsonl_list(ctx.path("documents.jsonl")):
                titles[d.get("paper_id", "")] = d.get("title", "")
        return titles

    # ---------------- 纵向结构图 (§五) ----------------
    def _build_structure(
        self,
        store: GraphStore,
        blocks: List[EvidenceBlock],
        paper_titles: Dict[str, str],
    ) -> None:
        seen_papers: set = set()
        seen_sections: set = set()

        def ensure_paper(paper_id: str) -> str:
            """创建文献根节点 + 标题点 (图谱构建.pdf §五 树根 / §二 标题点)。"""
            pid = f"{paper_id}::PAPER"
            if pid not in seen_papers:
                seen_papers.add(pid)
                store.upsert_node(
                    PointNode(
                        node_id=pid,
                        node_type=NodeType.PAPER.value,
                        content=paper_titles.get(paper_id, paper_id),
                        address=Address(paper_id=paper_id),
                    )
                )
                title = paper_titles.get(paper_id, "")
                if title:
                    tid = f"{paper_id}::TITLE"
                    store.upsert_node(
                        PointNode(
                            node_id=tid,
                            node_type=NodeType.TITLE.value,
                            content=title,
                            normalized_content=title,
                            address=Address(paper_id=paper_id),
                            canonical_id=_canonical(NodeType.TITLE.value, title),
                        )
                    )
                    store.upsert_edge(
                        Edge(
                            source=pid,
                            target=tid,
                            edge_type="contains",
                            build_reason="document_structure",
                            paper_id=paper_id,
                        )
                    )
            return pid

        def ensure_section(paper_id: str, section_path: str) -> str:
            """逐级建 Section 节点；顶层 Section 挂到 Paper 根，返回叶子 Section id。"""
            paper_node = ensure_paper(paper_id)
            parts = [p for p in section_path.split("/") if p] or ["ROOT"]
            parent_id = paper_node
            acc: List[str] = []
            leaf_id = paper_node
            for part in parts:
                acc.append(part)
                path = "/".join(acc)
                node_id = f"{paper_id}::SEC::{_slug(path, 60)}"
                if node_id not in seen_sections:
                    seen_sections.add(node_id)
                    store.upsert_node(
                        PointNode(
                            node_id=node_id,
                            node_type=NodeType.SECTION.value,
                            content=part,
                            address=Address(paper_id=paper_id, section_path=path),
                            canonical_id=_canonical(NodeType.SECTION.value, part),
                        )
                    )
                    store.upsert_edge(
                        Edge(
                            source=parent_id,
                            target=node_id,
                            edge_type="contains",
                            build_reason="document_structure",
                            paper_id=paper_id,
                            section_path=path,
                        )
                    )
                parent_id = node_id
                leaf_id = node_id
            return leaf_id

        for b in blocks:
            paper_id = b.address.paper_id
            section_node = ensure_section(paper_id, b.address.section_path)

            if b.modality == Modality.TEXT.value:
                node_id = b.address.chunk_id
                store.upsert_node(
                    PointNode(
                        node_id=node_id,
                        node_type=NodeType.CHUNK.value,
                        content=b.content[:500],
                        address=b.address,
                        modality=b.modality,
                    )
                )
                store.upsert_edge(
                    Edge(
                        source=section_node,
                        target=node_id,
                        edge_type="contains",
                        build_reason="document_structure",
                        paper_id=paper_id,
                        section_path=b.address.section_path,
                        chunk_id=node_id,
                    )
                )
            elif b.modality in (
                Modality.TABLE.value,
                Modality.FIGURE.value,
                Modality.FORMULA.value,
            ):
                ntype = {
                    Modality.TABLE.value: NodeType.TABLE.value,
                    Modality.FIGURE.value: NodeType.FIGURE.value,
                    Modality.FORMULA.value: NodeType.EQUATION.value,
                }[b.modality]
                node_id = b.address.chunk_id
                store.upsert_node(
                    PointNode(
                        node_id=node_id,
                        node_type=ntype,
                        content=(b.caption or b.content)[:500],
                        address=b.address,
                        modality=b.modality,
                        domain_schema_tag=str(b.extra.get("label", "")),
                    )
                )
                store.upsert_edge(
                    Edge(
                        source=section_node,
                        target=node_id,
                        edge_type="contains",
                        build_reason="document_structure",
                        paper_id=paper_id,
                        section_path=b.address.section_path,
                    )
                )
            elif b.modality == Modality.CAPTION.value:
                node_id = b.block_id
                store.upsert_node(
                    PointNode(
                        node_id=node_id,
                        node_type=NodeType.CAPTION.value,
                        content=b.content[:500],
                        address=b.address,
                        modality=b.modality,
                    )
                )
                parent = b.extra.get("parent")
                if parent and store.has_node(parent):
                    store.upsert_edge(
                        Edge(
                            source=parent,
                            target=node_id,
                            edge_type="describes",
                            build_reason="document_structure",
                            paper_id=paper_id,
                        )
                    )

    # ---------------- 表格行列点与共现边 (§四) ----------------
    def _build_table_cells(self, store: GraphStore, blocks: List[EvidenceBlock]) -> int:
        """解析表格 HTML 的行列结构，建单元格点与 same_table_row/same_table_column 边。

        典型语义 (图谱构建.pdf §四)：列表头=AttributeNode(指标)，行标签=ConceptNode(对象)，
        数据单元格=ValueNode。列表头→单元格 得 has_value (Attribute→Value)，
        行标签→单元格 记 same_table_row 共现；表头括号内的单位另建 UnitNode + has_unit。
        所有单元格点归属其 TableNode (contains)。
        """
        n_cells = 0
        for b in blocks:
            if b.modality != Modality.TABLE.value:
                continue
            table_id = b.address.chunk_id
            if not store.has_node(table_id):
                continue
            html = b.content or ""
            grid = parse_table_html(html)
            if not grid.cells:
                continue

            # 列表头 → AttributeNode（并尝试抽单位）；按列号索引
            col_attr: Dict[int, str] = {}
            col_unit: Dict[int, str] = {}
            for hc in grid.header_row():
                if hc.col == 0 or not hc.text.strip():
                    continue
                # 质量门控：通用列头 (Value/Parameters/No.) 不建属性，
                # 避免下游产生"该参数的数值是多少"这类无信息数值题
                if not is_meaningful_attribute(hc.text):
                    continue
                aid = f"{table_id}::col{hc.col}"
                store.upsert_node(
                    PointNode(
                        node_id=aid,
                        node_type=NodeType.ATTRIBUTE.value,
                        content=hc.text,
                        normalized_content=hc.text,
                        address=_cell_addr(b.address, hc.row, hc.col),
                        modality=Modality.TABLE.value,
                        canonical_id=_canonical(NodeType.ATTRIBUTE.value, hc.text),
                    )
                )
                store.upsert_edge(_contains(table_id, aid, b.address))
                col_attr[hc.col] = aid
                n_cells += 1
                unit = unit_from_header(hc.text)
                if unit:
                    uid = f"{table_id}::col{hc.col}::unit"
                    store.upsert_node(
                        PointNode(
                            node_id=uid,
                            node_type=NodeType.UNIT.value,
                            content=unit,
                            normalized_content=unit,
                            address=_cell_addr(b.address, hc.row, hc.col),
                            modality=Modality.TABLE.value,
                            canonical_id=_canonical(NodeType.UNIT.value, unit),
                        )
                    )
                    col_unit[hc.col] = uid
                    n_cells += 1

            # 行标签 → ConceptNode；按行号索引
            row_concept: Dict[int, str] = {}
            header_rows = {c.row for c in grid.cells if c.is_header} or {0}
            for c in grid.cells:
                if c.col == 0 and c.row not in header_rows and c.text.strip():
                    # 丢弃单字符/符号行标签 (如参数符号表的 p/n/b)，避免平凡 QA
                    if not is_valid_point(NodeType.CONCEPT.value, c.text, c.text):
                        continue
                    cid = f"{table_id}::row{c.row}"
                    store.upsert_node(
                        PointNode(
                            node_id=cid,
                            node_type=NodeType.CONCEPT.value,
                            content=c.text,
                            normalized_content=c.text,
                            address=_cell_addr(b.address, c.row, c.col),
                            modality=Modality.TABLE.value,
                            canonical_id=_canonical(NodeType.CONCEPT.value, c.text),
                        )
                    )
                    store.upsert_edge(_contains(table_id, cid, b.address))
                    row_concept[c.row] = cid
                    n_cells += 1

            # 数据单元格 → ValueNode，连列表头(同列)与行标签(同行)
            for c in grid.body_cells():
                aid = col_attr.get(c.col)
                cid = row_concept.get(c.row)
                # 列属性与行概念都被过滤掉的孤立单元格无信息，跳过
                if not aid and not cid:
                    continue
                vid = f"{table_id}::r{c.row}c{c.col}"
                store.upsert_node(
                    PointNode(
                        node_id=vid,
                        node_type=NodeType.VALUE.value if is_numeric_cell(c.text)
                        else NodeType.CONCEPT.value,
                        content=c.text,
                        normalized_content=c.text,
                        address=_cell_addr(b.address, c.row, c.col),
                        modality=Modality.TABLE.value,
                    )
                )
                store.upsert_edge(_contains(table_id, vid, b.address))
                n_cells += 1

                # 同列：列表头 → 单元格 (Attribute→Value 命中 has_value)
                if aid:
                    self._table_edge(store, b, aid, NodeType.ATTRIBUTE.value, vid,
                                     store.get_node(vid)["node_type"], "same_table_column")
                    uid = col_unit.get(c.col)
                    if uid and is_numeric_cell(c.text):
                        # Value→Unit has_unit
                        self._table_edge(store, b, vid, NodeType.VALUE.value, uid,
                                         NodeType.UNIT.value, "same_table_column")
                # 同行：行标签 → 单元格
                if cid:
                    self._table_edge(store, b, cid, NodeType.CONCEPT.value, vid,
                                     store.get_node(vid)["node_type"], "same_table_row")
        return n_cells

    @staticmethod
    def _table_edge(
        store: GraphStore,
        b: EvidenceBlock,
        id1: str,
        t1: str,
        id2: str,
        t2: str,
        scope: str,
    ) -> None:
        r = resolve_edge(t1, t2)
        if r:
            sem, flip = r
            s, t = (id2, id1) if flip else (id1, id2)
        else:
            sem, s, t = "co_occurs_with", id1, id2
        store.upsert_edge(
            Edge(
                source=s,
                target=t,
                edge_type=sem,
                build_reason="physical_cooccurrence",
                cooccur_scope=scope,
                paper_id=b.address.paper_id,
                section_path=b.address.section_path,
                chunk_id=b.address.chunk_id,
            )
        )

    # ---------------- 横向共现图 (§四) ----------------
    def _extract_and_link(
        self,
        ctx: PipelineContext,
        store: GraphStore,
        blocks: List[EvidenceBlock],
        lang: str,
    ) -> int:
        # §四 / §十.3：文本块与图注块都抽点 (图注内的概念需与正文对齐)
        target_blocks = [
            b
            for b in blocks
            if b.modality in (Modality.TEXT.value, Modality.CAPTION.value)
            and (b.content or "").strip()
        ]
        if not target_blocks:
            return 0
        max_blocks = self.params.get("max_text_blocks")
        if max_blocks:
            # 限额只作用于正文文本块；图注块短且是跨模态对齐 (§十.3) 的必要输入，
            # 全部保留 (此前图注被截掉导致 aligns_with 恒为 0)
            text_blocks = [b for b in target_blocks if b.modality == Modality.TEXT.value]
            caption_blocks = [b for b in target_blocks if b.modality == Modality.CAPTION.value]
            target_blocks = text_blocks[: int(max_blocks)] + caption_blocks

        prompts = [
            build_prompt(b.content, b.modality, b.address.section_path, lang)
            for b in target_blocks
        ]
        self.log("LLM 点抽取: %d 个文本/图注块", len(prompts))
        responses = ctx.llm.generate_batch(prompts)

        total_points = 0
        for b, resp in zip(target_blocks, responses):
            points, relations = parse_extraction(resp)
            if not points:
                continue
            # 容器节点：文本块挂到 chunk 节点；图注块挂到 caption 节点
            container = (
                b.address.chunk_id
                if b.modality == Modality.TEXT.value
                else b.block_id
            )
            sentences = (
                b.extra.get("sentence_list", []) if isinstance(b.extra, dict) else []
            )
            name_to_id: Dict[str, Tuple[str, str]] = {}
            for i, p in enumerate(points):
                ntype = LABEL_TO_NODE_TYPE.get(p["type"].strip().lower())
                if not ntype:
                    continue
                # 质量过滤：丢弃符号噪声 / 通用词 / 元数据点 (强化 benchmark)
                if not is_valid_point(ntype, p["name"], p["content"]):
                    continue
                node_id = f"{container}::p{i}"
                # §一 可选细粒度地址：尽量定位点所在句
                addr = dataclasses.replace(
                    b.address, sentence_id=_locate_sentence(p["content"], sentences)
                )
                store.upsert_node(
                    PointNode(
                        node_id=node_id,
                        node_type=ntype,
                        content=p["content"],
                        normalized_content=p["name"],
                        address=addr,
                        modality=b.modality,
                        canonical_id=_canonical(ntype, p["name"]),
                    )
                )
                name_to_id[p["name"].strip().lower()] = (node_id, ntype)
                if store.has_node(container):
                    store.upsert_edge(
                        Edge(
                            source=container,
                            target=node_id,
                            edge_type="contains",
                            build_reason="document_structure",
                            paper_id=b.address.paper_id,
                            chunk_id=b.address.chunk_id,
                        )
                    )
                total_points += 1

            self._add_cooccur_edges(store, b, points, relations, name_to_id, sentences)

        return total_points

    def _add_cooccur_edges(
        self,
        store: GraphStore,
        block: EvidenceBlock,
        points: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        name_to_id: Dict[str, Tuple[str, str]],
        sentences: List[str],
    ) -> None:
        is_caption = block.modality == Modality.CAPTION.value

        def scope(c1: str, c2: str) -> str:
            if is_caption:
                return "same_caption"
            for s in sentences:
                if c1 and c2 and c1[:20] in s and c2[:20] in s:
                    return "same_sentence"
            return "same_chunk"

        def add(id1: str, t1: str, id2: str, t2: str, sc: str, fallback: str) -> None:
            """按 §三 规范化方向后落库；m5 据方向遍历不漏边。"""
            r = resolve_edge(t1, t2)
            if r:
                sem, flip = r
                s, t = (id2, id1) if flip else (id1, id2)
            else:
                sem, s, t = fallback, id1, id2
            store.upsert_edge(
                Edge(
                    source=s,
                    target=t,
                    edge_type=sem,
                    build_reason="physical_cooccurrence",
                    cooccur_scope=sc,
                    paper_id=block.address.paper_id,
                    section_path=block.address.section_path,
                    chunk_id=block.address.chunk_id,
                    evidence_span=block.content[:200],
                )
            )

        linked: set = set()
        for r in relations:
            src = name_to_id.get(r["source"].strip().lower())
            tgt = name_to_id.get(r["target"].strip().lower())
            if not src or not tgt:
                continue
            sc = scope(_content_of(points, r["source"]), _content_of(points, r["target"]))
            add(src[0], src[1], tgt[0], tgt[1], sc, r.get("relation") or "co_occurs_with")
            linked.add(frozenset((src[0], tgt[0])))

        # 规则补全：同块内符合标签规则但 LLM 未连的点对
        ids = list(name_to_id.values())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                (id1, t1), (id2, t2) = ids[i], ids[j]
                if frozenset((id1, id2)) in linked:
                    continue
                if resolve_edge(t1, t2):
                    add(id1, t1, id2, t2, "same_caption" if is_caption else "same_chunk", "co_occurs_with")

    # ---------------- 跨模态引用边 (§十.1) ----------------
    def _cross_modal_edges(self, store: GraphStore, blocks: List[EvidenceBlock]) -> int:
        label_idx: Dict[Tuple[str, str, str], str] = {}
        for n, d in store.all_nodes():
            nt = d.get("node_type")
            if nt in (NodeType.FIGURE.value, NodeType.TABLE.value, NodeType.EQUATION.value):
                tag = str(d.get("domain_schema_tag", ""))
                paper = d.get("address", {}).get("paper_id", "")
                if tag:
                    label_idx[(paper, nt, tag)] = n

        count = 0
        for b in blocks:
            if b.modality != Modality.TEXT.value:
                continue
            chunk_id = b.address.chunk_id
            if not store.has_node(chunk_id):
                continue
            paper = b.address.paper_id
            for refs, ntype, sc in (
                (b.figure_refs, NodeType.FIGURE.value, "figure_text_reference"),
                (b.table_refs, NodeType.TABLE.value, "table_text_reference"),
                (b.equation_refs, NodeType.EQUATION.value, "equation_text_reference"),
            ):
                for ref in refs:
                    target = label_idx.get((paper, ntype, str(ref)))
                    if target:
                        store.upsert_edge(
                            Edge(
                                source=chunk_id,
                                target=target,
                                edge_type="references",
                                build_reason="cross_modal_reference",
                                cooccur_scope=sc,
                                paper_id=paper,
                                chunk_id=chunk_id,
                            )
                        )
                        count += 1
        return count

    # ---------------- 跨模态对象对齐 (§十.3) ----------------
    def _intra_paper_alignment(self, store: GraphStore) -> int:
        """同一文献内、同 canonical 的 Concept/Attribute 若分布在不同模态/容器，
        建 aligns_with 边——把图注里的对象与正文里的同名对象合并为同一跨模态证据集合。
        """
        groups = store.index_by_canonical()
        count = 0
        for _cid, node_ids in groups.items():
            by_paper: Dict[str, List[str]] = defaultdict(list)
            for nid in node_ids:
                d = store.get_node(nid) or {}
                if d.get("node_type") not in (
                    NodeType.CONCEPT.value,
                    NodeType.ATTRIBUTE.value,
                ):
                    continue
                by_paper[d.get("address", {}).get("paper_id", "")].append(nid)
            for _paper, members in by_paper.items():
                # 跨不同模态或不同容器才对齐 (避免同块内重复连)
                modalities = {
                    (store.get_node(m) or {}).get("modality") for m in members
                }
                if len(members) >= 2 and len(modalities) >= 2:
                    for k in range(len(members) - 1):
                        store.upsert_edge(
                            Edge(
                                source=members[k],
                                target=members[k + 1],
                                edge_type="aligns_with",
                                build_reason="cross_modal_alignment",
                                weight=1.0,
                            )
                        )
                        count += 1
        return count

    # ---------------- 跨文献聚合 (§八) ----------------
    def _cross_paper_edges(self, store: GraphStore) -> int:
        groups = store.index_by_canonical()
        count = 0
        for _cid, node_ids in groups.items():
            by_paper: Dict[str, List[str]] = defaultdict(list)
            for nid in node_ids:
                d = store.get_node(nid) or {}
                if d.get("node_type") not in (
                    NodeType.CONCEPT.value,
                    NodeType.ATTRIBUTE.value,
                ):
                    continue
                by_paper[d.get("address", {}).get("paper_id", "")].append(nid)
            if len(by_paper) < 2:
                continue
            reps = [v[0] for v in by_paper.values()]
            for i in range(len(reps) - 1):
                store.upsert_edge(
                    Edge(
                        source=reps[i],
                        target=reps[i + 1],
                        edge_type="same_as",
                        build_reason="cross_paper",
                        weight=1.0,
                    )
                )
                count += 1
        return count

    def _mark_graph_status(self, ctx: PipelineContext) -> None:
        path = ctx.path("documents.jsonl")
        docs = read_jsonl_list(path)
        for d in docs:
            if d.get("parsing_status") in ("parsed", "low_quality"):
                d["graph_status"] = "done"
        if docs:
            write_jsonl(path, docs)


def _content_of(points: List[Dict[str, Any]], name: str) -> str:
    for p in points:
        if p["name"].strip().lower() == name.strip().lower():
            return p["content"]
    return ""


def _cell_addr(base: Address, row: int, col: int) -> Address:
    """表格单元格地址：复用表格块地址，用 span 记录行列坐标 (§一 细粒度地址)。"""
    return dataclasses.replace(base, span=f"r{row}c{col}")


def _contains(parent_id: str, child_id: str, base: Address) -> Edge:
    return Edge(
        source=parent_id,
        target=child_id,
        edge_type="contains",
        build_reason="document_structure",
        paper_id=base.paper_id,
        section_path=base.section_path,
        chunk_id=base.chunk_id,
    )
