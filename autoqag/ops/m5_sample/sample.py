"""模块四(下)：子图采样与 Question Plan 生成 (论文创新三 §5.4)。

把 QA 生成从"自由文本生成"转化为"图结构约束下的子图采样 + 问题规划"。
为 8 类题型各自定义子图采样模板 (论文 §5.4 列出的结构模式)，难度由结构变量决定。
不调用 LLM，纯图遍历，确保题型/难度/证据路径可控。输出 question_plans.jsonl。
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from autoqag.common.graph_store import GraphStore
from autoqag.common.io import write_jsonl
from autoqag.ops.base import BaseStage, PipelineContext
from autoqag.ops.m5_sample.planner import compute_difficulty, difficulty_features
from autoqag.ops.m4_graph.quality import is_valid_point
from autoqag.registry import STAGES
from autoqag.schema import NodeType, QuestionPlan, QuestionType


def _node_ok(v: "_GraphView", nid: str) -> bool:
    """采样级质量门：跳过符号噪声 / 通用词节点。"""
    d = v.nodes.get(nid, {})
    return is_valid_point(
        d.get("node_type", ""),
        d.get("normalized_content", "") or d.get("content", ""),
        d.get("content", ""),
    )


class _GraphView:
    """把 GraphStore 载入内存邻接，提供按边语义类型的定向遍历。"""

    def __init__(self, store: GraphStore):
        self.nodes: Dict[str, Dict[str, Any]] = {n: d for n, d in store.all_nodes()}
        self.out: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        self.inc: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        for s, t, d in store.all_edges():
            self.out[s].append((t, d))
            self.inc[t].append((s, d))

    def of_type(self, node_type: str) -> List[str]:
        return [n for n, d in self.nodes.items() if d.get("node_type") == node_type]

    def neighbors_by_edge(self, node_id: str, edge_type: str, direction: str = "out") -> List[str]:
        src = self.out if direction == "out" else self.inc
        return [t for t, d in src.get(node_id, []) if d.get("edge_type") == edge_type]

    def neighbors_by_target_type(
        self, node_id: str, target_type: str, direction: str = "out"
    ) -> List[str]:
        src = self.out if direction == "out" else self.inc
        return [
            t
            for t, _ in src.get(node_id, [])
            if self.nodes.get(t, {}).get("node_type") == target_type
        ]

    def datas(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        return [self.nodes[n] for n in node_ids if n in self.nodes]


@STAGES.register_module("sample")
class SampleStage(BaseStage):
    declared_inputs = ["nodes.jsonl", "edges.jsonl"]
    declared_outputs = ["question_plans.jsonl"]

    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        store = GraphStore.load_jsonl(ctx.path("nodes.jsonl"), ctx.path("edges.jsonl"))
        if store.node_count() == 0:
            self.log("图谱为空，先运行 graph")
            return {"plans": 0}

        view = _GraphView(store)
        per_type = int(self.params.get("per_type", 30))
        types = self.params.get(
            "types",
            [
                "atomic",
                "numerical",
                "condition",
                "comparative",
                "table",
                "formula",
                "multi_hop",
                "summary",
            ],
        )
        domain = ctx.global_params.get("domain", "")

        finders: Dict[str, Callable[[_GraphView, int], List[Dict[str, Any]]]] = {
            "atomic": _find_atomic,
            "numerical": _find_numerical,
            "condition": _find_condition,
            "comparative": _find_comparative,
            "table": _find_table_figure,
            "formula": _find_formula,
            "multi_hop": _find_multihop,
            "summary": _find_summary,
        }

        plans: List[QuestionPlan] = []
        counter = 0
        stats: Dict[str, int] = {}
        for qtype in types:
            finder = finders.get(qtype)
            if not finder:
                continue
            subgraphs = finder(view, per_type)
            for sg in subgraphs:
                counter += 1
                plan = _build_plan(f"q{counter:06d}", qtype, sg, view, domain)
                plans.append(plan)
            stats[qtype] = len(subgraphs)

        n = write_jsonl(ctx.path("question_plans.jsonl"), [p.to_dict() for p in plans])
        self.log("生成 %d 个 question plan: %s", n, stats)
        return {"plans": n, **stats}


# ----------------- 题型子图采样器 (论文 §5.4 结构模式) -----------------
def _find_atomic(v: _GraphView, k: int) -> List[Dict[str, Any]]:
    """L1：单个 Concept/Claim 节点 (跳过符号噪声，要求内容>=4字符)。"""
    out = []
    for nt in (NodeType.CONCEPT.value, NodeType.CLAIM.value, NodeType.METHOD.value):
        for n in v.of_type(nt):
            if not _node_ok(v, n):
                continue
            content = (v.nodes[n].get("normalized_content") or v.nodes[n].get("content") or "").strip()
            if len(content) < 4:  # 太短的概念产不出有意义的事实题
                continue
            out.append({"nodes": [n], "edges": [], "path_length": 0})
            if len(out) >= k:
                return out
    return out


def _find_numerical(v: _GraphView, k: int) -> List[Dict[str, Any]]:
    """Concept-Attribute-Value-Unit 链 (数值敏感题)，优先带单位的子图。"""
    with_unit, without_unit = [], []
    for attr in v.of_type(NodeType.ATTRIBUTE.value):
        if not _node_ok(v, attr):  # 通用列头 (Value/Parameters) 不出数值题
            continue
        concepts = [c for c in v.neighbors_by_target_type(attr, NodeType.CONCEPT.value, "inc") if _node_ok(v, c)]
        values = v.neighbors_by_target_type(attr, NodeType.VALUE.value, "out")
        for val in values:
            units = v.neighbors_by_target_type(val, NodeType.UNIT.value, "out")
            nodes = [attr, val] + units[:1] + concepts[:1]
            edges = [(attr, val)] + [(val, u) for u in units[:1]] + [(c, attr) for c in concepts[:1]]
            sg = {"nodes": nodes, "edges": edges, "path_length": len(nodes) - 1}
            (with_unit if units else without_unit).append(sg)
    # 带单位的优先 (数值题最有价值)
    return (with_unit + without_unit)[:k]


def _find_condition(v: _GraphView, k: int) -> List[Dict[str, Any]]:
    """Condition-Attribute/Value (条件边界题)。"""
    out = []
    for cond in v.of_type(NodeType.CONDITION.value):
        targets = [t for t, _ in v.out.get(cond, [])]
        for t in targets:
            out.append({"nodes": [cond, t], "edges": [(cond, t)], "path_length": 1})
            if len(out) >= k:
                return out
    return out


def _find_comparative(v: _GraphView, k: int) -> List[Dict[str, Any]]:
    """多个 Concept 共享同一 Attribute (比较题)，含跨文献 same_as。"""
    out = []
    # 按 Attribute 的 canonical 聚合不同 Concept
    attr_to_concepts: Dict[str, List[str]] = defaultdict(list)
    for attr in v.of_type(NodeType.ATTRIBUTE.value):
        cid = v.nodes[attr].get("canonical_id", attr)
        concepts = v.neighbors_by_target_type(attr, NodeType.CONCEPT.value, "inc")
        for c in concepts:
            attr_to_concepts[cid].append(c)
            attr_to_concepts[cid].append(attr)
    for cid, members in attr_to_concepts.items():
        concepts = [m for m in members if v.nodes.get(m, {}).get("node_type") == NodeType.CONCEPT.value]
        if len(set(concepts)) >= 2:
            uniq = list(dict.fromkeys(members))
            out.append({"nodes": uniq, "edges": [], "path_length": 2})
            if len(out) >= k:
                return out
    return out


def _find_table_figure(v: _GraphView, k: int) -> List[Dict[str, Any]]:
    """Table/Figure(+Caption) 及其支撑的 Claim / 描述的 Concept (图表证据题)。"""
    out = []
    for nt in (NodeType.TABLE.value, NodeType.FIGURE.value):
        for node in v.of_type(nt):
            nbrs = [t for t, _ in v.out.get(node, [])] + [s for s, _ in v.inc.get(node, [])]
            nodes = [node] + nbrs[:3]
            edges = [(node, t) for t, _ in v.out.get(node, [])][:3]
            out.append({"nodes": nodes, "edges": edges, "path_length": 1})
            if len(out) >= k:
                return out
    return out


def _find_formula(v: _GraphView, k: int) -> List[Dict[str, Any]]:
    """Equation-Attribute/Value (公式依赖题)。"""
    out = []
    for eq in v.of_type(NodeType.EQUATION.value):
        targets = [t for t, _ in v.out.get(eq, [])][:3]
        # 公式常被正文引用，取引用它的 chunk 作为上下文
        refs = [s for s, _ in v.inc.get(eq, [])][:1]
        nodes = [eq] + targets + refs
        out.append({"nodes": nodes, "edges": [(eq, t) for t in targets], "path_length": 1 + len(targets)})
        if len(out) >= k:
            return out
    return out


def _find_multihop(v: _GraphView, k: int) -> List[Dict[str, Any]]:
    """跨 chunk/section 的多跳路径 (多跳综合题)：从 Concept 出发 BFS 3-4 跳。"""
    out = []
    concepts = v.of_type(NodeType.CONCEPT.value)
    for start in concepts:
        path = _bfs_path(v, start, max_len=4)
        if len(path) >= 3:
            sections = {v.nodes[n].get("address", {}).get("section_path", "") for n in path}
            # 优先保留跨 section 的路径以体现难度
            edges = list(zip(path, path[1:]))
            out.append({"nodes": path, "edges": edges, "path_length": len(path) - 1})
            if len(out) >= k:
                return out
    return out


def _find_summary(v: _GraphView, k: int) -> List[Dict[str, Any]]:
    """章节级聚合 (综述/综合题)：Section 节点 + 其下 Claim/Concept。"""
    out = []
    for sec in v.of_type(NodeType.SECTION.value):
        members = []
        # section → chunk → point
        for chunk in v.neighbors_by_edge(sec, "contains", "out"):
            for t, d in v.out.get(chunk, []):
                if v.nodes.get(t, {}).get("node_type") in (
                    NodeType.CLAIM.value,
                    NodeType.CONCEPT.value,
                ):
                    members.append(t)
        if len(members) >= 2:
            nodes = [sec] + members[:5]
            out.append({"nodes": nodes, "edges": [], "path_length": 2})
            if len(out) >= k:
                return out
    return out


def _bfs_path(v: _GraphView, start: str, max_len: int = 4) -> List[str]:
    """从 start 出发取一条尽量长的简单路径 (优先跨 section)。"""
    path = [start]
    visited = {start}
    cur = start
    for _ in range(max_len - 1):
        cands = [t for t, _ in v.out.get(cur, [])] + [s for s, _ in v.inc.get(cur, [])]
        nxt = None
        for c in cands:
            if c not in visited:
                nxt = c
                break
        if nxt is None:
            break
        path.append(nxt)
        visited.add(nxt)
        cur = nxt
    return path


# ----------------- 组装 QuestionPlan -----------------
def _build_plan(
    qid: str,
    qtype: str,
    sg: Dict[str, Any],
    v: _GraphView,
    domain: str,
) -> QuestionPlan:
    node_ids = sg["nodes"]
    datas = v.datas(node_ids)
    feats = difficulty_features(datas, sg.get("path_length", 0))
    difficulty = compute_difficulty(feats)

    # 证据 span 与约束字段
    evidence_spans = [
        {
            "node_id": d_id,
            "content": v.nodes[d_id].get("content", "")[:200],
            "address": v.nodes[d_id].get("address", {}),
        }
        for d_id in node_ids
        if d_id in v.nodes
    ]
    constraints = {"number": [], "unit": [], "condition": [], "formula": [], "table": []}
    for d in datas:
        nt = d.get("node_type")
        if nt == NodeType.VALUE.value:
            constraints["number"].append(d.get("content", ""))
        elif nt == NodeType.UNIT.value:
            constraints["unit"].append(d.get("content", ""))
        elif nt == NodeType.CONDITION.value:
            constraints["condition"].append(d.get("content", ""))
        elif nt == NodeType.EQUATION.value:
            constraints["formula"].append(d.get("content", "")[:80])
        elif nt == NodeType.TABLE.value:
            constraints["table"].append(d.get("content", "")[:80])

    papers = list({d.get("address", {}).get("paper_id", "") for d in datas if d})

    return QuestionPlan(
        qid=qid,
        domain=domain,
        question_type=qtype,
        difficulty=difficulty,
        target_subgraph=node_ids,
        required_nodes=node_ids,
        required_edges=[list(e) for e in sg.get("edges", [])],
        evidence_spans=evidence_spans,
        constraints=constraints,
        expected_answer_form=_answer_form(qtype),
        forbidden_generalization=_forbidden(qtype, constraints),
        generation_instruction=_instruction(qtype, feats),
        paper_id_list=papers,
    )


def _answer_form(qtype: str) -> str:
    return {
        QuestionType.NUMERICAL.value: "数值 + 单位",
        QuestionType.CONDITION.value: "结论 + 显式条件限定",
        QuestionType.COMPARATIVE.value: "比较结论 (含对象、指标、条件)",
        QuestionType.MULTI_HOP.value: "结论 + evidence_path",
        QuestionType.FORMULA.value: "公式相关结论 (保留适用条件)",
    }.get(qtype, "简短事实结论")


def _forbidden(qtype: str, constraints: Dict[str, List[Any]]) -> List[str]:
    out = []
    if constraints["condition"]:
        out.append("不得省略条件限定，不得泛化到所有场景")
    if constraints["unit"]:
        out.append("不得更改或省略单位")
    return out


def _instruction(qtype: str, feats: Dict[str, Any]) -> str:
    base = {
        "atomic": "针对单一事实点生成一个简短问答。",
        "numerical": "生成数值敏感问答，答案必须包含原文数值与单位。",
        "condition": "生成条件边界问答，答案必须显式保留实验/边界条件。",
        "comparative": "生成比较问答，比较对象与指标需一致、条件需对齐。",
        "table": "生成图表证据问答，答案需引用正确图/表并与图注一致。",
        "formula": "生成公式依赖问答，需引用正确公式并保留适用条件。",
        "multi_hop": "生成多跳综合问答，答案需给出完整 evidence_path。",
        "summary": "生成章节级综合问答，整合该区域的多个结论。",
    }.get(qtype, "生成问答。")
    if feats["cross_paper"]:
        base += " (跨文献：注意区分不同论文来源)"
    return base
