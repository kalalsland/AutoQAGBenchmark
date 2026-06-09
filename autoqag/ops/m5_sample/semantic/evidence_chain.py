"""候选节点扩展与证据链展开 (语义层子图构建.pdf §3.2)。

两个核心操作 (在固定物理证据图 G0 上，只读、不修改):
- downward_walk : 向下游走获取证据链
    High-level Node → Concept/Method/Claim → Attribute/Condition → Value/Unit
                    → Figure/Table/Equation → Evidence Span
- upward_locate : 向上定位确定语义归属 (避免数值孤立)
    Value/Unit → Attribute → Concept/Method → Chunk → Section → Paper
- build_role_pool : 形成候选角色池 RolePool(q)，后续虚拟边只在池内建立，
                    从而降低搜索空间并提高问题相关性。

所有函数对 `view` 只做读操作。view 需提供 .nodes(dict)/.out/.inc/.of_type/.datas，
即 m5_sample.sample._GraphView 的接口 (duck typing，避免循环依赖)。
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from autoqag.ops.m4_graph.quality import is_evidence_eligible
from autoqag.schema import NodeType

# 向下游走时优先经过的边 (从抽象到具体)
_DOWNWARD_EDGES = {
    "has_attribute", "has_value", "has_unit", "under_condition",
    "supports", "derived_from", "describes", "references", "has_caption",
}

# 节点类型的抽象层级 (数字越小越抽象/高层)；用于向下/向上判定
_LEVEL = {
    NodeType.PAPER.value: 0,
    NodeType.TITLE.value: 1,
    NodeType.SECTION.value: 2,
    NodeType.CHUNK.value: 3,
    NodeType.CLAIM.value: 4,
    NodeType.METHOD.value: 4,
    NodeType.CONCEPT.value: 4,
    NodeType.ATTRIBUTE.value: 5,
    NodeType.CONDITION.value: 6,
    NodeType.VALUE.value: 7,
    NodeType.UNIT.value: 8,
    NodeType.FIGURE.value: 6,
    NodeType.TABLE.value: 6,
    NodeType.EQUATION.value: 6,
    NodeType.CAPTION.value: 7,
    NodeType.EVIDENCE.value: 7,
}


def _level(view, nid: str) -> int:
    return _LEVEL.get(view.nodes.get(nid, {}).get("node_type", ""), 5)


def downward_walk(view, start: str, max_depth: int = 4, max_nodes: int = 40) -> List[str]:
    """从高层节点向下展开其完整证据链 (§3.2.1)。

    BFS，仅沿"更具体"方向 (目标层级 >= 当前) 或经下行语义边游走，收集证据链节点。
    """
    if start not in view.nodes:
        return []
    seen: Set[str] = {start}
    order: List[str] = [start]
    frontier = [start]
    for _ in range(max_depth):
        nxt: List[str] = []
        for cur in frontier:
            for t, d in view.out.get(cur, []):
                if t in seen:
                    continue
                et = d.get("edge_type", "")
                if et in _DOWNWARD_EDGES or _level(view, t) >= _level(view, cur):
                    seen.add(t)
                    order.append(t)
                    nxt.append(t)
                    if len(order) >= max_nodes:
                        return order
        frontier = nxt
        if not frontier:
            break
    return order


def upward_locate(view, start: str, max_depth: int = 6) -> List[str]:
    """从底层节点 (如 Value/Unit) 向上定位语义归属 (§3.2.2)。

    沿入边走向"更抽象"方向 (目标层级 <= 当前)，直到 Paper。返回从 start 到根的链。
    """
    if start not in view.nodes:
        return []
    chain = [start]
    cur = start
    seen = {start}
    for _ in range(max_depth):
        best = None
        for s, _d in view.inc.get(cur, []):
            if s in seen:
                continue
            if _level(view, s) <= _level(view, cur):
                best = s
                break
        if best is None:
            break
        chain.append(best)
        seen.add(best)
        cur = best
    return chain


def chunk_mates(view, nid: str, limit: int = 40) -> List[str]:
    """返回与 nid 同属一个 ChunkNode 的节点 (含该 ChunkNode 本身)。

    节点 address.chunk_id 即其所属 ChunkNode 的 node_id，因此同 chunk 的
    Claim/Condition/Value/Evidence 等可一并作为该问题的候选角色与物理证据。
    """
    cid = view.nodes.get(nid, {}).get("address", {}).get("chunk_id", "")
    if not cid:
        return []
    out: List[str] = []
    if cid in view.nodes:  # ChunkNode 本身即物理证据 span
        out.append(cid)
    for other, d in view.nodes.items():
        if other == nid or other == cid:
            continue
        if d.get("address", {}).get("chunk_id", "") == cid:
            out.append(other)
            if len(out) >= limit:
                break
    return out


def section_chunks(view, nid: str, limit: int = 6) -> List[str]:
    """返回与 nid 同 (paper_id, section_path) 的 ChunkNode。

    EquationNode/FigureNode/TableNode 在图中常近乎孤立 (chunk_id 指向 eq/fig
    伪块)，借同章节的正文 ChunkNode 作其物理证据邻域。
    """
    addr = view.nodes.get(nid, {}).get("address", {})
    pid, sec = addr.get("paper_id", ""), addr.get("section_path", "")
    if not pid or not sec:
        return []
    out: List[str] = []
    for other, d in view.nodes.items():
        if other == nid or d.get("node_type") != NodeType.CHUNK.value:
            continue
        a = d.get("address", {})
        if a.get("paper_id") == pid and a.get("section_path") == sec:
            out.append(other)
            if len(out) >= limit:
                break
    return out


def expand_candidates(view, seeds: List[str], max_total: int = 120) -> Set[str]:
    """围绕种子节点，结合向下游走、向上定位与同 chunk 邻居，得到候选节点集合。

    物理图中跨类型边稀疏 (如 EquationNode 几乎孤立、SectionNode 只连 ChunkNode)，
    而 chunk_id 是最可靠的连接组织，故额外:
    - 对到达的 ChunkNode 再取一层 chunk 成员，拉出其中的 Claim/Value/Condition；
    - 对孤立的 Equation/Figure/Table 种子，补入同章节正文 ChunkNode。
    """
    _MEDIA = {NodeType.EQUATION.value, NodeType.FIGURE.value, NodeType.TABLE.value}
    cands: Set[str] = set()
    for s in seeds:
        for n in downward_walk(view, s):
            cands.add(n)
        for n in upward_locate(view, s):
            cands.add(n)
        for n in chunk_mates(view, s):
            cands.add(n)
        if view.nodes.get(s, {}).get("node_type") in _MEDIA:
            for ch in section_chunks(view, s):
                cands.add(ch)
        if len(cands) >= max_total:
            break

    # 二级展开：把已到达 ChunkNode 的成员节点纳入 (Claim/Value/Condition 等)
    reached_chunks = [
        c for c in list(cands)
        if view.nodes.get(c, {}).get("node_type") == NodeType.CHUNK.value
    ][:8]
    for ch in reached_chunks:
        if len(cands) >= max_total:
            break
        for m in chunk_mates(view, ch, limit=20):
            cands.add(m)
    return cands


def build_role_pool(view, candidates: Set[str]) -> Dict[str, List[str]]:
    """形成候选角色池 RolePool(q) (§3.2.3)：按 NodeType 桶装候选节点。"""
    pool: Dict[str, List[str]] = {
        "candidate_objects": [],
        "candidate_attributes": [],
        "candidate_values": [],
        "candidate_units": [],
        "candidate_conditions": [],
        "candidate_methods": [],
        "candidate_claims": [],
        "candidate_figures": [],
        "candidate_tables": [],
        "candidate_equations": [],
        "candidate_evidence_spans": [],
    }
    bucket = {
        NodeType.CONCEPT.value: "candidate_objects",
        NodeType.ATTRIBUTE.value: "candidate_attributes",
        NodeType.VALUE.value: "candidate_values",
        NodeType.UNIT.value: "candidate_units",
        NodeType.CONDITION.value: "candidate_conditions",
        NodeType.METHOD.value: "candidate_methods",
        NodeType.CLAIM.value: "candidate_claims",
        NodeType.FIGURE.value: "candidate_figures",
        NodeType.TABLE.value: "candidate_tables",
        NodeType.EQUATION.value: "candidate_equations",
        NodeType.EVIDENCE.value: "candidate_evidence_spans",
        NodeType.CHUNK.value: "candidate_evidence_spans",
        NodeType.CAPTION.value: "candidate_evidence_spans",
    }
    for nid in candidates:
        nt = view.nodes.get(nid, {}).get("node_type", "")
        key = bucket.get(nt)
        if key:
            pool[key].append(nid)
    return pool


def _tok(s: str) -> Set[str]:
    return {t for t in (s or "").lower().replace(",", " ").replace("/", " ").split() if len(t) > 1}


def _content(view, nid: str) -> str:
    d = view.nodes.get(nid, {})
    return d.get("normalized_content") or d.get("content") or ""


def _edge_ok(d: Dict[str, Any]) -> bool:
    """边能否作正向答案证据：极性为正且置信度达标 (图谱构建.pdf 刀1)。

    被原文否定/对比/假设的共现边 (LLM 断言但极性非正) 即便仍在图中，
    也不能用于构造正向事实 (属性/数值/单位)。旧图无 polarity/confidence 字段时
    默认 positive/1.0，向后兼容。
    """
    return is_evidence_eligible(
        d.get("confidence", 1.0), d.get("polarity", "positive")
    )


def attributes_of(view, obj: str) -> List[str]:
    """obj --has_attribute--> Attribute 的属性节点列表 (仅取可作证据的正向边)。"""
    return [
        t for t, d in view.out.get(obj, [])
        if d.get("edge_type") == "has_attribute" and _edge_ok(d)
    ]


def values_via_attribute(
    view, obj: str, attr: str = "", limit: int = 12
) -> List[str]:
    """沿 obj --has_attribute--> A --has_value--> V 取属于该对象的数值节点。

    若给定 attr，则只经与 attr 同一节点或内容 token 重叠的属性 (保证取到的是
    "该对象在该共享指标上的值"，而非全局任意数值)，从而让 value 真正绑定 object。
    两跳均要求边可作证据 (正向 + 置信度达标)，避免被否定的 "B does not reach X" 入题。
    """
    attr_tok = _tok(_content(view, attr)) if attr else None
    out: List[str] = []
    for a, d in view.out.get(obj, []):
        if d.get("edge_type") != "has_attribute" or not _edge_ok(d):
            continue
        if attr:
            if a != attr and not (attr_tok and (_tok(_content(view, a)) & attr_tok)):
                continue
        for v, d2 in view.out.get(a, []):
            if d2.get("edge_type") == "has_value" and _edge_ok(d2) and \
                    view.nodes.get(v, {}).get("node_type") == NodeType.VALUE.value:
                out.append(v)
                if len(out) >= limit:
                    return out
    return out


def unit_of_value(view, value: str) -> str:
    """value --has_unit--> Unit；返回该数值真正的单位节点 (无则空串)。"""
    for u, d in view.out.get(value, []):
        if d.get("edge_type") == "has_unit" and _edge_ok(d) and \
                view.nodes.get(u, {}).get("node_type") == NodeType.UNIT.value:
            return u
    return ""


def column_unit(view, col: str) -> str:
    """表格列 (AttributeNode) --has_unit--> Unit；表格数值的单位挂在列头而非单元格。"""
    for u, d in view.out.get(col, []):
        if d.get("edge_type") == "has_unit" and \
                view.nodes.get(u, {}).get("node_type") == NodeType.UNIT.value:
            return u
    return ""


def table_row_columns(view, row: str) -> Dict[str, str]:
    """表格行 row 在各列上的取值：返回 {列节点(AttributeNode): 单元格(ValueNode)}。

    表结构：列 --has_value--> 单元格；行 --(compares/co_occurs_with)--> 单元格。
    取行邻接的每个单元格，回溯其所属列 (入边 has_value 来源)，得到 列→单元格 映射，
    用于"比较题对象为表行、指标为共享列、值为该行该列单元格"的语义绑定。
    """
    out: Dict[str, str] = {}
    for cell, _ in view.out.get(row, []):
        for src, d in view.inc.get(cell, []):
            if d.get("edge_type") == "has_value" and \
                    view.nodes.get(src, {}).get("node_type") == NodeType.ATTRIBUTE.value:
                out.setdefault(src, cell)
                break
    return out


def as_table_row(view, nid: str) -> str:
    """规范化为表行节点 (ConceptNode)，供比较题对象绑定。

    行标签单元格 (r{N}c1) 与行节点同为 ConceptNode，类型匹配可能误选标签单元格作对象。
    若 nid 本身即表行 (有指向单元格的 compares/co_occurs_with 出边) 则原样返回；
    否则经入边回溯到包含它的行节点；无法定位时返回原 nid。
    """
    if table_row_columns(view, nid):
        return nid
    for src, d in view.inc.get(nid, []):
        if d.get("edge_type") in ("compares", "co_occurs_with") and table_row_columns(view, src):
            return src
    return nid


def shared_metric(view, row_a: str, row_b: str, prefer: str = ""):
    """两表行共享的指标列及各自单元格：返回 (col, cell_a, cell_b)；无共享列返回 None。

    优先沿用已选 prefer 列，否则优先取数值型单元格的列 (真正可比较的定量指标)。
    """
    cols_a = table_row_columns(view, row_a)
    cols_b = table_row_columns(view, row_b)
    shared = [c for c in cols_a if c in cols_b]
    if not shared:
        return None
    col = prefer if prefer in shared else ""
    if not col:
        for c in shared:
            if any(ch.isdigit() for ch in _content(view, cols_a[c])):
                col = c
                break
        col = col or shared[0]
    return col, cols_a[col], cols_b[col]



def same_as_peers(view, nid: str, other_paper: bool = False) -> List[str]:
    """经 same_as / aligns_with 边 (双向) 关联的跨文献对齐实例。

    other_paper=True 时只返回不同 paper_id 的实例 (真正的跨文献对齐对)。
    """
    pid = view.nodes.get(nid, {}).get("address", {}).get("paper_id", "")
    peers: List[str] = []
    for t, d in view.out.get(nid, []):
        if d.get("edge_type") in ("same_as", "aligns_with"):
            peers.append(t)
    for s, d in view.inc.get(nid, []):
        if d.get("edge_type") in ("same_as", "aligns_with"):
            peers.append(s)
    if other_paper:
        peers = [
            p for p in peers
            if view.nodes.get(p, {}).get("address", {}).get("paper_id", "") != pid
        ]
    return list(dict.fromkeys(peers))


def find_backing_path(view, source: str, target: str, max_depth: int = 4) -> List[str]:
    """为一条虚拟边寻找物理证据回落路径 (§3.4.2)。

    在 G0 上 (无向意义) BFS 找 source→target 的物理/结构路径。找到则返回 node_id 列表，
    否则返回 []。这是虚拟边能否进入最终问题子图的硬条件。
    """
    if source == target:
        return [source]
    if source not in view.nodes or target not in view.nodes:
        return []
    # 双向邻接 (物理证据图的边方向不应限制证据可达性)
    prev: Dict[str, str] = {source: ""}
    frontier = [source]
    for _ in range(max_depth):
        nxt = []
        for cur in frontier:
            nbrs = [t for t, _ in view.out.get(cur, [])] + [s for s, _ in view.inc.get(cur, [])]
            for nb in nbrs:
                if nb in prev:
                    continue
                prev[nb] = cur
                if nb == target:
                    # 回溯
                    path = [target]
                    while prev[path[-1]]:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                nxt.append(nb)
        frontier = nxt
        if not frontier:
            break
    return []
