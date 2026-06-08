"""综合子图评分与逻辑充分性验证 (语义规划层方法论.pdf §3.4-3.5；语义层子图构建.pdf §3.4)。

Score(S_q | T, D) =
    a1·SemanticScore + a2·RoleCompleteness + a3·EvidenceSufficiency
  + a4·ConstraintCoverage + a5·StructuralCoherence + a6·DifficultyMatch
  - b1·ShortcutPenalty - b2·NoisePenalty

LogicalSufficiency(S_q) = True iff 各分量分别 >= / <= 阈值。
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from autoqag.ops.m4_graph.quality import is_valid_point
from autoqag.ops.m5_sample.semantic import roles as R
from autoqag.schema import NodeType, QuestionType

# 综合评分权重
_A = {
    "semantic": 0.8,
    "role": 1.5,
    "evidence": 1.3,
    "constraint": 1.0,
    "structure": 0.7,
    "difficulty": 0.8,
}
_B = {"shortcut": 1.2, "noise": 0.8}

# 逻辑充分性阈值 (§3.5)
THRESHOLDS = {
    "tau_role": 0.99,        # 最小角色集合须全部填满
    "tau_evidence": 0.6,
    "tau_constraint": 0.5,
    "tau_structure": 0.5,
    "tau_diff": 0.5,
    "tau_shortcut": 0.5,
}

# 只有这些题型把"约束覆盖 (数值/单位/条件)"作为硬门槛 (§3.4.4)。
# atomic/mechanism/multi_hop/summary/cross_paper 等以语义角色为主，
# 约束覆盖仅作软性加分，不应一刀切地卡掉本不含数值的合法问题。
_CONSTRAINT_REQUIRED = {
    QuestionType.NUMERICAL.value,
    QuestionType.FORMULA.value,
    QuestionType.COMPARATIVE.value,
    QuestionType.TABLE.value,
}

# 难度 → 期望的最小 (chunk 数, evidence span 数, 物理路径长)
_DIFF_REQ = {
    "L1": (1, 1, 0),
    "L2": (1, 2, 1),
    "L3": (2, 3, 2),
    "L4": (3, 4, 3),
}


def role_completeness(qtype: str, role_assignment: Dict[str, str]) -> float:
    """填满的最小角色数 / 最小角色总数 (§3.4.2)。"""
    req = R.min_roles(qtype)
    if not req:
        return 1.0
    filled = sum(1 for r in req if r in R.ABSTRACT_ROLES or role_assignment.get(r))
    return filled / len(req)


def missing_roles(qtype: str, role_assignment: Dict[str, str]) -> List[str]:
    return [
        r for r in R.min_roles(qtype)
        if r not in R.ABSTRACT_ROLES and not role_assignment.get(r)
    ]


def evidence_sufficiency(view, qtype: str, role_assignment: Dict[str, str]) -> float:
    """每个关键角色是否能回到物理证据节点 / span (§3.4.3)。"""
    req = [r for r in R.min_roles(qtype) if r not in R.ABSTRACT_ROLES]
    key = [r for r in req if R.is_evidence_role(r) or _is_concrete(view, role_assignment.get(r))]
    if not key:
        key = req
    if not key:
        return 1.0
    grounded = sum(1 for r in key if _grounded(view, role_assignment.get(r)))
    return grounded / len(key)


def _is_concrete(view, nid) -> bool:
    if not nid:
        return False
    nt = view.nodes.get(nid, {}).get("node_type", "")
    return nt in (
        NodeType.VALUE.value, NodeType.EVIDENCE.value, NodeType.FIGURE.value,
        NodeType.TABLE.value, NodeType.CHUNK.value,
    )


def _grounded(view, nid) -> bool:
    """该节点是否带物理地址 (chunk_id / span) 可追溯。"""
    if not nid or nid not in view.nodes:
        return False
    addr = view.nodes[nid].get("address", {})
    return bool(addr.get("chunk_id") or addr.get("span") or addr.get("section_path"))


def constraint_coverage(view, node_ids: List[str]) -> float:
    """数值/单位/条件/公式是否被纳入子图 (§3.4.4)。"""
    have = set()
    for nid in node_ids:
        nt = view.nodes.get(nid, {}).get("node_type", "")
        if nt == NodeType.VALUE.value:
            have.add("number")
        elif nt == NodeType.UNIT.value:
            have.add("unit")
        elif nt == NodeType.CONDITION.value:
            have.add("condition")
        elif nt == NodeType.EQUATION.value:
            have.add("formula")
    # 至少含数值或条件之一即视为基本覆盖；全含为满分
    target = {"number", "unit", "condition"}
    return len(have & target) / len(target) if target else 1.0


def semantic_score(view, node_ids: List[str], theme: str) -> float:
    """高级节点聚合语义评分：与主题 token 重叠 (轻量近似，避免引入向量依赖)。"""
    if not theme:
        return 0.5
    theme_tokens = {t for t in theme.lower().split() if len(t) > 1}
    if not theme_tokens:
        return 0.5
    hit = 0
    total = 0
    for nid in node_ids:
        c = (view.nodes.get(nid, {}).get("normalized_content")
             or view.nodes.get(nid, {}).get("content") or "").lower()
        toks = {t for t in c.split() if len(t) > 1}
        if not toks:
            continue
        total += 1
        if theme_tokens & toks:
            hit += 1
    return hit / total if total else 0.5


def structural_coherence(view, node_ids: List[str], edges: List[Any]) -> float:
    """子图是否形成连通逻辑链，而非散点 (§3.4.5)。"""
    if len(node_ids) <= 1:
        return 1.0
    adj: Dict[str, Set[str]] = {n: set() for n in node_ids}
    nodeset = set(node_ids)
    # 用 G0 上节点间真实边判定连通
    for n in node_ids:
        for t, _ in view.out.get(n, []):
            if t in nodeset:
                adj[n].add(t)
                adj[t].add(n)
        for s, _ in view.inc.get(n, []):
            if s in nodeset:
                adj[n].add(s)
                adj[s].add(n)
    # 最大连通分量占比
    seen: Set[str] = set()
    best = 0
    for start in node_ids:
        if start in seen:
            continue
        comp = 0
        stack = [start]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp += 1
            stack.extend(adj[x] - seen)
        best = max(best, comp)
    return best / len(node_ids)


def difficulty_match(view, node_ids: List[str], difficulty: str, path_length: int) -> float:
    """子图是否符合目标难度 (跨 chunk / 多 span / 路径长)。"""
    req_chunks, req_spans, req_path = _DIFF_REQ.get(difficulty, (1, 1, 0))
    chunks = {view.nodes.get(n, {}).get("address", {}).get("chunk_id", "") for n in node_ids}
    chunks.discard("")
    spans = sum(1 for n in node_ids if _grounded(view, n))
    s_chunk = min(1.0, len(chunks) / req_chunks) if req_chunks else 1.0
    s_span = min(1.0, spans / req_spans) if req_spans else 1.0
    s_path = min(1.0, path_length / req_path) if req_path else 1.0
    return (s_chunk + s_span + s_path) / 3.0


def shortcut_penalty(view, node_ids: List[str], difficulty: str) -> float:
    """单 chunk shortcut 惩罚 (§3.4.5 / 双重多跳)。高难度题集中于单 chunk 时重罚。"""
    chunks = {view.nodes.get(n, {}).get("address", {}).get("chunk_id", "") for n in node_ids}
    chunks.discard("")
    if difficulty in ("L3", "L4") and len(chunks) <= 1:
        return 1.0
    if difficulty == "L2" and len(chunks) == 0:
        return 0.5
    return 0.0


def noise_penalty(view, node_ids: List[str]) -> float:
    """子图中低质量 / 符号噪声节点占比。"""
    if not node_ids:
        return 0.0
    bad = 0
    for n in node_ids:
        d = view.nodes.get(n, {})
        if not is_valid_point(
            d.get("node_type", ""),
            d.get("normalized_content", "") or d.get("content", ""),
            d.get("content", ""),
        ):
            bad += 1
    return bad / len(node_ids)


def utility_score(
    view,
    qtype: str,
    difficulty: str,
    node_ids: List[str],
    edges: List[Any],
    role_assignment: Dict[str, str],
    theme: str = "",
    path_length: int = 0,
) -> Dict[str, float]:
    """综合子图评分，返回分项 + total (Subgraph Utility Score)。"""
    parts = {
        "semantic": semantic_score(view, node_ids, theme),
        "role": role_completeness(qtype, role_assignment),
        "evidence": evidence_sufficiency(view, qtype, role_assignment),
        "constraint": constraint_coverage(view, node_ids),
        "structure": structural_coherence(view, node_ids, edges),
        "difficulty": difficulty_match(view, node_ids, difficulty, path_length),
        "shortcut": shortcut_penalty(view, node_ids, difficulty),
        "noise": noise_penalty(view, node_ids),
    }
    total = (
        _A["semantic"] * parts["semantic"]
        + _A["role"] * parts["role"]
        + _A["evidence"] * parts["evidence"]
        + _A["constraint"] * parts["constraint"]
        + _A["structure"] * parts["structure"]
        + _A["difficulty"] * parts["difficulty"]
        - _B["shortcut"] * parts["shortcut"]
        - _B["noise"] * parts["noise"]
    )
    parts["total"] = round(total, 4)
    return parts


def logical_sufficiency(parts: Dict[str, float], qtype: str = "") -> bool:
    """逻辑充分性验证 (§3.5)。约束覆盖仅对数值/公式/比较/图表题作硬门槛。"""
    constraint_ok = (
        parts.get("constraint", 0) >= THRESHOLDS["tau_constraint"]
        if qtype in _CONSTRAINT_REQUIRED
        else True
    )
    return (
        parts.get("role", 0) >= THRESHOLDS["tau_role"]
        and parts.get("evidence", 0) >= THRESHOLDS["tau_evidence"]
        and constraint_ok
        and parts.get("structure", 0) >= THRESHOLDS["tau_structure"]
        and parts.get("difficulty", 0) >= THRESHOLDS["tau_diff"]
        and parts.get("shortcut", 1) <= THRESHOLDS["tau_shortcut"]
    )


def dual_multihop_ok(view, node_ids: List[str], difficulty: str, semantic_path_len: int) -> bool:
    """双重多跳有效性：语义多跳 + 物理多跳 (§3.4.4)。

    高难度 (L3/L4) 须跨多 chunk 且语义路径足够长，避免伪多跳。
    """
    if difficulty not in ("L3", "L4"):
        return True
    chunks = {view.nodes.get(n, {}).get("address", {}).get("chunk_id", "") for n in node_ids}
    chunks.discard("")
    ks = 2 if difficulty == "L3" else 3
    return len(chunks) >= 2 and semantic_path_len >= ks
