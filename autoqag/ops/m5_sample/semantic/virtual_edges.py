"""问题级虚拟语义边构建与评分 (语义层子图构建.pdf §3.3 + 语义规划层方法论.pdf §3.6)。

虚拟边 = 语义覆盖层 Ωq 的元素，只服务问题规划，最终须回落到物理证据。
本模块负责:
- 按缺失角色 / 题型生成候选虚拟边 (针对缺失角色，而非随意加边)
- 问题级打分 Score_q(e)
- 证据可回落验证 (每条虚拟边须至少有一个 backing evidence path)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from autoqag.ops.m4_graph.quality import is_valid_point
from autoqag.ops.m5_sample.semantic import evidence_chain as ec
from autoqag.ops.m5_sample.semantic import roles as R
from autoqag.schema import InferenceOp, NodeType, QuestionType, VirtualEdge, VirtualEdgeType

# Score_q(e) 权重 (语义层子图构建.pdf §3.3.4)；可按题型覆写
_SCORE_WEIGHTS = {
    "goal_relevance": 1.0,
    "semantic_similarity": 0.8,
    "schema_compatibility": 1.0,
    "physical_backing": 1.2,
    "evidence_diversity": 0.8,
    "difficulty_gain": 0.6,
    "condition_coverage": 0.7,
    "shortcut_risk": 1.0,   # 惩罚
    "ambiguity_risk": 1.0,  # 惩罚
}

# 按题型覆写部分权重 (§3.3.4：比较题重 schema/condition；机制题重 backing/diversity)
_TYPE_WEIGHT_OVERRIDE = {
    QuestionType.COMPARATIVE.value: {"schema_compatibility": 1.3, "condition_coverage": 1.2},
    "mechanism": {"physical_backing": 1.5, "evidence_diversity": 1.1},
    QuestionType.CONDITION.value: {"condition_coverage": 1.4},
    "cross_paper": {"semantic_similarity": 1.1, "evidence_diversity": 1.2},
}


def _content(view, nid: str) -> str:
    d = view.nodes.get(nid, {})
    return (d.get("normalized_content") or d.get("content") or "").strip().lower()


def _token_overlap(a: str, b: str) -> float:
    sa = {t for t in a.replace("/", " ").split() if len(t) > 1}
    sb = {t for t in b.replace("/", " ").split() if len(t) > 1}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _same_chunk(view, a: str, b: str) -> bool:
    aa = view.nodes.get(a, {}).get("address", {})
    bb = view.nodes.get(b, {}).get("address", {})
    return bool(aa.get("chunk_id")) and aa.get("chunk_id") == bb.get("chunk_id")


def _paper(view, nid: str) -> str:
    return view.nodes.get(nid, {}).get("address", {}).get("paper_id", "")


# ---------------------------------------------------------------------------
# 候选虚拟边生成 (针对缺失角色 / 题型意图)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 核心推断边 always 铺设 (operational_flow.md §3.6.0)
# ---------------------------------------------------------------------------
def propose_core_edges(
    view,
    qtype: str,
    role_assignment: Dict[str, str],
    role_pool: Dict[str, List[str]],
) -> List[VirtualEdge]:
    """按题型核心推断算子铺设核心虚拟边 (即使最小角色集已填满也建)。

    核心边承载题目要求做的那一步推断 (COMPARISON / CAUSAL_CHAIN / ...)，其两端取自
    已绑定的角色节点；若角色尚空则跳过该核心边 (留待补全边或证据接地后由 planner 重试)。
    """
    spec = R.core_inference(qtype)
    op = str(spec.get("op", "") or "")
    edges: List[VirtualEdge] = []
    if not op:
        return edges
    for src_role, tgt_role, vtype, qrole in spec.get("edges", []):  # type: ignore[union-attr]
        src = role_assignment.get(src_role)
        tgt = role_assignment.get(tgt_role)
        if not src or not tgt or src == tgt:
            continue
        edges.append(
            VirtualEdge(
                source=src, target=tgt, virtual_type=vtype,
                question_role=qrole, inference_op=op, is_core=True,
                posited_relation=_posited(view, op, src, tgt, role_assignment),
                reason=f"核心推断边 ({op})：题型 {qtype} 要求的推理主干",
            )
        )
    return edges


def _posited(view, op: str, src: str, tgt: str, role_assignment: Dict[str, str]) -> str:
    """人读的被推断关系串 (诊断用)。"""
    s = _content(view, src) or src
    t = _content(view, tgt) or tgt
    attr = _content(view, role_assignment.get("shared_attribute", "")) if role_assignment.get("shared_attribute") else ""
    if op == InferenceOp.COMPARISON.value:
        return f"compare({s}, {t}){' on ' + attr if attr else ''}"
    if op == InferenceOp.CROSS_PAPER_ALIGN.value:
        return f"align&compare({s} @paperA, {t} @paperB)"
    if op == InferenceOp.CAUSAL_CHAIN.value:
        return f"{s} --causes--> {t}"
    if op == InferenceOp.CONDITION_BIND.value:
        return f"{s} holds only under {t}"
    if op == InferenceOp.VISUAL_GROUNDING.value:
        return f"{s} grounded by visual {t}"
    return f"{op}({s}, {t})"


# ---------------------------------------------------------------------------
# 缺失角色补全边 (operational_flow.md §3.6.1；原 propose_virtual_edges)
# ---------------------------------------------------------------------------
def propose_completion_edges(
    view,
    qtype: str,
    role_assignment: Dict[str, str],
    missing_roles: List[str],
    role_pool: Dict[str, List[str]],
    max_per_role: int = 3,
) -> List[VirtualEdge]:
    """根据缺失角色与题型，提出候选补全虚拟边 (operational_flow.md §3.6.1 表)。"""
    edges: List[VirtualEdge] = []
    core_op = R.core_inference_op(qtype)

    def add(src, tgt, vtype, role, reason):
        if not src or not tgt or src == tgt:
            return
        edges.append(
            VirtualEdge(
                source=src, target=tgt, virtual_type=vtype,
                question_role=role, reason=reason, inference_op=core_op,
            )
        )

    # 比较题：缺比较对象 / 对齐条件
    if qtype == QuestionType.COMPARATIVE.value:
        attr = role_assignment.get("shared_attribute")
        obj_a = role_assignment.get("object_A")
        if attr and ("object_B" in missing_roles or "value_B" in missing_roles):
            for cand in _comparable_objects(view, attr, obj_a, role_pool, max_per_role):
                add(attr, cand, VirtualEdgeType.SEARCH_COMPARABLE.value, "object_B",
                    "按共享属性/单位/条件寻找可比对象")
        if obj_a and role_assignment.get("object_B"):
            add(obj_a, role_assignment["object_B"], VirtualEdgeType.COMPARABLE.value,
                "comparison_criterion", "建立两个对象在共享指标上的可比关系")

    # 条件边界题：缺条件边界
    if qtype == QuestionType.CONDITION.value and "condition_boundary" in missing_roles:
        anchor = role_assignment.get("claim") or role_assignment.get("attribute_or_result")
        for cond in role_pool.get("candidate_conditions", [])[:max_per_role]:
            add(anchor, cond, VirtualEdgeType.LIMITED_BY.value, "condition_boundary",
                "从 Claim/Value 周围寻找 ConditionNode 作适用边界")

    # 机制题：缺中间机制
    if qtype == "mechanism" and "intermediate_mechanism" in missing_roles:
        method = role_assignment.get("method_or_intervention")
        result = role_assignment.get("observed_result") or role_assignment.get("target_attribute")
        for mech in role_pool.get("candidate_claims", [])[:max_per_role]:
            add(method, mech, VirtualEdgeType.SEEK_MECHANISM.value, "intermediate_mechanism",
                "寻找同时连接方法和结果的 Claim/Mechanism")
        if method and result:
            add(method, result, VirtualEdgeType.METHOD_EFFECT.value, "observed_result",
                "连接方法与结果指标")

    # 跨文献题：缺跨文献实例
    if qtype == "cross_paper":
        canon = role_assignment.get("canonical_concept")
        if canon:
            for inst in _cross_paper_aligned(view, canon, max_per_role):
                add(canon, inst, VirtualEdgeType.CROSS_PAPER_ALIGN.value, "paper_B_instance",
                    "按 normalized concept 查找其他 paper_id 中的同类节点")

    # 图表证据缺失 (跨模态需求)：任意题型缺 evidence 时尝试图表
    if any(r.startswith("evidence") or r == "supporting_evidence" for r in missing_roles):
        anchor = (
            role_assignment.get("claim")
            or role_assignment.get("observed_result")
            or role_assignment.get("shared_attribute")
            or next(iter(role_assignment.values()), None)
        )
        for fig in (role_pool.get("candidate_figures", []) + role_pool.get("candidate_tables", []))[:max_per_role]:
            add(anchor, fig, VirtualEdgeType.NEED_VISUAL_EVIDENCE.value, "supporting_evidence",
                "通过正文引用和图注合并寻找 Figure/Table 证据")

    return edges


def _comparable_objects(view, attr: str, exclude: Optional[str], pool, k: int) -> List[str]:
    """找共享同一 (canonical) 属性的其它 Concept 作可比对象。"""
    out: List[str] = []
    cid = view.nodes.get(attr, {}).get("canonical_id", "")
    # 同 canonical 属性下的概念
    for c in pool.get("candidate_objects", []):
        if c == exclude:
            continue
        # 概念是否经 has_attribute 指向同名属性
        for t, d in view.out.get(c, []):
            if d.get("edge_type") == "has_attribute":
                a = view.nodes.get(t, {})
                if (cid and a.get("canonical_id") == cid) or _token_overlap(
                    _content(view, t), _content(view, attr)
                ) > 0.5:
                    out.append(c)
                    break
        if len(out) >= k:
            break
    return out


def _cross_paper_aligned(view, canon: str, k: int) -> List[str]:
    cid = view.nodes.get(canon, {}).get("canonical_id", "")
    src_paper = _paper(view, canon)
    if not cid:
        return []
    out = []
    for nid, d in view.nodes.items():
        if nid == canon:
            continue
        if d.get("canonical_id") == cid and _paper(view, nid) != src_paper:
            out.append(nid)
            if len(out) >= k:
                break
    return out


# ---------------------------------------------------------------------------
# 问题级打分 Score_q(e) (§3.3.4)
# ---------------------------------------------------------------------------
def score_edge(view, edge: VirtualEdge, qtype: str, difficulty: str) -> float:
    weights = dict(_SCORE_WEIGHTS)
    weights.update(_TYPE_WEIGHT_OVERRIDE.get(qtype, {}))

    sc, tc = _content(view, edge.source), _content(view, edge.target)
    goal_relevance = 1.0 if edge.question_role else 0.3
    semantic_similarity = _token_overlap(sc, tc)
    schema_compatibility = _schema_compatible(view, edge)
    physical_backing = 1.0 if edge.backing_evidence_paths else 0.0
    # 证据多样性：跨 chunk / 跨 paper
    evidence_diversity = 0.0
    if not _same_chunk(view, edge.source, edge.target):
        evidence_diversity += 0.5
    if _paper(view, edge.source) != _paper(view, edge.target):
        evidence_diversity += 0.5
    # 难度增益：高难度题更看重跨证据
    difficulty_gain = {"L1": 0.0, "L2": 0.3, "L3": 0.6, "L4": 1.0}.get(difficulty, 0.3) * evidence_diversity
    # 条件覆盖：优先用条件兼容门结果 (operational_flow.md §3.6)；无则退回 target 是否 Condition
    compat_status = edge.compatibility.get("status") if edge.compatibility else None
    if compat_status == "ok":
        condition_coverage = 1.0
    elif compat_status == "weak":
        condition_coverage = 0.5
    elif compat_status == "conflict":
        condition_coverage = 0.0
    else:
        condition_coverage = 1.0 if view.nodes.get(edge.target, {}).get("node_type") == NodeType.CONDITION.value else 0.0
    shortcut_risk = 1.0 if _same_chunk(view, edge.source, edge.target) else 0.0
    ambiguity_risk = 1.0 - schema_compatibility

    score = (
        weights["goal_relevance"] * goal_relevance
        + weights["semantic_similarity"] * semantic_similarity
        + weights["schema_compatibility"] * schema_compatibility
        + weights["physical_backing"] * physical_backing
        + weights["evidence_diversity"] * evidence_diversity
        + weights["difficulty_gain"] * difficulty_gain
        + weights["condition_coverage"] * condition_coverage
        - weights["shortcut_risk"] * shortcut_risk
        - weights["ambiguity_risk"] * ambiguity_risk
    )
    return round(score, 4)


# schema 兼容性矩阵 (§3.3.3 第二点)
_COMPATIBLE = {
    NodeType.METHOD.value: {NodeType.CLAIM.value, NodeType.ATTRIBUTE.value, NodeType.CONCEPT.value},
    NodeType.ATTRIBUTE.value: {NodeType.VALUE.value, NodeType.UNIT.value, NodeType.CONDITION.value, NodeType.CONCEPT.value},
    NodeType.CONCEPT.value: {NodeType.ATTRIBUTE.value, NodeType.CONCEPT.value, NodeType.CLAIM.value},
    NodeType.CLAIM.value: {NodeType.CONDITION.value, NodeType.FIGURE.value, NodeType.TABLE.value, NodeType.EVIDENCE.value, NodeType.VALUE.value},
}


def _schema_compatible(view, edge: VirtualEdge) -> float:
    st = view.nodes.get(edge.source, {}).get("node_type", "")
    tt = view.nodes.get(edge.target, {}).get("node_type", "")
    if tt in _COMPATIBLE.get(st, set()) or st in _COMPATIBLE.get(tt, set()):
        return 1.0
    return 0.3


# ---------------------------------------------------------------------------
# 证据可回落验证 (§3.4.2)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 证据可回落验证 + 条件兼容门 + 核心边反捷径 (operational_flow.md §3.6)
# ---------------------------------------------------------------------------
_COMPAT_OPS = {InferenceOp.COMPARISON.value, InferenceOp.CROSS_PAPER_ALIGN.value}


def validate_backing(
    view,
    edge: VirtualEdge,
    difficulty: str = "",
    compat_nodes: Optional[tuple] = None,
    max_depth: int = 4,
) -> VirtualEdge:
    """为虚拟边寻找物理凭据路径并施加三道门 (operational_flow.md §3.6)：

    1) 证据可回落 (硬条件)：凭据路径只走可作证据的正向边；找不到 → rejected。
    2) 条件兼容门 (硬门槛，仅 COMPARISON / CROSS_PAPER_ALIGN)：两操作数条件冲突 → rejected。
       compat_nodes=(node_a, node_b) 显式指定承载条件的操作数 (通常是 value_A/value_B)；
       缺省时退回用 edge.source / edge.target。
    3) 核心边反捷径 (硬门槛)：is_core 且目标难度 L3/L4 时，凭据触及 chunk < 2 → rejected。
    """
    path = ec.find_backing_path(view, edge.source, edge.target, max_depth=max_depth)
    if not path:
        edge.status = "rejected"
        return edge
    edge.backing_evidence_paths = [path]
    edge.required_physical_nodes = path
    edge.backing_chunks = ec.path_chunks(view, path)
    edge.status = "accepted"

    # 条件兼容门
    if edge.inference_op in _COMPAT_OPS:
        na, nb = compat_nodes if compat_nodes else (edge.source, edge.target)
        status, reason = ec.condition_compatible(view, na, nb)
        edge.compatibility = {"status": status, "reason": reason}
        if status == "conflict":
            edge.status = "rejected"
            return edge

    # 核心边反捷径：高难度核心推断边凭据必须跨 ≥2 chunk
    if edge.is_core and difficulty in ("L3", "L4") and len(edge.backing_chunks) < 2:
        edge.status = "rejected"
        edge.reason += " | 反捷径硬拒：高难度核心边凭据未跨 ≥2 chunk"
    return edge
