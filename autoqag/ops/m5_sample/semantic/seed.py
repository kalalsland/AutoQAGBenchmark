"""种子节点初始化与评分 (语义层子图构建.pdf §3.1.2 / §3.1.3)。

种子不是随机选择，而由问题逻辑规划驱动 + SeedScore 排序。
SeedScore(v) = α·domain_importance + β·evidence_richness + γ·node_level_weight
             + δ·under_coverage + ε·expert_priority + ζ·historical_success
             - λ·ambiguity_risk
"""

from __future__ import annotations

from typing import Any, Dict, List

from autoqag.ops.m4_graph.quality import is_symbolic_name, is_stopword_name
from autoqag.schema import NodeType

# 高层节点优先级 (node_level_weight)：高层节点更适合作复杂问题起点
_LEVEL_WEIGHT = {
    NodeType.CLAIM.value: 1.0,
    NodeType.METHOD.value: 0.95,
    NodeType.CONCEPT.value: 0.9,
    NodeType.ATTRIBUTE.value: 0.7,
    NodeType.SECTION.value: 0.6,
    NodeType.EQUATION.value: 0.55,
    NodeType.FIGURE.value: 0.5,
    NodeType.TABLE.value: 0.5,
    NodeType.CONDITION.value: 0.45,
    NodeType.VALUE.value: 0.3,
    NodeType.UNIT.value: 0.1,
}

_WEIGHTS = {
    "domain_importance": 1.0,
    "evidence_richness": 1.0,
    "node_level_weight": 0.8,
    "under_coverage": 0.5,
    "expert_priority": 0.6,
    "historical_success": 0.5,
    "ambiguity_risk": 1.0,  # 作为惩罚
}


def _evidence_richness(view, nid: str) -> float:
    """节点是否连接足够证据、条件、图表、数值 (邻居数归一)。"""
    deg = len(view.out.get(nid, [])) + len(view.inc.get(nid, []))
    return min(1.0, deg / 8.0)


def _ambiguity_risk(view, nid: str) -> float:
    d = view.nodes.get(nid, {})
    name = (d.get("normalized_content") or d.get("content") or "").strip()
    if not name or is_symbolic_name(name) or is_stopword_name(name):
        return 1.0
    if len(name) < 4:
        return 0.5
    return 0.0


def seed_score(
    view,
    nid: str,
    domain_tags: Dict[str, float] | None = None,
    under_coverage: float = 0.0,
    expert_priority: float = 0.0,
    historical_success: float = 0.0,
) -> float:
    d = view.nodes.get(nid, {})
    nt = d.get("node_type", "")
    tag = d.get("domain_schema_tag", "")
    domain_importance = 1.0 if tag else 0.3
    if domain_tags and tag in domain_tags:
        domain_importance = domain_tags[tag]
    feats = {
        "domain_importance": domain_importance,
        "evidence_richness": _evidence_richness(view, nid),
        "node_level_weight": _LEVEL_WEIGHT.get(nt, 0.4),
        "under_coverage": under_coverage,
        "expert_priority": expert_priority,
        "historical_success": historical_success,
        "ambiguity_risk": _ambiguity_risk(view, nid),
    }
    score = (
        _WEIGHTS["domain_importance"] * feats["domain_importance"]
        + _WEIGHTS["evidence_richness"] * feats["evidence_richness"]
        + _WEIGHTS["node_level_weight"] * feats["node_level_weight"]
        + _WEIGHTS["under_coverage"] * feats["under_coverage"]
        + _WEIGHTS["expert_priority"] * feats["expert_priority"]
        + _WEIGHTS["historical_success"] * feats["historical_success"]
        - _WEIGHTS["ambiguity_risk"] * feats["ambiguity_risk"]
    )
    return score


def rank_seeds(
    view,
    seed_node_types: List[str],
    top_k: int = 20,
    node_ok=None,
    memory_boost=None,
) -> List[str]:
    """按 SeedScore 对候选种子排序，返回 top_k。

    node_ok: 可选质量门函数 node_ok(view, nid)->bool。
    memory_boost: 可选 nid->historical_success 映射 (长期记忆)。
    """
    cands: List[str] = []
    for nt in seed_node_types:
        cands.extend(view.of_type(nt))
    scored: List[tuple] = []
    for nid in cands:
        if node_ok is not None and not node_ok(view, nid):
            continue
        hs = 0.0
        if memory_boost:
            hs = memory_boost.get(nid, 0.0)
        scored.append((seed_score(view, nid, historical_success=hs), nid))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [nid for _, nid in scored[:top_k]]
