"""内部 (intrinsic) 指标：只依赖 question_plans.jsonl + 图谱，无需 LLM。

这些指标度量"子图规划"本身的质量，用于消融 (A0..A6) 证明各模块逐步有效。
所有指标可复现、确定性，不消耗 API。用法见 run_ablation.py。

指标分组：
  覆盖/结构  : n_plans / type_coverage / difficulty_dist / avg_evidence_spans
  多跳真实性 : real_cross_chunk_ratio / pseudo_multihop_rate / cross_paper_ratio
  逻辑完整性 : role_completeness / overlay_grounded_ratio / avg_utility(+breakdown)
  语义绑定   : comparative_value_object_bind / cross_paper_result_cross / unit_grounded
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from autoqag.common.graph_store import GraphStore
from autoqag.ops.m5_sample.sample import _GraphView, _node_ok
from autoqag.ops.m5_sample.semantic import evidence_chain as ec
from autoqag.ops.m5_sample.semantic import roles as R
from autoqag.schema import QuestionType, Difficulty


# ---------------------------------------------------------------------- #
# 基础工具
# ---------------------------------------------------------------------- #
def load_view(work_dir: str) -> _GraphView:
    import os

    store = GraphStore.load_jsonl(
        os.path.join(work_dir, "nodes.jsonl"), os.path.join(work_dir, "edges.jsonl")
    )
    return _GraphView(store)


def _chunk_id(view: _GraphView, nid: str) -> str:
    return view.nodes.get(nid, {}).get("address", {}).get("chunk_id", "")


def _paper_id(view: _GraphView, nid: str) -> str:
    return view.nodes.get(nid, {}).get("address", {}).get("paper_id", "")


def _n_chunks(view: _GraphView, node_ids: List[str]) -> int:
    cs = {_chunk_id(view, n) for n in node_ids}
    cs.discard("")
    return len(cs)


def _n_papers(view: _GraphView, node_ids: List[str]) -> int:
    ps = {_paper_id(view, n) for n in node_ids}
    ps.discard("")
    return len(ps)


def _avg(xs: List[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 3) if xs else None


# ---------------------------------------------------------------------- #
# 覆盖 / 结构
# ---------------------------------------------------------------------- #
def coverage_structure(plans: List[Dict[str, Any]], view: _GraphView) -> Dict[str, Any]:
    by_type: Dict[str, int] = defaultdict(int)
    by_diff: Dict[str, int] = defaultdict(int)
    ev_spans: List[float] = []
    nodes_per: List[float] = []
    for p in plans:
        by_type[p.get("question_type", "")] += 1
        by_diff[p.get("difficulty", "")] += 1
        ev_spans.append(len(p.get("evidence_spans", [])))
        nodes_per.append(len(p.get("required_nodes", [])))
    all_types = [t.value for t in QuestionType if not t.value.startswith("virtual_")]
    covered = sum(1 for t in all_types if by_type.get(t, 0) > 0)
    return {
        "n_plans": len(plans),
        "n_types_covered": covered,
        "type_coverage": round(covered / len(all_types), 3),
        "by_type": dict(by_type),
        "by_difficulty": {d: by_diff.get(d, 0) for d in [x.value for x in Difficulty]},
        "avg_evidence_spans": _avg(ev_spans),
        "avg_required_nodes": _avg(nodes_per),
    }


# ---------------------------------------------------------------------- #
# 多跳真实性
# ---------------------------------------------------------------------- #
def multihop_authenticity(plans: List[Dict[str, Any]], view: _GraphView) -> Dict[str, Any]:
    cross_chunk = 0
    cross_paper = 0
    hard = [p for p in plans if p.get("difficulty") in ("L3", "L4")]
    pseudo = 0  # 标 L3/L4 但实际单 chunk
    for p in plans:
        nodes = p.get("required_nodes", [])
        if _n_chunks(view, nodes) >= 2:
            cross_chunk += 1
        if _n_papers(view, nodes) >= 2:
            cross_paper += 1
    for p in hard:
        if _n_chunks(view, p.get("required_nodes", [])) < 2:
            pseudo += 1
    n = max(1, len(plans))
    return {
        "real_cross_chunk_ratio": round(cross_chunk / n, 3),
        "cross_paper_ratio": round(cross_paper / n, 3),
        "n_cross_paper_plans": cross_paper,
        "hard_plans": len(hard),
        "pseudo_multihop_rate": round(pseudo / max(1, len(hard)), 3),
    }


# ---------------------------------------------------------------------- #
# 逻辑完整性
# ---------------------------------------------------------------------- #
def logical_completeness(plans: List[Dict[str, Any]], view: _GraphView) -> Dict[str, Any]:
    role_fill: List[float] = []
    util: List[float] = []
    breakdown: Dict[str, List[float]] = defaultdict(list)
    overlay_accepted: List[float] = []
    overlay_grounded: List[float] = []
    for p in plans:
        ra = p.get("role_assignment", {}) or {}
        req = p.get("required_roles") or R.min_roles(p.get("question_type", ""))
        req = [r for r in req if r not in R.ABSTRACT_ROLES]
        if req:
            filled = sum(1 for r in req if ra.get(r))
            role_fill.append(filled / len(req))
        if p.get("utility_score") is not None:
            util.append(float(p.get("utility_score", 0.0)))
        for k, v in (p.get("score_breakdown", {}) or {}).items():
            if isinstance(v, (int, float)):
                breakdown[k].append(float(v))
        edges = p.get("semantic_overlay_edges", []) or []
        acc = [e for e in edges if e.get("status") == "accepted"]
        overlay_accepted.append(len(acc))
        if acc:
            grounded = sum(1 for e in acc if e.get("backing_evidence_paths"))
            overlay_grounded.append(grounded / len(acc))
    return {
        "role_completeness": _avg(role_fill),
        "avg_utility": _avg(util),
        "utility_breakdown": {k: _avg(v) for k, v in breakdown.items()},
        "avg_overlay_edges": _avg(overlay_accepted),
        "overlay_grounded_ratio": _avg(overlay_grounded),
    }


# ---------------------------------------------------------------------- #
# 语义绑定正确性 (本项目核心创新点的直接度量)
# ---------------------------------------------------------------------- #
def _value_bound(view: _GraphView, obj: str, val: str, shared_attr: str) -> bool:
    """value 是否真属于 object 在 shared_attr 指标上的取值 (两条物理路径任一成立)。

      通用链路：val ∈ object --has_attribute--> 指标 --has_value--> value
      表格路径：object 为表行时，shared_attr 列上该行的单元格恰为 val
    """
    if not (obj and val):
        return False
    if val in set(ec.values_via_attribute(view, obj)):
        return True
    cols = ec.table_row_columns(view, obj)
    if shared_attr and cols.get(shared_attr) == val:
        return True
    return val in set(cols.values())


def binding_correctness(plans: List[Dict[str, Any]], view: _GraphView) -> Dict[str, Any]:
    # comparative: value_X 是否真经 object_X 的指标链/表行列结构可达
    comp = [p for p in plans if p.get("question_type") == "comparative"]
    comp_ok = 0
    comp_den = 0
    for p in comp:
        ra = p.get("role_assignment", {}) or {}
        sa = ra.get("shared_attribute", "")
        for obj_role, val_role in (("object_A", "value_A"), ("object_B", "value_B")):
            obj, val = ra.get(obj_role), ra.get(val_role)
            if not (obj and val):
                continue
            comp_den += 1
            if _value_bound(view, obj, val, sa):
                comp_ok += 1

    # cross_paper: result_A / result_B 是否落在不同论文；paper_B_instance 是否真实例
    cp = [p for p in plans if p.get("question_type") == "cross_paper"]
    cp_result_cross = 0
    cp_result_den = 0
    cp_real_inst = 0
    cp_inst_den = 0
    for p in cp:
        ra = p.get("role_assignment", {}) or {}
        ra_, rb_ = ra.get("result_A"), ra.get("result_B")
        if ra_ and rb_:
            cp_result_den += 1
            if _paper_id(view, ra_) and _paper_id(view, ra_) != _paper_id(view, rb_):
                cp_result_cross += 1
        bi = ra.get("paper_B_instance")
        if bi:
            cp_inst_den += 1
            nt = view.nodes.get(bi, {}).get("node_type", "")
            from autoqag.schema import NodeType

            real = nt in (NodeType.CONCEPT.value, NodeType.METHOD.value, NodeType.VALUE.value) \
                and _node_ok(view, bi)
            if real:
                cp_real_inst += 1

    # 单位接地：value 与 unit 是否经 has_unit (数值自身或表格列头) 相连
    unit_ok = 0
    unit_den = 0
    for p in plans:
        if p.get("question_type") not in ("numerical", "formula", "comparative"):
            continue
        ra = p.get("role_assignment", {}) or {}
        col_unit = ec.column_unit(view, ra.get("shared_attribute", "")) if ra.get("shared_attribute") else ""
        for val_role in ("value", "value_A", "value_B"):
            val = ra.get(val_role)
            unit = ra.get("unit")
            if not (val and unit):
                continue
            unit_den += 1
            if ec.unit_of_value(view, val) == unit or (col_unit and col_unit == unit):
                unit_ok += 1

    def _rate(a: int, b: int) -> Optional[Dict[str, Any]]:
        if b == 0:
            return {"rate": None, "n": 0}
        return {"rate": round(a / b, 3), "ok": a, "n": b}

    return {
        "comparative_value_object_bind": _rate(comp_ok, comp_den),
        "cross_paper_result_cross": _rate(cp_result_cross, cp_result_den),
        "cross_paper_instance_real": _rate(cp_real_inst, cp_inst_den),
        "unit_grounded": _rate(unit_ok, unit_den),
    }


# ---------------------------------------------------------------------- #
def compute_internal(plans: List[Dict[str, Any]], view: _GraphView) -> Dict[str, Any]:
    return {
        "coverage_structure": coverage_structure(plans, view),
        "multihop": multihop_authenticity(plans, view),
        "logical": logical_completeness(plans, view),
        "binding": binding_correctness(plans, view),
    }
