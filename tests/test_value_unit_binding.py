"""回归测试：value↔unit 绑定 (修复约束构造把游离单位张冠李戴到数值的 bug)。

bug 背景：planner._finalize 旧实现把子图内所有 VALUE 塞进 number、所有 UNIT 塞进
unit，两个独立扁平 list，零绑定。于是同表/同段里别的值的单位 (如表头的 GHz) 会被
错标到一个本是百分比的数值上 (实测样本 sp000013: 2.2% 被标 unit=GHz)。

修复：evidence_chain.resolve_value_unit 只沿该数值自己的 has_unit (或其所属表列头的
has_unit) 取单位，游离单位节点不会被绑定。
"""

from types import SimpleNamespace

from autoqag.ops.m5_sample.semantic.evidence_chain import resolve_value_unit
from autoqag.schema import NodeType


def _view(nodes, edges):
    """构造满足 evidence_chain duck-typing 的最小 view (.nodes/.out/.inc)。
    edges: List[(src, dst, edge_type)]。"""
    out, inc = {}, {}
    for s, t, et in edges:
        out.setdefault(s, []).append((t, {"edge_type": et, "confidence": 1.0, "polarity": "positive"}))
        inc.setdefault(t, []).append((s, {"edge_type": et, "confidence": 1.0, "polarity": "positive"}))
    return SimpleNamespace(nodes=nodes, out=out, inc=inc)


def test_direct_has_unit_binds_correct_unit():
    nodes = {
        "v_pct": {"node_type": NodeType.VALUE.value, "content": "2.2"},
        "u_pct": {"node_type": NodeType.UNIT.value, "content": "%"},
        "u_ghz": {"node_type": NodeType.UNIT.value, "content": "GHz"},  # 游离单位
    }
    view = _view(nodes, [("v_pct", "u_pct", "has_unit")])
    assert view.nodes[resolve_value_unit(view, "v_pct")]["content"] == "%"


def test_floating_unit_not_misbound():
    """子图里存在 GHz 单位节点，但它不与该数值有 has_unit 边 → 不得被绑定 (核心 bug 场景)。"""
    nodes = {
        "v_pct": {"node_type": NodeType.VALUE.value, "content": "2.2"},
        "u_ghz": {"node_type": NodeType.UNIT.value, "content": "GHz"},
    }
    view = _view(nodes, [])  # 无 has_unit 边
    assert resolve_value_unit(view, "v_pct") == ""  # 而非错绑到 GHz


def test_column_header_unit_fallback():
    """表格数值的单位挂在列头：value <-has_value- column -has_unit-> unit。"""
    nodes = {
        "cell": {"node_type": NodeType.VALUE.value, "content": "3.4"},
        "col": {"node_type": NodeType.ATTRIBUTE.value, "content": "RBWT"},
        "u_pct": {"node_type": NodeType.UNIT.value, "content": "%"},
    }
    view = _view(nodes, [("col", "cell", "has_value"), ("col", "u_pct", "has_unit")])
    assert view.nodes[resolve_value_unit(view, "cell")]["content"] == "%"


def test_direct_unit_takes_precedence_over_column():
    nodes = {
        "cell": {"node_type": NodeType.VALUE.value, "content": "3.4"},
        "col": {"node_type": NodeType.ATTRIBUTE.value, "content": "RBWT"},
        "u_db": {"node_type": NodeType.UNIT.value, "content": "dB"},
        "u_pct": {"node_type": NodeType.UNIT.value, "content": "%"},
    }
    view = _view(nodes, [
        ("cell", "u_db", "has_unit"),     # 数值自身单位
        ("col", "cell", "has_value"),
        ("col", "u_pct", "has_unit"),     # 列头单位 (应让位于数值自身单位)
    ])
    assert view.nodes[resolve_value_unit(view, "cell")]["content"] == "dB"


def test_no_unit_returns_empty():
    nodes = {"v": {"node_type": NodeType.VALUE.value, "content": "42"}}
    view = _view(nodes, [])
    assert resolve_value_unit(view, "v") == ""
