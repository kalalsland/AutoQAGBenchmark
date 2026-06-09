"""刀1 防护测试：错误共现边 (否定/对比/假设) 的检测、降权与抑制。

覆盖三层：
1) 纯函数层 —— detect_polarity / edge_confidence / is_evidence_eligible
2) 建图层 (m4) —— _add_cooccur_edges：规则补全的负向边被抑制不建；
   LLM 断言的负向边保留但标 polarity + 降权。
3) 证据消费层 (m5) —— values_via_attribute / attributes_of 跳过不可作证据的边。

运行： (在 AutoQAGBenchmark/ 下) python -m pytest tests/test_edge_polarity.py -v
"""

from __future__ import annotations

import pytest

from autoqag.common.graph_store import GraphStore
from autoqag.ops.m4_graph.graph import GraphStage
from autoqag.ops.m4_graph.quality import (
    EVIDENCE_CONFIDENCE_THRESHOLD,
    detect_polarity,
    edge_confidence,
    is_evidence_eligible,
)
from autoqag.ops.m5_sample.semantic.evidence_chain import (
    attributes_of,
    values_via_attribute,
)
from autoqag.schema import Address, EvidenceBlock, Modality, NodeType, PointNode


# --------------------------------------------------------------------------
# 1) detect_polarity —— 真实科研句的错误样例
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, a, b, expected",
    [
        # 正向：原文确实断言该数值
        ("The PCE of device A reaches 23.5% under 85C aging.", "PCE", "23.5", "positive"),
        # 否定：does NOT reach —— 经典刀1 反例
        ("Unlike device A, device B does not reach a PCE of 30 percent.", "PCE", "30", "negative"),
        ("Strategy B fails to improve the retention rate of 90 percent.", "retention rate", "90", "negative"),
        ("The coating shows no improvement in stability of 95 percent.", "stability", "95", "negative"),
        # 对比/让步：Unlike A, ... —— 关系被对比反转
        ("Unlike strategy A, strategy B improves PCE markedly.", "strategy B", "PCE", "contrastive"),
        ("Method X is fast, whereas method Y reaches 12 ms latency.", "method Y", "12", "contrastive"),
        # 假设/情态：非实测值
        ("If the temperature increased, the PCE would rise to 25 percent.", "temperature", "25", "hypothetical"),
        ("The model could potentially achieve 40 percent efficiency.", "model", "40", "hypothetical"),
        # 中文否定 / 对比 / 假设
        ("与策略A不同，策略B并不能提升PCE的稳定性。", "策略B", "PCE", "negative"),
        ("尽管成本上升，方法Y达到了12的延迟。", "方法Y", "12", "contrastive"),
        ("假设温度升高，PCE可能会上升到25。", "温度", "25", "hypothetical"),
    ],
)
def test_detect_polarity(text, a, b, expected):
    assert detect_polarity(text, a, b) == expected


def test_detect_polarity_window_localizes():
    """否定出现在两提及之间才算数；远处的无关否定不应误伤正向断言。"""
    # 'not' 在句首与目标对无关从句里，PCE→23.5 之间无否定线索
    text = "This is not the main result. Here the PCE of device A reaches 23.5 percent."
    assert detect_polarity(text, "PCE", "23.5") == "positive"


# --------------------------------------------------------------------------
# 2) edge_confidence / is_evidence_eligible
# --------------------------------------------------------------------------
def test_confidence_by_scope_and_polarity():
    # 同句正向 = 满分，可作证据
    assert edge_confidence("same_sentence", "positive") == 1.0
    assert is_evidence_eligible(1.0, "positive")
    # 同段正向可作证据
    assert is_evidence_eligible(edge_confidence("same_paragraph", "positive"), "positive")
    # 同块 all-pairs 规则补全 (LLM 未断言) → 低于阈值，仅召回不作证据
    c = edge_confidence("same_chunk", "positive", rule_completion=True)
    assert c < EVIDENCE_CONFIDENCE_THRESHOLD
    assert not is_evidence_eligible(c, "positive")


def test_negated_edge_never_eligible():
    for scope in ("same_sentence", "same_paragraph", "same_chunk"):
        for pol in ("negative", "contrastive", "hypothetical"):
            c = edge_confidence(scope, pol)
            assert not is_evidence_eligible(c, pol), (scope, pol, c)


# --------------------------------------------------------------------------
# 3) 建图层：_add_cooccur_edges 的抑制与标记
# --------------------------------------------------------------------------
def _mk_block(content: str) -> EvidenceBlock:
    return EvidenceBlock(
        block_id="blk1",
        modality=Modality.TEXT.value,
        content=content,
        address=Address(paper_id="paperX", section_path="Results", chunk_id="paperX_c1"),
        figure_refs=[],
        table_refs=[],
        equation_refs=[],
        extra={},
    )


def _seed_points(store: GraphStore, specs):
    """specs: [(node_id, name, NodeType.value, content)]；建点并返回 name_to_id。"""
    name_to_id = {}
    for nid, name, ntype, content in specs:
        store.upsert_node(
            PointNode(
                node_id=nid,
                node_type=ntype,
                content=content,
                normalized_content=name,
                address=Address(paper_id="paperX", section_path="Results", chunk_id="paperX_c1"),
            )
        )
        name_to_id[name.lower()] = (nid, ntype)
    return name_to_id


def _cooccur_edge_types(store: GraphStore):
    return {
        (d["source"], d["target"], d["edge_type"]): d
        for _s, _t, d in store.all_edges()
        if d.get("build_reason") == "physical_cooccurrence"
    }


def test_rule_completion_negated_edge_suppressed():
    """LLM 未断言 + 原文否定 → 纯共现边直接不建 (核心避免行为)。"""
    stage = GraphStage()
    store = GraphStore()
    text = "Unlike device A, device B does not reach a PCE of 30 percent."
    block = _mk_block(text)
    specs = [
        ("n_concept", "device B", NodeType.CONCEPT.value, "device B"),
        ("n_attr", "PCE", NodeType.ATTRIBUTE.value, "PCE"),
        ("n_val", "30", NodeType.VALUE.value, "30 percent"),
    ]
    name_to_id = _seed_points(store, specs)
    points = [{"name": n, "type": t, "content": c} for _i, n, t, c in specs]
    # relations 为空 → 全部走规则补全
    stage._add_cooccur_edges(store, block, points, [], name_to_id, [text])

    edges = _cooccur_edge_types(store)
    # PCE--has_value-->30 与 device B--has_attribute-->PCE 都应被抑制
    assert not any(et == "has_value" for (_s, _t, et) in edges), edges
    assert not any(et == "has_attribute" for (_s, _t, et) in edges), edges


def test_rule_completion_positive_edge_built():
    """同样结构但原文为正向断言 → 边正常建立，polarity=positive。"""
    stage = GraphStage()
    store = GraphStore()
    text = "Here device B reaches a PCE of 30 percent under standard test."
    block = _mk_block(text)
    specs = [
        ("n_concept", "device B", NodeType.CONCEPT.value, "device B"),
        ("n_attr", "PCE", NodeType.ATTRIBUTE.value, "PCE"),
        ("n_val", "30", NodeType.VALUE.value, "30 percent"),
    ]
    name_to_id = _seed_points(store, specs)
    points = [{"name": n, "type": t, "content": c} for _i, n, t, c in specs]
    stage._add_cooccur_edges(store, block, points, [], name_to_id, [text])

    edges = _cooccur_edge_types(store)
    hv = edges.get(("n_attr", "n_val", "has_value"))
    assert hv is not None, edges
    assert hv["polarity"] == "positive"


def test_llm_asserted_negated_edge_kept_but_downweighted():
    """LLM 显式断言了关系 → 边保留 (不丢信号)，但标 negative + 降权，不进证据层。"""
    stage = GraphStage()
    store = GraphStore()
    text = "Unlike device A, device B does not reach a PCE of 30 percent."
    block = _mk_block(text)
    specs = [
        ("n_attr", "PCE", NodeType.ATTRIBUTE.value, "PCE"),
        ("n_val", "30", NodeType.VALUE.value, "30 percent"),
    ]
    name_to_id = _seed_points(store, specs)
    points = [{"name": n, "type": t, "content": c} for _i, n, t, c in specs]
    relations = [{"source": "PCE", "target": "30", "relation": "does_not_reach"}]
    stage._add_cooccur_edges(store, block, points, relations, name_to_id, [text])

    edges = _cooccur_edge_types(store)
    hv = edges.get(("n_attr", "n_val", "has_value"))
    assert hv is not None, "LLM 断言的边应保留"
    assert hv["polarity"] == "negative"
    assert hv["confidence"] < EVIDENCE_CONFIDENCE_THRESHOLD
    assert not is_evidence_eligible(hv["confidence"], hv["polarity"])


# --------------------------------------------------------------------------
# 4) 证据消费层 (m5)：负向/低置信边不被取作正向事实
# --------------------------------------------------------------------------
class _FakeView:
    def __init__(self, nodes, out, inc=None):
        self.nodes = nodes
        self.out = out
        self.inc = inc or {}


def _value_view(has_value_polarity: str, has_value_conf: float):
    nodes = {
        "obj": {"node_type": NodeType.CONCEPT.value, "content": "device B"},
        "attr": {"node_type": NodeType.ATTRIBUTE.value, "content": "PCE"},
        "val": {"node_type": NodeType.VALUE.value, "content": "30"},
    }
    out = {
        "obj": [("attr", {"edge_type": "has_attribute", "polarity": "positive", "confidence": 1.0})],
        "attr": [("val", {"edge_type": "has_value", "polarity": has_value_polarity, "confidence": has_value_conf})],
    }
    return _FakeView(nodes, out)


def test_m5_skips_negated_value_edge():
    """has_value 边被否定 → values_via_attribute 不返回该数值。"""
    view = _value_view("negative", 0.25)
    assert values_via_attribute(view, "obj") == []


def test_m5_keeps_positive_value_edge():
    view = _value_view("positive", 1.0)
    assert values_via_attribute(view, "obj") == ["val"]


def test_m5_skips_low_confidence_rule_completion_edge():
    """正向但低置信 (规则补全 0.4 < 阈值) → 不作证据。"""
    view = _value_view("positive", 0.4)
    assert values_via_attribute(view, "obj") == []


def test_m5_backward_compatible_without_fields():
    """旧图边无 polarity/confidence 字段 → 默认 positive/1.0，照常可用。"""
    nodes = {
        "obj": {"node_type": NodeType.CONCEPT.value},
        "attr": {"node_type": NodeType.ATTRIBUTE.value},
        "val": {"node_type": NodeType.VALUE.value},
    }
    out = {
        "obj": [("attr", {"edge_type": "has_attribute"})],
        "attr": [("val", {"edge_type": "has_value"})],
    }
    view = _FakeView(nodes, out)
    assert values_via_attribute(view, "obj") == ["val"]


def test_m5_attributes_of_skips_negated():
    nodes = {"obj": {"node_type": NodeType.CONCEPT.value}, "attr": {"node_type": NodeType.ATTRIBUTE.value}}
    out = {"obj": [("attr", {"edge_type": "has_attribute", "polarity": "negative", "confidence": 0.25})]}
    view = _FakeView(nodes, out)
    assert attributes_of(view, "obj") == []
