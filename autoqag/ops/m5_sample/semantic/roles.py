"""题型逻辑角色 schema (语义层子图构建.pdf §3.1.1 / §3.4.1；语义规划层方法论.pdf §3.1)。

每个题型定义一组逻辑角色 (required_reasoning_roles)，并区分:
- ROLE_SCHEMAS         : 完整逻辑角色 (规划目标)
- MIN_ROLE_SETS        : 最小角色集合 (角色完整性检查的硬门槛 §3.4.1)
- ROLE_NODE_TYPES      : 每个逻辑角色可由哪些 NodeType 承担 (从 G0 找候选节点用)
- SEED_STRATEGY        : 不同题型的初始化种子策略 (§3.1.4)
"""

from __future__ import annotations

from typing import Dict, List

from autoqag.schema import InferenceOp, NodeType, QuestionType

# ---------------------------------------------------------------------------
# 完整逻辑角色 (语义层子图构建.pdf §3.1.1)
# ---------------------------------------------------------------------------
ROLE_SCHEMAS: Dict[str, List[str]] = {
    QuestionType.COMPARATIVE.value: [
        "object_A", "object_B", "shared_attribute", "value_A", "value_B",
        "unit", "aligned_condition", "evidence_A", "evidence_B", "comparison_criterion",
    ],
    "mechanism": [
        "method_or_intervention", "intermediate_mechanism", "target_attribute",
        "observed_result", "supporting_evidence", "condition", "conclusion",
    ],
    QuestionType.CONDITION.value: [
        "claim", "condition_boundary", "attribute_or_result",
        "evidence_span", "invalid_generalization_risk",
    ],
    "cross_paper": [
        "canonical_concept", "paper_A_instance", "paper_B_instance",
        "aligned_attribute", "condition_A", "condition_B",
        "result_A", "result_B", "evidence_A", "evidence_B", "cross_paper_relation",
    ],
    QuestionType.NUMERICAL.value: [
        "concept", "attribute", "value", "unit", "condition", "evidence",
    ],
    QuestionType.FORMULA.value: [
        "equation", "attribute_or_result", "value", "condition", "evidence",
    ],
    QuestionType.TABLE.value: [
        "text_claim", "figure_or_table", "caption", "value", "evidence",
    ],
    QuestionType.MULTI_HOP.value: [
        "concept", "attribute", "intermediate_node", "evidence", "conclusion",
    ],
    QuestionType.SUMMARY.value: ["section", "claim", "concept", "evidence"],
    QuestionType.ATOMIC.value: ["concept", "evidence"],
}

# ---------------------------------------------------------------------------
# 最小角色集合 —— 角色完整性硬门槛 (语义层子图构建.pdf §3.4.1)
# 缺少其中任一角色，问题子图不允许进入生成阶段。
# ---------------------------------------------------------------------------
MIN_ROLE_SETS: Dict[str, List[str]] = {
    QuestionType.COMPARATIVE.value: [
        "object_A", "object_B", "shared_attribute", "value_A", "value_B",
        "evidence_A", "evidence_B",
    ],
    "mechanism": [
        "method_or_intervention", "intermediate_mechanism",
        "target_attribute", "observed_result", "supporting_evidence",
    ],
    QuestionType.CONDITION.value: [
        "claim", "condition_boundary", "attribute_or_result", "evidence_span",
    ],
    "cross_paper": [
        "canonical_concept", "paper_A_instance", "paper_B_instance",
        "aligned_attribute", "evidence_A", "evidence_B",
    ],
    QuestionType.NUMERICAL.value: ["attribute", "value", "evidence"],
    QuestionType.FORMULA.value: ["equation", "evidence"],
    QuestionType.TABLE.value: ["figure_or_table", "evidence"],
    QuestionType.MULTI_HOP.value: ["concept", "intermediate_node", "evidence", "conclusion"],
    QuestionType.SUMMARY.value: ["section", "claim"],
    QuestionType.ATOMIC.value: ["concept"],
}

# ---------------------------------------------------------------------------
# 逻辑角色 → 可承担的 NodeType (从 G0 为角色找候选节点；语义层子图构建.pdf §3.2.3)
# ---------------------------------------------------------------------------
ROLE_NODE_TYPES: Dict[str, List[str]] = {
    "object_A": [NodeType.CONCEPT.value, NodeType.METHOD.value],
    "object_B": [NodeType.CONCEPT.value, NodeType.METHOD.value],
    "shared_attribute": [NodeType.ATTRIBUTE.value],
    "aligned_attribute": [NodeType.ATTRIBUTE.value],
    "target_attribute": [NodeType.ATTRIBUTE.value],
    "attribute": [NodeType.ATTRIBUTE.value],
    "attribute_or_result": [NodeType.ATTRIBUTE.value, NodeType.VALUE.value, NodeType.CLAIM.value],
    "value": [NodeType.VALUE.value],
    "value_A": [NodeType.VALUE.value],
    "value_B": [NodeType.VALUE.value],
    "result_A": [NodeType.VALUE.value, NodeType.CLAIM.value],
    "result_B": [NodeType.VALUE.value, NodeType.CLAIM.value],
    "observed_result": [NodeType.CLAIM.value, NodeType.VALUE.value, NodeType.ATTRIBUTE.value],
    "unit": [NodeType.UNIT.value],
    "condition": [NodeType.CONDITION.value],
    "condition_A": [NodeType.CONDITION.value],
    "condition_B": [NodeType.CONDITION.value],
    "condition_boundary": [NodeType.CONDITION.value],
    "aligned_condition": [NodeType.CONDITION.value],
    "claim": [NodeType.CLAIM.value],
    "conclusion": [NodeType.CLAIM.value],
    "text_claim": [NodeType.CLAIM.value],
    "comparison_criterion": [NodeType.ATTRIBUTE.value, NodeType.CLAIM.value],
    "method_or_intervention": [NodeType.METHOD.value],
    "intermediate_mechanism": [NodeType.CLAIM.value, NodeType.METHOD.value, NodeType.CONCEPT.value],
    "intermediate_node": [
        NodeType.CONCEPT.value, NodeType.ATTRIBUTE.value,
        NodeType.METHOD.value, NodeType.CLAIM.value,
    ],
    "canonical_concept": [NodeType.CONCEPT.value, NodeType.ATTRIBUTE.value, NodeType.CLAIM.value],
    "concept": [NodeType.CONCEPT.value, NodeType.METHOD.value, NodeType.CLAIM.value],
    "paper_A_instance": [NodeType.CONCEPT.value, NodeType.ATTRIBUTE.value, NodeType.VALUE.value],
    "paper_B_instance": [NodeType.CONCEPT.value, NodeType.ATTRIBUTE.value, NodeType.VALUE.value],
    "equation": [NodeType.EQUATION.value],
    "figure_or_table": [NodeType.FIGURE.value, NodeType.TABLE.value],
    "caption": [NodeType.CAPTION.value],
    "section": [NodeType.SECTION.value],
    "evidence": [
        NodeType.EVIDENCE.value, NodeType.CHUNK.value, NodeType.FIGURE.value,
        NodeType.TABLE.value, NodeType.CLAIM.value, NodeType.VALUE.value,
    ],
    "evidence_A": [
        NodeType.EVIDENCE.value, NodeType.CHUNK.value, NodeType.FIGURE.value,
        NodeType.TABLE.value, NodeType.VALUE.value,
    ],
    "evidence_B": [
        NodeType.EVIDENCE.value, NodeType.CHUNK.value, NodeType.FIGURE.value,
        NodeType.TABLE.value, NodeType.VALUE.value,
    ],
    "evidence_span": [NodeType.EVIDENCE.value, NodeType.CHUNK.value, NodeType.CLAIM.value],
    "supporting_evidence": [
        NodeType.EVIDENCE.value, NodeType.FIGURE.value, NodeType.TABLE.value,
        NodeType.CHUNK.value, NodeType.CLAIM.value,
    ],
}

# 抽象/规划性角色：无独立证据节点，由约束或规划标记体现 (不参与证据可回落硬检查)。
# aligned_condition: 比较题"条件须对齐"是规划约束，由 generation_instruction /
# forbidden_generalization 表达；物理图常无可直接挂载的 ConditionNode，故不作硬门槛。
ABSTRACT_ROLES = {
    "invalid_generalization_risk",
    "risk_of_overgeneralization",
    "comparison_criterion",
    "cross_paper_relation",
    "aligned_condition",
}

# ---------------------------------------------------------------------------
# 不同题型的初始化种子策略 (语义层子图构建.pdf §3.1.4 + 语义规划层方法论.pdf §3.3)
# 每项: (优先种子 NodeType 列表, 初始化逻辑说明)
# ---------------------------------------------------------------------------
SEED_STRATEGY: Dict[str, Dict[str, object]] = {
    QuestionType.NUMERICAL.value: {
        "seed_types": [NodeType.ATTRIBUTE.value, NodeType.VALUE.value],
        "logic": "先确定指标/数值，向上找 Concept，向下找 Unit 和 Condition。",
    },
    QuestionType.CONDITION.value: {
        "seed_types": [NodeType.CLAIM.value, NodeType.CONDITION.value, NodeType.ATTRIBUTE.value],
        "logic": "先确定结论或条件，再寻找适用范围、属性结果和证据 span。",
    },
    QuestionType.COMPARATIVE.value: {
        "seed_types": [NodeType.ATTRIBUTE.value],
        "logic": "先确定共享指标，再寻找多个可比对象、数值、单位和对齐条件。",
    },
    "mechanism": {
        "seed_types": [NodeType.METHOD.value, NodeType.CLAIM.value],
        "logic": "先确定方法或结论，再向中间机制、结果指标和图表证据扩展。",
    },
    "cross_paper": {
        "seed_types": [NodeType.CONCEPT.value, NodeType.ATTRIBUTE.value],
        "logic": "先确定跨文献对齐对象 (canonical)，再寻找不同 paper_id 下的实例和证据。",
    },
    QuestionType.TABLE.value: {
        "seed_types": [NodeType.FIGURE.value, NodeType.TABLE.value, NodeType.CLAIM.value],
        "logic": "先确定图表或正文结论，再通过图注和引用关系扩展证据。",
    },
    QuestionType.FORMULA.value: {
        "seed_types": [NodeType.EQUATION.value],
        "logic": "先确定公式，再寻找其推导的属性/数值与适用条件。",
    },
    QuestionType.MULTI_HOP.value: {
        "seed_types": [NodeType.CONCEPT.value, NodeType.CLAIM.value, NodeType.METHOD.value],
        "logic": "从高层节点出发，跨 chunk/section 扩展多步推理链。",
    },
    QuestionType.SUMMARY.value: {
        "seed_types": [NodeType.SECTION.value],
        "logic": "以章节为种子，聚合其下的 Claim/Concept。",
    },
    QuestionType.ATOMIC.value: {
        "seed_types": [NodeType.CONCEPT.value, NodeType.CLAIM.value, NodeType.METHOD.value],
        "logic": "单一事实点。",
    },
}


def role_schema(qtype: str) -> List[str]:
    return ROLE_SCHEMAS.get(qtype, ROLE_SCHEMAS[QuestionType.ATOMIC.value])


def min_roles(qtype: str) -> List[str]:
    return MIN_ROLE_SETS.get(qtype, MIN_ROLE_SETS[QuestionType.ATOMIC.value])


def node_types_for_role(role: str) -> List[str]:
    return ROLE_NODE_TYPES.get(role, [])


def seed_types(qtype: str) -> List[str]:
    return list(SEED_STRATEGY.get(qtype, SEED_STRATEGY[QuestionType.ATOMIC.value])["seed_types"])


def is_evidence_role(role: str) -> bool:
    return "evidence" in role or role in ("supporting_evidence", "evidence_span")


# ---------------------------------------------------------------------------
# 题型核心推断算子 + 核心虚拟边模板 (operational_flow.md §3.6.0)
# 每项: (核心 InferenceOp, [(source_role, target_role, VirtualEdgeType, question_role), ...])
# 核心边在角色绑定完成后 always 铺设 (即使最小角色集已被物理扩展填满)，
# 把虚拟边从"事后补缺的补丁"提升为"每道题的推理主干"。
# 单跳事实题 (numerical/formula/atomic) 不强制核心边，仅缺角色时补全。
# ---------------------------------------------------------------------------
from autoqag.schema import VirtualEdgeType as _VET  # noqa: E402

CORE_INFERENCE: Dict[str, Dict[str, object]] = {
    QuestionType.COMPARATIVE.value: {
        "op": InferenceOp.COMPARISON.value,
        "edges": [("object_A", "object_B", _VET.COMPARABLE.value, "comparison_criterion")],
    },
    QuestionType.CONDITION.value: {
        "op": InferenceOp.CONDITION_BIND.value,
        "edges": [("claim", "condition_boundary", _VET.LIMITED_BY.value, "condition_boundary")],
    },
    "mechanism": {
        "op": InferenceOp.CAUSAL_CHAIN.value,
        "edges": [
            ("method_or_intervention", "intermediate_mechanism", _VET.SEEK_MECHANISM.value, "intermediate_mechanism"),
            ("intermediate_mechanism", "observed_result", _VET.METHOD_EFFECT.value, "observed_result"),
        ],
    },
    "cross_paper": {
        "op": InferenceOp.CROSS_PAPER_ALIGN.value,
        "edges": [("paper_A_instance", "paper_B_instance", _VET.CROSS_PAPER_ALIGN.value, "cross_paper_relation")],
    },
    QuestionType.TABLE.value: {
        "op": InferenceOp.VISUAL_GROUNDING.value,
        "edges": [("text_claim", "figure_or_table", _VET.NEED_VISUAL_EVIDENCE.value, "figure_or_table")],
    },
    QuestionType.SUMMARY.value: {
        "op": InferenceOp.AGGREGATION.value,
        "edges": [],  # 聚合无单一二元核心边；算子下限仍生效
    },
}


def core_inference(qtype: str) -> Dict[str, object]:
    """返回题型的核心推断算子与核心虚拟边模板；无核心边的题型返回空模板。"""
    return CORE_INFERENCE.get(qtype, {"op": "", "edges": []})


def core_inference_op(qtype: str) -> str:
    return str(core_inference(qtype).get("op", "") or "")
