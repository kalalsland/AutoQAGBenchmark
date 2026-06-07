"""边语义规则 (论文 §5.3 + 图谱构建.pdf §三)。

核心思想：**点标签决定边的语义类型，物理共现/文章结构决定边是否建立**。
本文件只负责"语义类型 = f(端点标签)"这一半；"是否建立"由 graph.py 依据共现/结构判定。

边的方向规范化：_PAIR_RULES 的 key 顺序即规范方向 (如 Attribute→Value 为 has_value)。
resolve_edge 对反向输入返回 flip=True，graph.py 据此交换 source/target 落库，
保证下游 (m5 采样器按 out/inc 方向遍历) 不漏边。
"""

from __future__ import annotations

from typing import Optional, Tuple

from autoqag.schema import NodeType

# (源类型, 目标类型) → 边语义类型 (论文 §5.3 边语义解释表)。key 顺序 = 规范方向。
_PAIR_RULES = {
    (NodeType.CONCEPT.value, NodeType.ATTRIBUTE.value): "has_attribute",
    (NodeType.ATTRIBUTE.value, NodeType.VALUE.value): "has_value",
    (NodeType.VALUE.value, NodeType.UNIT.value): "has_unit",
    (NodeType.CONDITION.value, NodeType.VALUE.value): "under_condition",
    (NodeType.CONDITION.value, NodeType.ATTRIBUTE.value): "under_condition",
    (NodeType.TABLE.value, NodeType.CLAIM.value): "supports",
    (NodeType.FIGURE.value, NodeType.CLAIM.value): "supports",
    (NodeType.EQUATION.value, NodeType.ATTRIBUTE.value): "derived_from",
    (NodeType.EQUATION.value, NodeType.VALUE.value): "derived_from",
    (NodeType.FIGURE.value, NodeType.CONCEPT.value): "describes",
    (NodeType.TABLE.value, NodeType.CONCEPT.value): "describes",
    (NodeType.CONCEPT.value, NodeType.CONCEPT.value): "compares",
}


def resolve_edge(src_type: str, tgt_type: str) -> Optional[Tuple[str, bool]]:
    """按端点标签返回 (边语义类型, 是否需要交换方向)。

    - 正向命中：(sem, False)
    - 反向命中：(sem, True) —— 调用方应交换 source/target 后落库
    - 未命中：None (回退到通用 co_occurs_with)
    """
    if (src_type, tgt_type) in _PAIR_RULES:
        return _PAIR_RULES[(src_type, tgt_type)], False
    if (tgt_type, src_type) in _PAIR_RULES:
        return _PAIR_RULES[(tgt_type, src_type)], True
    return None


def edge_semantic_type(src_type: str, tgt_type: str) -> Optional[str]:
    """仅返回语义类型 (不关心方向) 的便捷封装。"""
    r = resolve_edge(src_type, tgt_type)
    return r[0] if r else None


# LLM 抽取阶段允许的 point_type 标签 → 规范 NodeType
LABEL_TO_NODE_TYPE = {
    "concept": NodeType.CONCEPT.value,
    "attribute": NodeType.ATTRIBUTE.value,
    "value": NodeType.VALUE.value,
    "unit": NodeType.UNIT.value,
    "condition": NodeType.CONDITION.value,
    "method": NodeType.METHOD.value,
    "claim": NodeType.CLAIM.value,
    "equation": NodeType.EQUATION.value,
    "figure": NodeType.FIGURE.value,
    "table": NodeType.TABLE.value,
    "caption": NodeType.CAPTION.value,
}
