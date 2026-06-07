"""类型特定约束验证器 (论文 §5.7，MVP 四类：数值/单位/条件/证据)。

每个 checker 接收 QAItem + 证据文本，返回 Violation 列表 (论文创新五格式)。
checker 与 stage 解耦，便于单测与按题型组合 (论文 §5.7)。
"""

from __future__ import annotations

import re
from typing import List

from autoqag.ops.m3_normalize.units import extract_value_units, units_compatible
from autoqag.schema import QAItem, Violation, VerifyLayer

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _evidence_text(qa: QAItem) -> str:
    return " ".join(e.get("content", "") for e in qa.evidence_spans)


def _nums_close(a: str, b: str, rel_tol: float = 0.02) -> bool:
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a == b
    if y == 0:
        return abs(x) < 1e-9
    return abs(x - y) / abs(y) <= rel_tol


def check_numeric(qa: QAItem, evidence: str) -> List[Violation]:
    """第一层：答案中的数值必须能在证据中找到 (允许 2% 容差)。"""
    out: List[Violation] = []
    ev_nums = _NUM_RE.findall(evidence)
    for num in _NUM_RE.findall(qa.answer):
        if not any(_nums_close(num, e) for e in ev_nums):
            out.append(
                Violation(
                    qid=qa.qid,
                    layer=VerifyLayer.CONSTRAINT.value,
                    field="number",
                    expected=f"a value present in evidence ({ev_nums[:5]})",
                    actual=num,
                    severity="critical",
                    repair_hint="把数值替换为证据中实际出现的值",
                )
            )
    return out


def check_unit(qa: QAItem, evidence: str) -> List[Violation]:
    """第一层：答案单位需与证据单位同量纲 (可换算)。"""
    out: List[Violation] = []
    ans_units = {u for _, u in extract_value_units(qa.answer) if u}
    ev_units = {u for _, u in extract_value_units(evidence) if u}
    if not ans_units or not ev_units:
        return out
    for au in ans_units:
        if not any(units_compatible(au, eu) for eu in ev_units):
            out.append(
                Violation(
                    qid=qa.qid,
                    layer=VerifyLayer.CONSTRAINT.value,
                    field="unit",
                    expected=f"unit compatible with evidence ({sorted(ev_units)})",
                    actual=au,
                    severity="major",
                    repair_hint="修正单位或进行单位换算",
                )
            )
    return out


def check_condition(qa: QAItem, evidence: str) -> List[Violation]:
    """第一层：条件型答案必须显式保留限定条件 (不得泛化)。"""
    out: List[Violation] = []
    conds = qa.constraints.get("condition", []) or []
    # 过度泛化措辞检测
    generalizers = ["all cases", "all conditions", "regardless", "any material", "所有情况", "所有条件"]
    if any(g in qa.answer.lower() for g in generalizers) and conds:
        out.append(
            Violation(
                qid=qa.qid,
                layer=VerifyLayer.CONSTRAINT.value,
                field="condition",
                expected=f"keep condition: {conds}",
                actual="over-generalized answer",
                severity="major",
                repair_hint="补充限定条件，避免泛化",
            )
        )
    for cond in conds:
        # 条件的关键 token 是否在答案中体现
        key = re.sub(r"[^0-9a-zA-Z一-鿿]+", " ", str(cond)).split()
        if key and not any(k in qa.answer for k in key if len(k) > 1):
            out.append(
                Violation(
                    qid=qa.qid,
                    layer=VerifyLayer.CONSTRAINT.value,
                    field="condition",
                    expected=str(cond),
                    actual="condition missing in answer",
                    severity="major",
                    repair_hint="在答案中显式保留该实验/边界条件",
                )
            )
    return out


def check_evidence(qa: QAItem, evidence: str) -> List[Violation]:
    """第三层：答案需可追溯到 evidence span (有引用且有 token 重叠)。"""
    out: List[Violation] = []
    if not qa.source_nodes and not qa.evidence_spans:
        out.append(
            Violation(
                qid=qa.qid,
                layer=VerifyLayer.EVIDENCE.value,
                field="evidence",
                expected="at least one supporting evidence span",
                actual="no evidence cited",
                severity="critical",
                repair_hint="为答案绑定证据 span",
            )
        )
        return out
    # token 重叠粗检
    ans_tokens = set(re.findall(r"[0-9a-zA-Z一-鿿]{2,}", qa.answer.lower()))
    ev_tokens = set(re.findall(r"[0-9a-zA-Z一-鿿]{2,}", evidence.lower()))
    if ans_tokens and not (ans_tokens & ev_tokens):
        out.append(
            Violation(
                qid=qa.qid,
                layer=VerifyLayer.EVIDENCE.value,
                field="evidence",
                expected="answer grounded in evidence",
                actual="no token overlap with evidence",
                severity="major",
                repair_hint="使答案与所引证据 span 内容一致",
            )
        )
    return out


# checker 注册表 (recipe 可选择启用哪些)
CHECKERS = {
    "numerical_check": check_numeric,
    "unit_check": check_unit,
    "condition_check": check_condition,
    "evidence_check": check_evidence,
}
