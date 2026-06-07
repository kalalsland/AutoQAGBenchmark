"""错误扰动构造器 (论文 §5.6)。

每个函数接收原始 QAItem + 图节点池，返回 (corrupted_answer, error_type) 或 None。
通过替换/删除同类型图节点构造 corrupted QA，用于训练 verifier 与验证评价器有效性。
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Tuple

from autoqag.schema import ErrorType, QAItem

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_UNITS = ["%", "MPa", "kPa", "GPa", "K", "°C", "eV", "nm", "µm", "mm", "V", "A", "Hz"]
_GENERALIZERS = [
    " in all cases",
    " under all conditions",
    " regardless of temperature",
    " for any material",
    "（适用于所有情况）",
]


def corrupt_wrong_number(qa: QAItem, pools: Dict[str, List[str]], rng: random.Random) -> Optional[Tuple[str, str]]:
    nums = _NUM_RE.findall(qa.answer)
    if not nums:
        return None
    target = rng.choice(nums)
    try:
        val = float(target)
    except ValueError:
        return None
    # 扰动 20%~200%，确保改变
    factor = rng.choice([0.5, 1.5, 2.0, 10.0, 0.1])
    new_val = val * factor
    new_str = str(int(new_val)) if new_val.is_integer() else f"{new_val:.3g}"
    new_answer = qa.answer.replace(target, new_str, 1)
    if new_answer == qa.answer:
        return None
    return new_answer, ErrorType.WRONG_NUMBER.value


def corrupt_unit_mismatch(qa: QAItem, pools: Dict[str, List[str]], rng: random.Random) -> Optional[Tuple[str, str]]:
    for u in _UNITS:
        if u in qa.answer:
            alt = rng.choice([x for x in _UNITS if x != u])
            return qa.answer.replace(u, alt, 1), ErrorType.UNIT_MISMATCH.value
    return None


def corrupt_entity_swap(qa: QAItem, pools: Dict[str, List[str]], rng: random.Random) -> Optional[Tuple[str, str]]:
    concepts = pools.get("concept", [])
    if len(concepts) < 2:
        return None
    for c in concepts:
        if c and c in qa.answer:
            alt = rng.choice([x for x in concepts if x and x != c] or [c])
            if alt != c:
                return qa.answer.replace(c, alt, 1), ErrorType.ENTITY_SWAP.value
    return None


def corrupt_over_generalization(qa: QAItem, pools: Dict[str, List[str]], rng: random.Random) -> Optional[Tuple[str, str]]:
    # 删除条件 + 追加泛化措辞
    answer = qa.answer
    for cond in qa.constraints.get("condition", []) or []:
        if cond and cond in answer:
            answer = answer.replace(cond, "").strip()
    answer = answer.rstrip(".。") + rng.choice(_GENERALIZERS)
    if answer == qa.answer:
        return None
    return answer, ErrorType.OVER_GENERALIZATION.value


def corrupt_boundary_violation(qa: QAItem, pools: Dict[str, List[str]], rng: random.Random) -> Optional[Tuple[str, str]]:
    conds = qa.constraints.get("condition", []) or []
    if not conds:
        return None
    # 把条件中的数值放大，制造越界
    cond = conds[0]
    nums = _NUM_RE.findall(cond)
    if not nums:
        return None
    target = nums[0]
    try:
        new = str(float(target) * 3)
    except ValueError:
        return None
    new_cond = cond.replace(target, new, 1)
    if cond in qa.answer:
        return qa.answer.replace(cond, new_cond, 1), ErrorType.BOUNDARY_VIOLATION.value
    return qa.answer + f" (condition changed to {new_cond})", ErrorType.BOUNDARY_VIOLATION.value


def corrupt_missing_hop(qa: QAItem, pools: Dict[str, List[str]], rng: random.Random) -> Optional[Tuple[str, str]]:
    if len(qa.evidence_path) < 2:
        return None
    # 截断答案，模拟漏跳；标记错误
    sents = re.split(r"(?<=[.!?。！？])\s+", qa.answer)
    if len(sents) < 2:
        return qa.answer + " (missing intermediate reasoning)", ErrorType.MISSING_HOP.value
    return sents[0], ErrorType.MISSING_HOP.value


def corrupt_unsupported(qa: QAItem, pools: Dict[str, List[str]], rng: random.Random) -> Optional[Tuple[str, str]]:
    # 追加未被证据支持的断言
    extra = rng.choice(
        [
            " This is the highest value ever reported.",
            " The mechanism is fully understood.",
            "（该结论已被所有后续研究证实）",
        ]
    )
    return qa.answer.rstrip() + extra, ErrorType.UNSUPPORTED_ANSWER.value


def corrupt_table_misread(qa: QAItem, pools: Dict[str, List[str]], rng: random.Random) -> Optional[Tuple[str, str]]:
    if qa.question_type != "table":
        return None
    return corrupt_wrong_number(qa, pools, rng) or (
        qa.answer + " (read from wrong row/column)",
        ErrorType.TABLE_MISREAD.value,
    )


def corrupt_formula_misuse(qa: QAItem, pools: Dict[str, List[str]], rng: random.Random) -> Optional[Tuple[str, str]]:
    if qa.question_type != "formula":
        return None
    return qa.answer + " (derived from a different equation)", ErrorType.FORMULA_MISUSE.value


def corrupt_evidence_drift(qa: QAItem, pools: Dict[str, List[str]], rng: random.Random) -> Optional[Tuple[str, str]]:
    # 保持答案，但污染证据 (在 QAItem 层由调用方处理 evidence)；这里给文本提示
    return qa.answer + " [evidence drifted to unrelated span]", ErrorType.EVIDENCE_DRIFT.value


ALL_CORRUPTORS = [
    corrupt_wrong_number,
    corrupt_unit_mismatch,
    corrupt_entity_swap,
    corrupt_over_generalization,
    corrupt_boundary_violation,
    corrupt_missing_hop,
    corrupt_unsupported,
    corrupt_table_misread,
    corrupt_formula_misuse,
    corrupt_evidence_drift,
]
