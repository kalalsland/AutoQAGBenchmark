"""指标计算 (论文 §6.3 / §6.4 中可直接从产物计算的部分)。

用法：python -m autoqag.experiments.metrics --work_dir outputs/mvp
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Any, Dict, List

from autoqag.common.io import read_jsonl_list
from autoqag.schema import QuestionType, Difficulty


def type_coverage(qa_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """实验三：题型覆盖与难度分布 (论文 §6.3)。"""
    by_type: Dict[str, int] = defaultdict(int)
    by_diff: Dict[str, int] = defaultdict(int)
    for q in qa_rows:
        by_type[q.get("question_type", "")] += 1
        by_diff[q.get("difficulty", "")] += 1
    all_types = [t.value for t in QuestionType]
    covered = sum(1 for t in all_types if by_type.get(t, 0) > 0)
    return {
        "type_coverage": round(covered / len(all_types), 3),
        "by_type": dict(by_type),
        "by_difficulty": dict(by_diff),
        "difficulty_levels_covered": sorted(
            d for d in by_diff if d in {x.value for x in Difficulty}
        ),
    }


def validator_detection(verifier_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """实验四：约束验证检出能力 (论文 §6.4)。

    用 corpus/verifier.jsonl 中 good/bad 标签近似 ground truth；
    这里报告样本构成，真实 TPR/TNR 需把验证器跑在这些样本上 (TODO)。
    """
    by_label: Dict[str, int] = defaultdict(int)
    by_error: Dict[str, int] = defaultdict(int)
    for r in verifier_rows:
        by_label[r.get("label", "")] += 1
        if r.get("error_type"):
            by_error[r["error_type"]] += 1
    return {"by_label": dict(by_label), "by_error_type": dict(by_error)}


def repair_effectiveness(repair_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """实验五：修复有效性 (论文 §6.5)。"""
    if not repair_rows:
        return {"repair_success_rate": None}
    success = sum(1 for r in repair_rows if r.get("success"))
    rounds = [len(r.get("steps", [])) for r in repair_rows]
    return {
        "n_repaired_attempts": len(repair_rows),
        "repair_success_rate": round(success / len(repair_rows), 3),
        "avg_repair_rounds": round(sum(rounds) / len(rounds), 2) if rounds else 0,
    }


def compute_all(work_dir: str) -> Dict[str, Any]:
    def _load(rel: str) -> List[Dict[str, Any]]:
        p = os.path.join(work_dir, rel)
        return read_jsonl_list(p) if os.path.exists(p) else []

    qa = _load("qa.jsonl")
    return {
        "type_difficulty": type_coverage(qa),
        "validator": validator_detection(_load("corpus/verifier.jsonl")),
        "repair": repair_effectiveness(_load("corpus/repair.jsonl")),
        "qa_pass_rate": round(
            sum(1 for q in qa if (q.get("validator_result") or {}).get("passed"))
            / max(1, len(qa)),
            3,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default="outputs/mvp")
    args = parser.parse_args()
    import json

    print(json.dumps(compute_all(args.work_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
