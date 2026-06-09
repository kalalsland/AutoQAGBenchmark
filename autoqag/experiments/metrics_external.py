"""外部对比指标 (extrinsic, 作用于已生成的 QA 产物)。

与 metrics_internal 的"无 LLM、作用于 question_plans"不同，本模块度量最终落地的
QA 数据集质量：结构有效性、验证器通过率与逐层违规、证据接地、题型/难度分布。
这些指标既可单点报告，也可在两个 work_dir 间做 before/after 对比。

用法：
  python -m autoqag.experiments.metrics_external \
      --before outputs/cmp_before --after outputs/cmp_after
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from autoqag.common.io import read_jsonl_list
from autoqag.schema import QuestionType, Difficulty


def _load(work_dir: str, rel: str) -> List[Dict[str, Any]]:
    p = os.path.join(work_dir, rel)
    return read_jsonl_list(p) if os.path.exists(p) else []


def _avg(xs: List[float]) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def _papers(q: Dict[str, Any]) -> List[str]:
    pl = q.get("paper_id_list") or []
    if pl:
        return pl
    seen = []
    for s in q.get("evidence_spans", []) or []:
        pid = (s.get("address") or {}).get("paper_id", "")
        if pid and pid not in seen:
            seen.append(pid)
    return seen


def structural_validity(qa: List[Dict[str, Any]]) -> Dict[str, Any]:
    """结构有效性：有非空问题、答案，且至少一条证据 span。"""
    n = len(qa)
    has_q = sum(1 for q in qa if (q.get("question") or "").strip())
    has_a = sum(1 for q in qa if (q.get("answer") or "").strip())
    has_ev = sum(1 for q in qa if (q.get("evidence_spans") or []))
    valid = sum(
        1 for q in qa
        if (q.get("question") or "").strip()
        and (q.get("answer") or "").strip()
        and (q.get("evidence_spans") or [])
    )
    return {
        "n_qa": n,
        "answerable_rate": round(has_a / max(1, n), 3),
        "has_question_rate": round(has_q / max(1, n), 3),
        "has_evidence_rate": round(has_ev / max(1, n), 3),
        "valid_qa_rate": round(valid / max(1, n), 3),
    }


def verify_quality(qa: List[Dict[str, Any]], violations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """验证器通过率 + 逐层/逐严重度违规密度。

    pass_rate         validator_result.passed 为真的比例 (问题语义/约束/证据全过)。
    violation_density 每题平均违规数 (越低越好)。
    per_layer_clean   各验证层"零违规"的题目比例 (numerical/unit/condition/evidence/semantic)。
    """
    n = len(qa)
    passed = sum(1 for q in qa if (q.get("validator_result") or {}).get("passed"))
    # 逐题违规层 (取自每题 validator_result.violations，回退到全局 violations.jsonl)
    by_layer = defaultdict(int)
    by_sev = defaultdict(int)
    qids_with_layer = defaultdict(set)
    total_v = 0
    rows = []
    for q in qa:
        vr = q.get("validator_result") or {}
        rows.extend(vr.get("violations") or [])
    if not rows:
        rows = violations
    for v in rows:
        total_v += 1
        layer = v.get("layer", "?")
        by_layer[layer] += 1
        by_sev[v.get("severity", "?")] += 1
        if v.get("qid"):
            qids_with_layer[layer].add(v["qid"])
    layers = ["numerical", "unit", "constraint", "condition", "evidence", "semantic"]
    per_layer_clean = {
        L: round(1 - len(qids_with_layer.get(L, set())) / max(1, n), 3)
        for L in layers if L in by_layer or True
    }
    return {
        "verify_pass_rate": round(passed / max(1, n), 3),
        "violation_density": round(total_v / max(1, n), 3),
        "violations_by_layer": dict(by_layer),
        "violations_by_severity": dict(by_sev),
        "per_layer_clean_rate": per_layer_clean,
    }


def evidence_grounding(qa: List[Dict[str, Any]]) -> Dict[str, Any]:
    """证据接地：每题平均证据 span 数、跨论文题占比、平均涉及论文数。"""
    spans = [len(q.get("evidence_spans") or []) for q in qa]
    npap = [len(_papers(q)) for q in qa]
    cross_paper = sum(1 for q in qa if len(_papers(q)) >= 2)
    return {
        "avg_evidence_spans": _avg(spans),
        "avg_papers_per_qa": _avg(npap),
        "cross_paper_qa": cross_paper,
        "cross_paper_ratio": round(cross_paper / max(1, len(qa)), 3),
    }


def coverage(qa: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_type = defaultdict(int)
    by_diff = defaultdict(int)
    for q in qa:
        by_type[q.get("question_type", "")] += 1
        by_diff[q.get("difficulty", "")] += 1
    all_types = [t.value for t in QuestionType]
    covered = sum(1 for t in all_types if by_type.get(t, 0) > 0)
    return {
        "type_coverage": round(covered / len(all_types), 3),
        "n_types": covered,
        "by_type": dict(by_type),
        "by_difficulty": dict(sorted(by_diff.items())),
    }


def compute_external(work_dir: str) -> Dict[str, Any]:
    qa = _load(work_dir, "qa.jsonl")
    violations = _load(work_dir, "violations.jsonl")
    return {
        "work_dir": work_dir,
        "structural": structural_validity(qa),
        "verify": verify_quality(qa, violations),
        "evidence": evidence_grounding(qa),
        "coverage": coverage(qa),
    }


# ---- 扁平化 + before/after 对比表 ----
def _flatten(ext: Dict[str, Any]) -> Dict[str, Any]:
    s, v, e, c = ext["structural"], ext["verify"], ext["evidence"], ext["coverage"]
    return {
        "n_qa": s["n_qa"],
        "valid_qa_rate": s["valid_qa_rate"],
        "verify_pass_rate": v["verify_pass_rate"],
        "violation_density": v["violation_density"],
        "avg_evidence_spans": e["avg_evidence_spans"],
        "cross_paper_qa": e["cross_paper_qa"],
        "type_coverage": c["type_coverage"],
        "n_types": c["n_types"],
    }


_METRIC_ORDER = [
    "n_qa", "valid_qa_rate", "verify_pass_rate", "violation_density",
    "avg_evidence_spans", "cross_paper_qa", "type_coverage", "n_types",
]


def _compare_md(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    b, a = _flatten(before), _flatten(after)
    head = "| metric | before | after | Δ |\n|---|---|---|---|"
    lines = [head]
    for m in _METRIC_ORDER:
        bv, av = b[m], a[m]
        try:
            d = round(av - bv, 3)
            d = f"+{d}" if d >= 0 else str(d)
        except TypeError:
            d = "-"
        lines.append(f"| {m} | {bv} | {av} | {d} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="outputs/cmp_before")
    ap.add_argument("--after", default="outputs/cmp_after")
    ap.add_argument("--out", default="autoqag/experiments/results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    before = compute_external(args.before)
    after = compute_external(args.after)
    payload = {"before": before, "after": after}
    with open(os.path.join(args.out, "external_compare.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md = _compare_md(before, after)
    with open(os.path.join(args.out, "external_compare.md"), "w", encoding="utf-8") as f:
        f.write("# 外部对比指标 (extrinsic: 无子图层 vs 完整系统)\n\n" + md + "\n")
    print(md)
    print(f"\n结果写入 {args.out}/external_compare.{{json,md}}")


if __name__ == "__main__":
    main()
