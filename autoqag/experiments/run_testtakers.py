"""外部"考生"实验 (extrinsic, LLM)：LLM-as-judge 质量评分 + 难度判别力测试。

两类外部评测 (作用于已生成 qa.jsonl，需要 LLM，消耗 API)：
  1) judge      多维 LLM 评审 (faithfulness/grounding/reasoning_depth/specificity/overall)；
  2) discriminate 难度判别力：同一考生模型在 闭卷(仅问题) 与 开卷(问题+证据) 下作答，
                 由裁判模型对照参考答案判对错，按难度分层统计准确率。
                 好的 benchmark 应满足：准确率随难度单调下降，且 开卷>闭卷 (证据有用)。

凭据：优先 AUTOQAG_API_KEY/AUTOQAG_BASE_URL；缺省回退 DASHSCOPE_API_KEY + 兼容端点。

用法：
  python -m autoqag.experiments.run_testtakers --mode judge \
      --work_dir outputs/cmp_after --limit 0
  python -m autoqag.experiments.run_testtakers --mode discriminate \
      --work_dir outputs/cmp_after --taker qwen-plus --limit 0
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional

from autoqag.common.llm import LLMClient
from autoqag.ops.m6_generate.json_utils import parse_json
from autoqag.ops.m8_verify.verifiers import _evidence_text
from autoqag.schema import QAItem

_DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _creds() -> Dict[str, str]:
    key = os.getenv("AUTOQAG_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    base = os.getenv("AUTOQAG_BASE_URL") or os.getenv("OPENAI_BASE_URL") or _DASHSCOPE_BASE
    if not key:
        raise SystemExit("缺少 API Key：请设置 AUTOQAG_API_KEY 或 DASHSCOPE_API_KEY")
    return {"api_key": key, "base_url": base}


def _load_items(work_dir: str, limit: int) -> List[QAItem]:
    path = os.path.join(work_dir, "qa.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if limit and limit > 0:
        rows = rows[:limit]
    return [QAItem.from_dict(r) for r in rows]


# ---------------- 1) LLM-as-judge ----------------
_JUDGE = """你是严格的科研QA质量评审。仅依据EVIDENCE判断，对下面QA在5个维度各打1-5分(整数)。
维度:
- faithfulness 答案是否忠实于证据、无幻觉外推
- grounding 答案是否可由所给证据充分支撑
- reasoning_depth 是否需要跨证据/多步推理(单点查找=低分,多跳整合=高分)
- specificity 是否保留数值/单位/条件等精确约束、不过度泛化
- overall 作为科研benchmark题目的总体质量
只输出JSON: {{"faithfulness":x,"grounding":x,"reasoning_depth":x,"specificity":x,"overall":x}}

QUESTION: {q}
ANSWER: {a}
EVIDENCE:
{e}
"""
_DIMS = ["faithfulness", "grounding", "reasoning_depth", "specificity", "overall"]


def run_judge(work_dir: str, model: str, limit: int) -> Dict[str, Any]:
    items = _load_items(work_dir, limit)
    llm = LLMClient(model=model, **_creds(), max_concurrency=6, json_mode=True)
    prompts = [_JUDGE.format(q=q.question, a=q.answer, e=_evidence_text(q)[:1800]) for q in items]
    resp = llm.generate_batch(prompts)
    acc: Dict[str, List[float]] = {d: [] for d in _DIMS}
    by_diff: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {d: [] for d in _DIMS})
    per_item = []
    for q, r in zip(items, resp):
        o = parse_json(r) or {}
        scores = {}
        for d in _DIMS:
            v = o.get(d)
            if isinstance(v, (int, float)) and 1 <= v <= 5:
                acc[d].append(float(v))
                by_diff[q.difficulty][d].append(float(v))
                scores[d] = float(v)
        per_item.append({"qid": q.qid, "difficulty": q.difficulty, "scores": scores})
    summary = {d: (round(statistics.mean(acc[d]), 3) if acc[d] else None) for d in _DIMS}
    diff_summary = {
        diff: {d: (round(statistics.mean(v[d]), 3) if v[d] else None) for d in _DIMS}
        for diff, v in sorted(by_diff.items())
    }
    return {"n": len(items), "model": model, "overall": summary,
            "by_difficulty": diff_summary, "per_item": per_item}


# ---------------- 2) 难度判别力 (闭卷/开卷) ----------------
_ANSWER_CLOSED = """你是该领域专家。仅凭已有知识回答下面问题，简洁作答，不要编造具体数值。
QUESTION: {q}
你的答案:"""
_ANSWER_OPEN = """你是该领域专家。依据下面提供的证据回答问题，简洁作答。
QUESTION: {q}
EVIDENCE:
{e}
你的答案:"""
_GRADE = """对照参考答案，判断"考生答案"是否在事实与关键约束(数值/单位/条件)上正确。
只输出JSON: {{"correct": true 或 false}}
QUESTION: {q}
参考答案: {ref}
考生答案: {cand}
"""


def _grade_batch(llm: LLMClient, items: List[QAItem], cand: List[str]) -> List[bool]:
    prompts = [_GRADE.format(q=q.question, ref=q.answer, cand=c) for q, c in zip(items, cand)]
    resp = llm.generate_batch(prompts)
    out = []
    for r in resp:
        o = parse_json(r) or {}
        out.append(bool(o.get("correct")))
    return out


def _acc_by_difficulty(items: List[QAItem], correct: List[bool]) -> Dict[str, Any]:
    buckets: Dict[str, List[bool]] = defaultdict(list)
    for q, c in zip(items, correct):
        buckets[q.difficulty].append(c)
    by = {d: round(sum(v) / len(v), 3) for d, v in sorted(buckets.items())}
    overall = round(sum(correct) / max(1, len(correct)), 3)
    # 单调性：按 L1<L2<L3<L4 顺序准确率应非递增
    order = [d for d in ("L1", "L2", "L3", "L4") if d in by]
    seq = [by[d] for d in order]
    monotonic = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
    return {"overall": overall, "by_difficulty": by, "order": order,
            "monotonic_non_increasing": monotonic}


def run_discriminate(work_dir: str, taker: str, judge_model: str, limit: int) -> Dict[str, Any]:
    items = _load_items(work_dir, limit)
    taker_llm = LLMClient(model=taker, **_creds(), max_concurrency=6)
    judge_llm = LLMClient(model=judge_model, **_creds(), max_concurrency=6, json_mode=True)

    closed = taker_llm.generate_batch([_ANSWER_CLOSED.format(q=q.question) for q in items])
    open_ = taker_llm.generate_batch(
        [_ANSWER_OPEN.format(q=q.question, e=_evidence_text(q)[:1800]) for q in items]
    )
    closed_ok = _grade_batch(judge_llm, items, closed)
    open_ok = _grade_batch(judge_llm, items, open_)

    closed_stats = _acc_by_difficulty(items, closed_ok)
    open_stats = _acc_by_difficulty(items, open_ok)
    gap = round(open_stats["overall"] - closed_stats["overall"], 3)
    return {
        "n": len(items), "taker": taker, "judge": judge_model,
        "closed_book": closed_stats, "open_book": open_stats,
        "open_minus_closed_gap": gap,
    }


def _md(payload: Dict[str, Any], mode: str) -> str:
    if mode == "judge":
        lines = ["| dimension | overall |", "|---|---|"]
        for d in _DIMS:
            lines.append(f"| {d} | {payload['overall'].get(d)} |")
        lines.append("\n按难度 (overall 维度)：\n")
        lines.append("| difficulty | " + " | ".join(_DIMS) + " |")
        lines.append("|" + "---|" * (len(_DIMS) + 1))
        for diff, sc in payload["by_difficulty"].items():
            lines.append(f"| {diff} | " + " | ".join(str(sc.get(d)) for d in _DIMS) + " |")
        return "\n".join(lines)
    # discriminate
    cb, ob = payload["closed_book"], payload["open_book"]
    diffs = sorted(set(cb["by_difficulty"]) | set(ob["by_difficulty"]))
    lines = ["| difficulty | closed-book acc | open-book acc |", "|---|---|---|"]
    for d in diffs:
        lines.append(f"| {d} | {cb['by_difficulty'].get(d)} | {ob['by_difficulty'].get(d)} |")
    lines.append(f"| **overall** | {cb['overall']} | {ob['overall']} |")
    lines.append(f"\n开卷-闭卷增益 (证据有用性): **{payload['open_minus_closed_gap']}**")
    lines.append(f"闭卷准确率随难度单调非增: **{cb['monotonic_non_increasing']}**；"
                 f"开卷: **{ob['monotonic_non_increasing']}**")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["judge", "discriminate"], default="judge")
    ap.add_argument("--work_dir", default="outputs/cmp_after")
    ap.add_argument("--taker", default="qwen-plus", help="考生模型 (discriminate 模式)")
    ap.add_argument("--judge_model", default="qwen-plus", help="裁判模型")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 条 (0=全部)；冒烟用小值省 API")
    ap.add_argument("--out", default="autoqag/experiments/results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.mode == "judge":
        payload = run_judge(args.work_dir, args.judge_model, args.limit)
        stem = "testtaker_judge"
    else:
        payload = run_discriminate(args.work_dir, args.taker, args.judge_model, args.limit)
        stem = "testtaker_discriminate"

    with open(os.path.join(args.out, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    md = _md(payload, args.mode)
    with open(os.path.join(args.out, stem + ".md"), "w", encoding="utf-8") as f:
        f.write(f"# 外部考生实验 ({args.mode})  work_dir={args.work_dir}\n\n" + md + "\n")
    print(md)
    print(f"\n结果写入 {args.out}/{stem}.{{json,md}}")


if __name__ == "__main__":
    main()
