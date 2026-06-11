"""CF-Probe：约束保真度解耦实验 (NMI 主 finding 的最小验证 pilot)。

主命题 (创新点分析_v2.md §10)：
  约束保真度 (Constraint-Faithfulness, CF) 是被现有评测折叠掉的独立能力轴；
  在科研 QA 上，模型的通用能力 / 排行榜名次几乎不能预测、甚至误导其 CF。

为何只有本仪器能测：现有 benchmark 把"流畅+相关+正确"压成 judge 的一个分；
本数据把数值/单位/条件约束带出处抽了出来，因此可以 **确定性、零 LLM** 地核对
"模型有没有守住约束"，把那个分劈成两个独立轴：
  - 轴 A (judge-overall)        = 现有评测奖励的"流畅/相关"代理
  - 轴 B (CF, 确定性核对)        = 只有本仪器能单独测的约束保真度

开卷设置 (给模型 问题+金标准证据)：去掉"知不知道"混杂，失败=不守约束而非不懂
(对应配菜 B "有证据≠守证据")，且对强模型最致命。

三个测试 + 自动 GO/NO-GO：
  T1 解耦      : Spearman( judge_overall , CF )  → 预测 ≈0 或负
  T2 排名倒挂  : Spearman( 预注册外部能力排名 , 各模型平均 CF ) → 预测偏低/负
  T3 越强越危险: judge 误放行率 (CF<1 却 judge_faithfulness>=4) 是否随能力上升

复用：common.llm.LLMClient、m8_verify.verifiers(_nums_close/is_refusal)、
      m3_normalize.units(extract_value_units/units_compatible)、run_testtakers 的 _JUDGE。

用法：
  # ceiling 冒烟 (2 模型、小样本，先量天花板)
  PYTHONIOENCODING=utf-8 python -m autoqag.experiments.run_cf_probe \
      --models qwen-turbo,qwen-max --subset multi --limit 6 --no_judge

  # 全量 (出 T1/T2/T3 + 判定)
  PYTHONIOENCODING=utf-8 python -m autoqag.experiments.run_cf_probe \
      --models qwen-turbo,deepseek-v3,qwen-plus,qwen-max,deepseek-r1 --subset multi
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from autoqag.common.llm import LLMClient
from autoqag.ops.m3_normalize.units import extract_value_units, units_compatible
from autoqag.ops.m6_generate.json_utils import parse_json
from autoqag.ops.m8_verify.verifiers import _NUM_RE, _evidence_text, _nums_close, is_refusal
from autoqag.schema import QAItem

_DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# GPT/Gemini/Claude 代理 (OpenAI 兼容)。密钥/地址走环境变量，勿写明文。
_PROXY_BASE = os.getenv("AUTOQAG_BASE_URL") or os.getenv("PROXY_BASE_URL", "")
_PROXY_KEY = os.getenv("AUTOQAG_API_KEY") or os.getenv("PROXY_API_KEY", "")


def _dashscope_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("AUTOQAG_API_KEY")
    if not key:
        raise SystemExit("缺少 DASHSCOPE_API_KEY")
    return key


# ---- 模型注册表：rank = 预注册的外部通用能力 tier (T2 用，看 CF 前钉死，防循环论证) ----
# 粗粒度 tier (依据公开 LMArena / 常识，Spearman 容许并列)：
#   1 小模型 · 2 中开源 · 3 大开源 · 4 前沿闭源 · 5 顶级/推理。
def _registry() -> Dict[str, Dict[str, Any]]:
    dk = _dashscope_key()
    P = {"base_url": _PROXY_BASE, "api_key": _PROXY_KEY}
    D = {"base_url": _DASHSCOPE_BASE, "api_key": dk}
    return {
        "qwen-turbo":  {"model": "qwen-turbo",  **D, "rank": 1},
        "deepseek-v3": {"model": "deepseek-v3", **D, "rank": 2},
        "qwen-plus":   {"model": "qwen-plus",   **D, "rank": 2},
        "qwen-max":    {"model": "qwen-max",    **D, "rank": 3},
        "gpt-4.1":     {"model": "gpt-4.1",     **P, "rank": 4},
        "gemini-2.5-pro": {"model": "gemini-2.5-pro", **P, "rank": 4},
        "claude-sonnet-4-5": {"model": "claude-sonnet-4-5-20250929", **P, "rank": 4},
        "gpt-5":       {"model": "gpt-5",       **P, "rank": 5},
        "deepseek-r1": {"model": "deepseek-r1", **D, "rank": 5},  # reasoning
    }


# ---------------- 题集加载与子集筛选 ----------------
def _hard_counts(qa: QAItem) -> Tuple[int, int]:
    c = qa.constraints or {}
    return len(c.get("number", []) or []), len(c.get("unit", []) or [])


def _load_items(work_dir: str, subset: str, limit: int) -> List[QAItem]:
    rows = [json.loads(l) for l in open(os.path.join(work_dir, "qa.jsonl"), encoding="utf-8")]
    items = [QAItem.from_dict(r) for r in rows]
    # 过滤退化样本：拒答 / 无证据
    items = [q for q in items if not is_refusal(q.answer) and q.evidence_spans]
    out = []
    for q in items:
        nn, nu = _hard_counts(q)
        if subset == "multi":
            # 判别力子集：比较题 或 含 >=2 个硬约束 (number+unit)
            if q.question_type == "comparative" or (nn + nu) >= 2:
                out.append(q)
        elif subset == "numeric":
            if nn + nu >= 1:
                out.append(q)
        else:  # all
            out.append(q)
    if limit and limit > 0:
        out = out[:limit]
    return out


# ---------------- 轴 B：确定性 CF 核对器 (零 LLM) ----------------
_RANGE_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*[~～\-–—]\s*([-+]?\d+(?:\.\d+)?)")


def _parse_range(s: str):
    """'1.31~ 1.46' / '2.8~3.4' → (lo, hi)；非区间返回 None。"""
    m = _RANGE_RE.search(str(s))
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    return (min(a, b), max(a, b))


def _num_reproduced(gold: str, ans_nums: List[str]) -> bool:
    """模型答案是否复现了金标准数值 (区间感知)。
    - 区间 a~b：答案含落入 [a,b] 的数 (含 2% 容差边界)，或同时含两端点；
    - 单值     ：答案含 2% 容差内的数。
    """
    rng = _parse_range(gold)
    if rng:
        lo, hi = rng
        lo_t, hi_t = lo * 0.98, hi * 1.02
        for a in ans_nums:
            try:
                x = float(a)
            except ValueError:
                continue
            if lo_t <= x <= hi_t:
                return True
        return _nums_close(str(lo), _closest(lo, ans_nums)) and _nums_close(str(hi), _closest(hi, ans_nums))
    return any(_nums_close(gold, a) for a in ans_nums)


def _closest(target: float, ans_nums: List[str]) -> str:
    best, bd = "", float("inf")
    for a in ans_nums:
        try:
            x = float(a)
        except ValueError:
            continue
        if abs(x - target) < bd:
            bd, best = abs(x - target), a
    return best


def cf_check(qa: QAItem, answer: str) -> Dict[str, Any]:
    """核对模型答案是否守住金标准约束。

    硬化点 (v2)：
    - 数值召回区间感知 (修复 '1.31~1.46' 这类区间被误判为漏)；
    - 单位**不**取自有 bug 的扁平 constraints.unit，而从干净的金标准答案文本按
      value-unit 邻接派生 (extract_value_units)，避免把游离单位当成漏单位。
    返回 cf_hard(数值召回) / cf_unit(单位保真) / cf_all。
    """
    c = qa.constraints or {}
    gold_nums = [str(x) for x in (c.get("number", []) or [])]
    gold_conds = [str(x) for x in (c.get("condition", []) or [])]
    ans_nums = _NUM_RE.findall(answer or "")
    ans_units = {u for _, u in extract_value_units(answer or "") if u}

    # 数值召回 (区间感知)
    num_flags = [_num_reproduced(gn, ans_nums) for gn in gold_nums]
    cf_hard = (sum(num_flags) / len(num_flags)) if num_flags else None

    # 单位保真：金标准答案里"带单位的数值"，模型答案是否保留了同一单位
    gold_vu = [(v, u) for v, u in extract_value_units(qa.answer or "") if u]
    unit_flags = [any(u == au or units_compatible(u, au) for au in ans_units) or (u in (answer or ""))
                  for _, u in gold_vu]
    cf_unit = (sum(unit_flags) / len(unit_flags)) if unit_flags else None

    # 条件保真：关键词重叠
    cond_flags = []
    for gc in gold_conds:
        key = [k for k in re.sub(r"[^0-9a-zA-Z一-鿿]+", " ", gc).split() if len(k) > 1]
        cond_flags.append(bool(key) and any(k in (answer or "") for k in key))

    all_elems = num_flags + unit_flags + cond_flags
    cf_all = (sum(all_elems) / len(all_elems)) if all_elems else None
    return {
        "cf_hard": cf_hard, "cf_unit": cf_unit, "cf_all": cf_all,
        "n_hard": len(num_flags),
        "violation": (cf_hard is not None and cf_hard < 1.0),
        "detail": {"numbers": list(zip(gold_nums, num_flags)),
                   "gold_value_units": list(zip([u for _, u in gold_vu], unit_flags)),
                   "condition": list(zip(gold_conds, cond_flags))},
    }


# ---------------- 轴 A：judge (复用 run_testtakers 的维度) ----------------
_JUDGE = """你是严格的科研QA质量评审。仅依据EVIDENCE判断，对下面"考生答案"在4个维度各打1-5分(整数)。
维度:
- faithfulness 答案是否忠实于证据、无幻觉外推
- grounding 答案是否可由所给证据充分支撑
- specificity 是否保留数值/单位/条件等精确约束、不过度泛化
- overall 作为科研答案的总体质量
只输出JSON: {{"faithfulness":x,"grounding":x,"specificity":x,"overall":x}}

QUESTION: {q}
考生答案: {a}
EVIDENCE:
{e}
"""
_JDIMS = ["faithfulness", "grounding", "specificity", "overall"]

_ANSWER_OPEN = """你是该领域专家。依据下面提供的证据回答问题，简洁作答，准确保留数值/单位/条件等约束。
QUESTION: {q}
EVIDENCE:
{e}
你的答案:"""


def _make_client(spec: Dict[str, Any], json_mode: bool = False) -> LLMClient:
    return LLMClient(
        model=spec["model"], api_key=spec["api_key"], base_url=spec["base_url"],
        max_concurrency=6, json_mode=json_mode, max_tokens=2048, temperature=0.0,
    )


# ---------------- 统计：Spearman (无 scipy 依赖) ----------------
def _rank(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: List[float], b: List[float]) -> Optional[float]:
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return None
    return cov / (va ** 0.5 * vb ** 0.5)


def spearman(a: List[float], b: List[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xa, xb = [p[0] for p in pairs], [p[1] for p in pairs]
    r = _pearson(_rank(xa), _rank(xb))
    return round(r, 3) if r is not None else None


# ---------------- 主流程 ----------------
def run(work_dir: str, model_keys: List[str], subset: str, limit: int,
        do_judge: bool, judge_key: str) -> Dict[str, Any]:
    items = _load_items(work_dir, subset, limit)
    if not items:
        raise SystemExit(f"子集 {subset} 下无可用题目")
    reg = _registry()
    for m in model_keys + ([judge_key] if do_judge else []):
        if m not in reg:
            raise SystemExit(f"未知模型 {m}；可选: {list(reg)}")

    evidences = [_evidence_text(q)[:1800] for q in items]
    # 固定裁判 (代表"全领域默认 LLM-as-judge"；对所有考生用同一把尺，T3 才能问它是否偏袒流畅模型)
    judge = _make_client(reg[judge_key], json_mode=True) if do_judge else None
    per_model: Dict[str, Any] = {}
    cells: List[Dict[str, Any]] = []  # 每个 (model,item) 一格

    for mk in model_keys:
        spec = reg[mk]
        taker = _make_client(spec, json_mode=False)
        answers = taker.generate_batch(
            [_ANSWER_OPEN.format(q=q.question, e=e) for q, e in zip(items, evidences)]
        )
        # 轴 B：确定性 CF
        cf = [cf_check(q, a) for q, a in zip(items, answers)]
        # 轴 A：固定裁判打分
        jscores: List[Dict[str, Optional[float]]] = [{} for _ in items]
        if do_judge:
            jprompts = [_JUDGE.format(q=q.question, a=a, e=e)
                        for q, a, e in zip(items, answers, evidences)]
            jresp = judge.generate_batch(jprompts)
            for i, r in enumerate(jresp):
                o = parse_json(r) or {}
                jscores[i] = {d: (float(o[d]) if isinstance(o.get(d), (int, float)) and 1 <= o[d] <= 5 else None)
                              for d in _JDIMS}

        cf_hard_vals = [c["cf_hard"] for c in cf if c["cf_hard"] is not None]
        cf_unit_vals = [c["cf_unit"] for c in cf if c["cf_unit"] is not None]
        cf_all_vals = [c["cf_all"] for c in cf if c["cf_all"] is not None]
        viol = sum(1 for c in cf if c["violation"])
        per_model[mk] = {
            "rank": spec["rank"], "n": len(items),
            "cf_hard_mean": round(statistics.mean(cf_hard_vals), 3) if cf_hard_vals else None,
            "cf_unit_mean": round(statistics.mean(cf_unit_vals), 3) if cf_unit_vals else None,
            "cf_all_mean": round(statistics.mean(cf_all_vals), 3) if cf_all_vals else None,
            "violation_rate": round(viol / len(items), 3),
            "judge_overall_mean": (round(statistics.mean([j["overall"] for j in jscores if j.get("overall") is not None]), 3)
                                   if do_judge else None),
        }
        for q, a, c, j in zip(items, answers, cf, jscores):
            cells.append({
                "model": mk, "rank": spec["rank"], "qid": q.qid,
                "qtype": q.question_type, "difficulty": q.difficulty,
                "cf_hard": c["cf_hard"], "cf_unit": c["cf_unit"], "cf_all": c["cf_all"], "violation": c["violation"],
                "judge_overall": j.get("overall"), "judge_faithfulness": j.get("faithfulness"),
                "answer": (a or "")[:500],
            })

    # ---- T1 解耦：per-cell Spearman(judge_overall, cf_hard) ----
    t1 = None
    if do_judge:
        jo = [c["judge_overall"] for c in cells]
        cf = [c["cf_hard"] for c in cells]
        t1 = spearman(jo, cf)

    # ---- T2 排名倒挂：Spearman(预注册 rank, 各模型 cf_hard_mean) ----
    ranks = [per_model[m]["rank"] for m in model_keys if per_model[m]["cf_hard_mean"] is not None]
    cfm = [per_model[m]["cf_hard_mean"] for m in model_keys if per_model[m]["cf_hard_mean"] is not None]
    t2 = spearman([float(r) for r in ranks], cfm)

    # ---- T3 越强越危险：judge 误放行率 vs 能力 rank ----
    t3 = None
    fa_by_model = {}
    if do_judge:
        for mk in model_keys:
            sub = [c for c in cells if c["model"] == mk and c["violation"]]
            fa = [c for c in sub if (c["judge_faithfulness"] or 0) >= 4]
            fa_by_model[mk] = {
                "n_violations": len(sub),
                "false_accept_rate": round(len(fa) / len(sub), 3) if sub else None,
            }
        rk = [per_model[m]["rank"] for m in model_keys if fa_by_model[m]["false_accept_rate"] is not None]
        fr = [fa_by_model[m]["false_accept_rate"] for m in model_keys if fa_by_model[m]["false_accept_rate"] is not None]
        t3 = spearman([float(x) for x in rk], fr)

    verdict = _verdict(t1, t2, t3)
    return {
        "work_dir": work_dir, "subset": subset, "n_items": len(items),
        "models": model_keys, "do_judge": do_judge, "judge_model": judge_key if do_judge else None,
        "per_model": per_model,
        "tests": {
            "T1_decoupling_spearman_judgeoverall_vs_cf": t1,
            "T2_inversion_spearman_caprank_vs_cf": t2,
            "T3_danger_spearman_caprank_vs_falseaccept": t3,
            "T3_false_accept_by_model": fa_by_model,
        },
        "verdict": verdict,
        "cells": cells,
    }


def _verdict(t1: Optional[float], t2: Optional[float], t3: Optional[float]) -> Dict[str, Any]:
    signals = []
    if t1 is not None and t1 <= 0.2:
        signals.append(f"T1 解耦：judge-overall 与 CF 几乎不相关 (ρ={t1})")
    if t2 is not None and t2 <= 0.5:
        signals.append(f"T2 倒挂：通用能力排名预测不了 CF 排名 (ρ={t2})")
    if t3 is not None and t3 >= 0.3:
        signals.append(f"T3 越强越危险：judge 误放行率随能力上升 (ρ={t3})")
    go = len(signals) >= 1
    return {
        "GO": go,
        "decision": ("GO —— 存在 NMI 级信号，值得向老师申请资源扩规模" if go
                     else "NO-GO —— 三测试均不显著，CF 未表现为独立轴；建议转 ACL/NeurIPS D&B"),
        "positive_signals": signals,
    }


def _md(p: Dict[str, Any]) -> str:
    L = [f"# CF-Probe  work_dir={p['work_dir']}  subset={p['subset']}  n={p['n_items']}",
         "",
         "## 各模型 (按预注册能力 rank 排序)",
         "| model | cap_rank | CF_hard(数值) | CF_unit(单位) | CF_all | 违约率 | judge_overall |",
         "|---|---|---|---|---|---|---|"]
    for m in sorted(p["models"], key=lambda x: p["per_model"][x]["rank"]):
        d = p["per_model"][m]
        L.append(f"| {m} | {d['rank']} | {d['cf_hard_mean']} | {d.get('cf_unit_mean')} | {d['cf_all_mean']} | "
                 f"{d['violation_rate']} | {d['judge_overall_mean']} |")
    t = p["tests"]
    L += ["", "## 三测试",
          f"- **T1 解耦** Spearman(judge_overall, CF_hard) = `{t['T1_decoupling_spearman_judgeoverall_vs_cf']}`  (预测≈0/负 → 解耦)",
          f"- **T2 倒挂** Spearman(能力rank, CF_hard) = `{t['T2_inversion_spearman_caprank_vs_cf']}`  (预测低/负 → 排名失效)",
          f"- **T3 越强越危险** Spearman(能力rank, judge误放行率) = `{t['T3_danger_spearman_caprank_vs_falseaccept']}`  (预测正 → 强模型错误更隐蔽)"]
    if t["T3_false_accept_by_model"]:
        L += ["", "judge 误放行率 (CF<1 却 faithfulness>=4)：",
              "| model | cap_rank | #违约题 | 误放行率 |", "|---|---|---|---|"]
        for m in sorted(p["models"], key=lambda x: p["per_model"][x]["rank"]):
            fa = t["T3_false_accept_by_model"].get(m, {})
            L.append(f"| {m} | {p['per_model'][m]['rank']} | {fa.get('n_violations')} | {fa.get('false_accept_rate')} |")
    v = p["verdict"]
    L += ["", "## 判定", f"**{v['decision']}**"]
    for s in v["positive_signals"]:
        L.append(f"- ✅ {s}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="outputs/cmp_after")
    ap.add_argument("--models", default="qwen-turbo,qwen-max")
    ap.add_argument("--subset", choices=["multi", "numeric", "all"], default="multi")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 题 (冒烟用)")
    ap.add_argument("--no_judge", action="store_true", help="冒烟时跳过 judge，只量 CF 天花板")
    ap.add_argument("--judge_model", default="qwen-max", help="固定裁判 (代表默认 LLM-as-judge)")
    ap.add_argument("--out", default="autoqag/experiments/results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    payload = run(args.work_dir, [m.strip() for m in args.models.split(",") if m.strip()],
                  args.subset, args.limit, do_judge=not args.no_judge, judge_key=args.judge_model)
    with open(os.path.join(args.out, "cf_probe.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    md = _md(payload)
    with open(os.path.join(args.out, "cf_probe.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n结果写入 {args.out}/cf_probe.{{json,md}}")


if __name__ == "__main__":
    main()
