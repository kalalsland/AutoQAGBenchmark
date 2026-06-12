"""消融实验运行器 (内部/intrinsic 部分，无需 LLM)。

逐步开启子图规划各模块，在同一张固定图谱上重新规划，度量内部指标，
证明各模块逐步有效 (A0 baseline → A6 完整规划)。

  A0  baseline_sample      纯物理模板采样 (无问题级语义覆盖层)
  A1  role_plan            +角色 schema 规划 (其余模块关闭)
  A2  score_guided         +评分引导扩展 Accept(v)
  A3  binding              +题型专属语义绑定
  A4  overlay              +虚拟逻辑补全 Ωq
  A5  dual_multihop        +双重多跳 & 难度封顶
  A6  full_plan            +逻辑充分性门槛 = 完整规划

用法：
  PYTHONIOENCODING=utf-8 python -m autoqag.experiments.run_ablation \
      --graph_dir outputs/five --per_type 12

只读源图谱 (nodes/edges)，结果写入各自 work_dir 与 results 目录，不触碰源目录。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Tuple

from autoqag.common.io import ensure_dir, read_jsonl_list
from autoqag.ops.base import PipelineContext
# 导入即触发 STAGES 注册并注入 _name
from autoqag.ops.m5_sample.sample import SampleStage
from autoqag.ops.m5_sample.semantic_plan import SemanticPlanStage
from autoqag.experiments.metrics_internal import compute_internal, load_view

# 累积开关顺序：逐行向右多开一个模块
_ORDER = [
    "use_score_guided", "use_binding", "use_overlay",
    "use_core_edges", "use_compat_gate", "use_dual_multihop", "use_sufficiency",
]


def _toggles(n_on: int) -> Dict[str, bool]:
    return {k: (i < n_on) for i, k in enumerate(_ORDER)}


def ablation_configs(per_type: int) -> List[Tuple[str, str, Dict[str, Any]]]:
    base = {"per_type": per_type}
    sp = lambda n: {**base, **_toggles(n)}  # noqa: E731
    return [
        ("A0_baseline_sample", "sample", dict(base)),
        ("A1_role_plan", "semantic_plan", sp(0)),
        ("A2_score_guided", "semantic_plan", sp(1)),
        ("A3_binding", "semantic_plan", sp(2)),
        ("A4_overlay", "semantic_plan", sp(3)),
        ("A5_core_edges", "semantic_plan", sp(4)),
        ("A6_compat_gate", "semantic_plan", sp(5)),
        ("A7_dual_multihop", "semantic_plan", sp(6)),
        ("A8_full_plan", "semantic_plan", sp(7)),
    ]


_STAGE_CLS = {"sample": SampleStage, "semantic_plan": SemanticPlanStage}


def run_one(name: str, stage_key: str, params: Dict[str, Any],
            graph_dir: str, runs_root: str, domain: str) -> Dict[str, Any]:
    work_dir = ensure_dir(os.path.join(runs_root, name))
    # 复制固定图谱 (只读源)，清掉旧产物保证干净重跑
    for f in ("nodes.jsonl", "edges.jsonl"):
        shutil.copyfile(os.path.join(graph_dir, f), os.path.join(work_dir, f))
    for f in ("question_plans.jsonl", "semantic_memory.json"):
        p = os.path.join(work_dir, f)
        if os.path.exists(p):
            os.remove(p)

    ctx = PipelineContext(work_dir=work_dir, global_params={"domain": domain})
    stage = _STAGE_CLS[stage_key](**params)
    stats = stage.run(ctx) or {}

    plans = read_jsonl_list(os.path.join(work_dir, "question_plans.jsonl"))
    view = load_view(work_dir)
    metrics = compute_internal(plans, view)
    return {"name": name, "stage": stage_key, "params": params,
            "run_stats": stats, "metrics": metrics}


def _flatten_row(r: Dict[str, Any]) -> Dict[str, Any]:
    m = r["metrics"]
    cs, mh, lg, bd = m["coverage_structure"], m["multihop"], m["logical"], m["binding"]
    def _b(x):
        return x.get("rate") if isinstance(x, dict) else None
    return {
        "config": r["name"],
        "n_plans": cs["n_plans"],
        "types": cs["n_types_covered"],
        "avg_ev": cs["avg_evidence_spans"],
        "cross_chunk": mh["real_cross_chunk_ratio"],
        "pseudo_mh": mh["pseudo_multihop_rate"],
        "cross_paper": mh["n_cross_paper_plans"],
        "role_compl": lg["role_completeness"],
        "utility": lg["avg_utility"],
        "overlay_grnd": lg["overlay_grounded_ratio"],
        "comp_bind": _b(bd["comparative_value_object_bind"]),
        "cp_result": _b(bd["cross_paper_result_cross"]),
        "cp_inst": _b(bd["cross_paper_instance_real"]),
        "unit_grnd": _b(bd["unit_grounded"]),
    }


def _markdown(rows: List[Dict[str, Any]]) -> str:
    cols = ["config", "n_plans", "types", "avg_ev", "cross_chunk", "pseudo_mh",
            "cross_paper", "role_compl", "utility", "overlay_grnd",
            "comp_bind", "cp_result", "cp_inst", "unit_grnd"]
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def _pin_hashseed() -> None:
    """候选集合用 set 组织，其字符串迭代序受 PYTHONHASHSEED 影响，会令
    各配置间的候选/绑定选择产生跨进程抖动。消融需可复现，故固定后重启解释器。"""
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", "autoqag.experiments.run_ablation", *sys.argv[1:]])


def main() -> None:
    _pin_hashseed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", default="outputs/five", help="提供 nodes.jsonl/edges.jsonl 的图谱目录")
    ap.add_argument("--per_type", type=int, default=12)
    ap.add_argument("--domain", default="metamaterials")
    ap.add_argument("--runs_root", default="autoqag/experiments/runs")
    ap.add_argument("--out", default="autoqag/experiments/results")
    args = ap.parse_args()

    ensure_dir(args.out)
    results = []
    for name, stage_key, params in ablation_configs(args.per_type):
        print(f">> running {name} ({stage_key}) {params}")
        results.append(run_one(name, stage_key, params, args.graph_dir, args.runs_root, args.domain))

    with open(os.path.join(args.out, "internal_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    rows = [_flatten_row(r) for r in results]
    md = _markdown(rows)
    with open(os.path.join(args.out, "internal_ablation.md"), "w", encoding="utf-8") as f:
        f.write("# 内部消融结果 (intrinsic, 无 LLM)\n\n" + md + "\n")
    print("\n" + md)
    print(f"\n结果写入 {args.out}/internal_ablation.{{json,md}}")


if __name__ == "__main__":
    main()
