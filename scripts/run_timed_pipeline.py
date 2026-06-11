"""计时驱动：按 recipe 顺序逐 stage 执行并记录每个环节墙钟耗时。

与 autoqag.pipeline.run_pipeline 等价的执行循环，但额外把各 stage 耗时
汇总写入 work_dir/pipeline_timing.json，并在结束时打印对照表。
generate 阶段的分题型计时由该 stage 自身写入 generate_timing.json，本脚本读取合并。

用法：
    python scripts/run_timed_pipeline.py --recipe recipes/timing.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# 允许以 `python scripts/run_timed_pipeline.py` 直接运行：把项目根加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoqag.common.io import ensure_dir
from autoqag.common.logging import logger
from autoqag.config import dump_recipe_snapshot, load_recipe
from autoqag.ops import load_all_stages
from autoqag.ops.base import PipelineContext
from autoqag.registry import STAGES

# 关注的核心环节中文名（用户要求的四项 + 其余如实记录）
STAGE_LABELS = {
    "parse": "PDF抽取",
    "graph": "物理图建立",
    "sample": "子图构建",
    "generate": "问题生成",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--model", default=None, help="覆盖 recipe 的 LLM model")
    ap.add_argument("--work-dir", default=None, help="覆盖 recipe 的 work_dir")
    ap.add_argument("--from-stage", default=None, help="从某 stage 开始")
    ap.add_argument("--to-stage", default=None, help="到某 stage 结束(含)")
    args = ap.parse_args()

    cfg = load_recipe(args.recipe)
    if args.model:
        cfg.global_params.setdefault("llm", {})["model"] = args.model
        cfg.raw.setdefault("global_params", {}).setdefault("llm", {})["model"] = args.model
    if args.work_dir:
        cfg.work_dir = args.work_dir
        cfg.raw["work_dir"] = args.work_dir
    load_all_stages()
    ensure_dir(cfg.work_dir)
    ctx = PipelineContext(work_dir=cfg.work_dir, global_params=cfg.global_params)
    dump_recipe_snapshot(cfg, ctx.path("recipe_snapshot.yaml"))

    specs = [s for s in cfg.stages if s.enabled]
    names = [s.stage for s in specs]
    start = names.index(args.from_stage) if args.from_stage else 0
    end = (names.index(args.to_stage) + 1) if args.to_stage else len(specs)
    specs = specs[start:end]
    logger.info("计时运行 stage: %s", [s.stage for s in specs])

    records = []
    t_all = time.perf_counter()
    for spec in specs:
        cls = STAGES.get(spec.stage)
        if cls is None:
            raise ValueError(f"未注册的 stage: {spec.stage}")
        stage = cls(**spec.params)
        logger.info(">> 开始 stage: %s", spec.stage)
        t0 = time.perf_counter()
        stats = stage.run(ctx) or {}
        dt = time.perf_counter() - t0
        logger.info("<< 完成 stage: %s (%.2fs) %s", spec.stage, dt, stats)
        records.append(
            {
                "stage": spec.stage,
                "label": STAGE_LABELS.get(spec.stage, spec.stage),
                "seconds": round(dt, 3),
                "stats": stats,
            }
        )
    total = time.perf_counter() - t_all

    # 合并分题型生成计时
    gen_timing = None
    gt_path = ctx.path("generate_timing.json")
    if os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            gen_timing = json.load(f)

    report = {
        "recipe": args.recipe,
        "work_dir": cfg.work_dir,
        "total_seconds": round(total, 3),
        "stages": records,
        "generate_per_type": gen_timing,
    }
    out = ctx.path("pipeline_timing.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印对照表
    print("\n" + "=" * 60)
    print("各环节耗时 (pipeline_timing.json)")
    print("=" * 60)
    for r in records:
        print(f"  {r['label']:<12} {r['stage']:<12} {r['seconds']:>10.2f}s")
    print("-" * 60)
    print(f"  {'总计':<12} {'TOTAL':<12} {total:>10.2f}s")
    if gen_timing and gen_timing.get("by_type"):
        print("\n分题型平均单题生成时间 (墙钟, 受并发影响):")
        print(f"  并发={gen_timing.get('max_concurrency')} 模型={gen_timing.get('model')}")
        for t, s in sorted(gen_timing["by_type"].items()):
            print(
                f"  {t:<14} n={s['count']:<3} avg={s['avg_sec']:>7.2f}s "
                f"min={s['min_sec']:>6.2f} max={s['max_sec']:>6.2f}"
            )
    print("=" * 60)
    print(f"已写入: {out}")


if __name__ == "__main__":
    main()
