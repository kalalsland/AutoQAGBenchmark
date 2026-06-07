"""流水线编排执行器。

按 recipe.yaml 声明的 stage 顺序依次执行；支持：
- `--only m4_graph`     只跑单个 stage (artifact 已存在即可，便于局部定位修改)
- `--from m4_graph`     从某 stage 开始
- `--to m8_verify`      到某 stage 结束
- `--skip m7_corrupt`   跳过某些 stage

用法：
    python -m autoqag.pipeline --recipe recipes/mvp.yaml
    python -m autoqag.pipeline --recipe recipes/mvp.yaml --only m4_graph
"""

from __future__ import annotations

import argparse
import time
from typing import List, Optional

from autoqag.common.io import ensure_dir
from autoqag.common.logging import logger
from autoqag.config import RecipeConfig, dump_recipe_snapshot, load_recipe
from autoqag.ops import load_all_stages
from autoqag.ops.base import PipelineContext
from autoqag.registry import STAGES


def run_pipeline(
    cfg: RecipeConfig,
    only: Optional[str] = None,
    from_stage: Optional[str] = None,
    to_stage: Optional[str] = None,
    skip: Optional[List[str]] = None,
) -> None:
    load_all_stages()  # 触发所有 stage 注册
    ensure_dir(cfg.work_dir)

    ctx = PipelineContext(work_dir=cfg.work_dir, global_params=cfg.global_params)

    specs = [s for s in cfg.stages if s.enabled]
    names = [s.stage for s in specs]

    # 选择要跑的 stage 子集
    selected = specs
    if only:
        selected = [s for s in specs if s.stage == only]
        if not selected:
            raise ValueError(f"--only {only} 不在 pipeline 中: {names}")
    else:
        start = 0
        end = len(specs)
        if from_stage:
            start = names.index(from_stage)
        if to_stage:
            end = names.index(to_stage) + 1
        selected = specs[start:end]
        if skip:
            selected = [s for s in selected if s.stage not in skip]

    logger.info("流水线工作目录: %s", cfg.work_dir)
    logger.info("将执行 stage: %s", [s.stage for s in selected])

    # 快照 recipe，保证复现
    dump_recipe_snapshot(cfg, ctx.path("recipe_snapshot.yaml"))

    for spec in selected:
        cls = STAGES.get(spec.stage)
        if cls is None:
            raise ValueError(
                f"未注册的 stage: {spec.stage}。已注册: {STAGES.list()}"
            )
        stage = cls(**spec.params)
        t0 = time.time()
        logger.info(">> 开始 stage: %s", spec.stage)
        stats = stage.run(ctx) or {}
        dt = time.time() - t0
        logger.info("<< 完成 stage: %s (%.1fs) %s", spec.stage, dt, stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoQAGBenchmark 流水线")
    parser.add_argument("--recipe", required=True, help="recipe.yaml 路径")
    parser.add_argument("--only", default=None, help="只运行某个 stage")
    parser.add_argument("--from", dest="from_stage", default=None, help="从某 stage 开始")
    parser.add_argument("--to", dest="to_stage", default=None, help="到某 stage 结束")
    parser.add_argument("--skip", nargs="*", default=None, help="跳过的 stage")
    args = parser.parse_args()

    cfg = load_recipe(args.recipe)
    run_pipeline(
        cfg,
        only=args.only,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        skip=args.skip,
    )


if __name__ == "__main__":
    main()
