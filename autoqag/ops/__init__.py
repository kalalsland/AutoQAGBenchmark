"""autoqag.ops: 流水线各模块 (stage)。

每个 mN_* 子包定义一个 BaseStage 子类并注册到 STAGES。
load_all_stages() 导入全部子模块以触发注册。
"""

import importlib

from autoqag.ops.base import BaseStage, PipelineContext

# 各模块的导入路径 (stage 注册名 -> 模块)
_STAGE_MODULES = [
    "autoqag.ops.m1_ingest.ingest",
    "autoqag.ops.m2_parse.parse",
    "autoqag.ops.m3_normalize.normalize",
    "autoqag.ops.m4_graph.graph",
    "autoqag.ops.m5_sample.sample",
    "autoqag.ops.m6_generate.generate",
    "autoqag.ops.m7_corrupt.corrupt",
    "autoqag.ops.m8_verify.verify",
    "autoqag.ops.m9_repair.repair",
    "autoqag.ops.m10_output.output",
]


def load_all_stages() -> None:
    for mod in _STAGE_MODULES:
        importlib.import_module(mod)


__all__ = ["BaseStage", "PipelineContext", "load_all_stages"]
