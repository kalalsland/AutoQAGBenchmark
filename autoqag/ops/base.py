"""Stage 基类与流水线上下文。

每个流水线模块 = 一个 BaseStage 子类，注册到 STAGES。
Stage 之间通过工作目录下的命名 artifact (jsonl) 通信，因此可单独运行某个 stage
(局部定位修改)。declared_inputs / declared_outputs 用于 pipeline 校验依赖。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from autoqag.common.io import ensure_dir
from autoqag.common.llm import LLMClient
from autoqag.common.logging import logger


@dataclass
class PipelineContext:
    """流水线全局上下文，贯穿所有 stage。"""

    work_dir: str  # 工作/输出目录，所有 artifact 落盘于此
    global_params: Dict[str, Any] = field(default_factory=dict)
    _llm: Optional[LLMClient] = None

    def path(self, name: str) -> str:
        """artifact 的绝对路径。"""
        return os.path.join(self.work_dir, name)

    def artifact_exists(self, name: str) -> bool:
        return os.path.exists(self.path(name))

    @property
    def llm(self) -> LLMClient:
        """惰性构建共享 LLM 客户端 (从 global_params.llm 或环境变量)。"""
        if self._llm is None:
            llm_cfg = dict(self.global_params.get("llm", {}))
            self._llm = LLMClient.from_env(**llm_cfg)
        return self._llm


class BaseStage:
    """所有流水线模块的基类。"""

    # 子类通过 @STAGES.register_module("name") 注册；_name 由 registry 注入
    _name: str = "base"

    # 声明输入/输出 artifact 文件名，供 pipeline 依赖校验与 --only 单跑提示
    declared_inputs: List[str] = []
    declared_outputs: List[str] = []

    def __init__(self, **params: Any):
        self.params = params

    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        """执行 stage，返回统计信息 dict。子类必须实现。"""
        raise NotImplementedError

    # 辅助
    def log(self, msg: str, *args: Any) -> None:
        try:
            logger.info(f"[{self._name}] " + msg, *args)
        except TypeError:
            logger.info(f"[{self._name}] {msg}")

    def ensure_work_dir(self, ctx: PipelineContext) -> str:
        return ensure_dir(ctx.work_dir)
