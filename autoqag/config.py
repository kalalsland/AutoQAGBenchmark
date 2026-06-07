"""recipe.yaml 载入与校验 (论文 §5.9 recipe 格式)。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StageSpec:
    stage: str  # 注册名，对应 STAGES 中的 key
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class RecipeConfig:
    work_dir: str
    global_params: Dict[str, Any] = field(default_factory=dict)
    stages: List[StageSpec] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


def load_recipe(path: str) -> RecipeConfig:
    """载入 recipe.yaml。

    结构：
        work_dir: outputs/run1
        global_params:
            llm: {model: ..., base_url: ...}
            domain: aerospace
        pipeline:
            - stage: ingest
              params: {input_dir: data/raw}
            - stage: parse
              params: {backend: pipeline}
            ...
    """
    import yaml

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    work_dir = raw.get("work_dir", "outputs/run")
    global_params = raw.get("global_params", {})

    stages: List[StageSpec] = []
    for item in raw.get("pipeline", []):
        if isinstance(item, str):
            stages.append(StageSpec(stage=item))
        elif isinstance(item, dict):
            # 支持两种写法: {stage: name, params: {...}} 或 {name: {...}}
            if "stage" in item:
                stages.append(
                    StageSpec(
                        stage=item["stage"],
                        params=item.get("params", {}),
                        enabled=item.get("enabled", True),
                    )
                )
            else:
                (name, params), = item.items()
                stages.append(StageSpec(stage=name, params=params or {}))
        else:
            raise ValueError(f"无法解析的 pipeline 项: {item!r}")

    return RecipeConfig(
        work_dir=work_dir,
        global_params=global_params,
        stages=stages,
        raw=raw,
    )


def dump_recipe_snapshot(cfg: RecipeConfig, out_path: str) -> None:
    """把 recipe 原文快照写入输出目录，保证可复现性 (论文 §5.9 输出四)。"""
    import yaml

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.raw, f, allow_unicode=True, sort_keys=False)
