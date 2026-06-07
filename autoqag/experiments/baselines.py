"""Baseline 对比方法 (论文 §6.2 / Day10)，当前为薄壳占位。

各 baseline 复用本项目已有组件，差异只在"用不用图谱/验证/修复"：
  - Full Ours          : 完整流水线 (recipes/mvp.yaml)
  - Ours w/o Graph     : 跳过 graph/sample，直接 chunk → QA (TODO: chunk-based generate)
  - Ours w/o Validation: pipeline --skip verify repair
  - Ours w/o Repair    : pipeline --skip repair
  - GraphGen-style QAG : 直接调用 GraphGen 生成 (需 GraphGen 环境)
  - Chunk/RAG/Vanilla  : 仅基于 chunk/检索/全文 prompt 生成 (TODO)

落地方式：大部分 baseline 可用 pipeline 的 --skip / --only 组合实现；
chunk/RAG/vanilla 需补一个 chunk-based 生成 stage。此处先登记接口与 TODO。
"""

from __future__ import annotations

from typing import Dict


BASELINES: Dict[str, Dict[str, str]] = {
    "full_ours": {
        "how": "python -m autoqag.pipeline --recipe recipes/mvp.yaml",
        "status": "ready",
    },
    "ours_wo_validation": {
        "how": "python -m autoqag.pipeline --recipe recipes/mvp.yaml --skip verify repair",
        "status": "ready",
    },
    "ours_wo_repair": {
        "how": "python -m autoqag.pipeline --recipe recipes/mvp.yaml --skip repair",
        "status": "ready",
    },
    "ours_wo_graph": {
        "how": "需补 chunk-based generate stage (绕过 graph/sample)",
        "status": "TODO",
    },
    "chunk_based_qag": {"how": "基于 evidence_blocks 直接 prompt 生成", "status": "TODO"},
    "rag_based_qag": {"how": "检索 top-k chunk 后生成", "status": "TODO"},
    "vanilla_llm_qag": {"how": "全文 prompt 直接生成", "status": "TODO"},
    "graphgen_style": {"how": "调用 GraphGen 生成 (外部环境)", "status": "TODO"},
}


def list_baselines() -> None:
    for name, info in BASELINES.items():
        print(f"[{info['status']:>5}] {name}: {info['how']}")


if __name__ == "__main__":
    list_baselines()
