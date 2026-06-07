"""模块九：数据输出与组织 (论文 §5.9)。

把全流程产物组织为论文 §5.9 的四类输出：
  输出一 Benchmark 数据集：benchmark/benchmark.jsonl (通过池) + low_quality / human_review 池
  输出二 高级训练语料：corpus/* (instruction/graph_trace/rag/refusal/verifier/preference/repair)
  输出三 图谱数据：graph/{nodes,edges}.jsonl + documents/chunks/evidence_spans
  输出四 pipeline 配置：recipe_snapshot.yaml (由 pipeline.py 写出)
并产出 dataset_manifest.json + stats.json 便于查阅与复现。
"""

from __future__ import annotations

import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List

from autoqag.common.io import (
    ensure_dir,
    read_jsonl_list,
    write_json,
    write_jsonl,
)
from autoqag.ops.base import BaseStage, PipelineContext
from autoqag.registry import STAGES
from autoqag.schema import QAItem


# 论文 §5.9 输出一：benchmark 字段
_BENCH_FIELDS = [
    "qid",
    "question",
    "answer",
    "evidence_spans",
    "evidence_path",
    "question_type",
    "difficulty",
    "domain",
    "paper_id_list",
    "source_nodes",
    "constraints",
    "validator_result",
]


@STAGES.register_module("output")
class OutputStage(BaseStage):
    declared_inputs = ["qa.jsonl"]
    declared_outputs = ["benchmark/", "dataset_manifest.json", "stats.json"]

    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        bench_dir = ensure_dir(ctx.path("benchmark"))
        graph_dir = ensure_dir(ctx.path("graph"))

        qa_rows = read_jsonl_list(ctx.path("qa.jsonl"))
        qa_items = [QAItem.from_dict(r) for r in qa_rows]

        # 按验证/修复结果分池
        final_pool: List[Dict[str, Any]] = []
        human_pool: List[Dict[str, Any]] = []
        for q in qa_items:
            vr = q.validator_result or {}
            row = {k: getattr(q, k) for k in _BENCH_FIELDS if hasattr(q, k)}
            if vr.get("passed"):
                final_pool.append(row)
            else:
                human_pool.append(row)

        n_final = write_jsonl(os.path.join(bench_dir, "benchmark.jsonl"), final_pool)
        n_human = write_jsonl(os.path.join(bench_dir, "human_review.jsonl"), human_pool)

        # 输出三：图谱数据归集
        for fn in ("nodes.jsonl", "edges.jsonl", "documents.jsonl", "evidence_blocks.jsonl"):
            src = ctx.path(fn)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(graph_dir, fn))

        # 更新文档 qa_status
        self._mark_qa_status(ctx, has_qa=n_final > 0)

        # manifest + stats
        manifest = self._build_manifest(ctx, n_final, n_human)
        write_json(ctx.path("dataset_manifest.json"), manifest)
        stats = self._build_stats(ctx, qa_items, n_final, n_human)
        write_json(ctx.path("stats.json"), stats)

        self.log("输出: benchmark=%d human_review=%d", n_final, n_human)
        self.log("manifest 与 stats 已写入 %s", ctx.work_dir)
        return {"benchmark": n_final, "human_review": n_human}

    def _mark_qa_status(self, ctx: PipelineContext, has_qa: bool) -> None:
        path = ctx.path("documents.jsonl")
        if not os.path.exists(path):
            return
        docs = read_jsonl_list(path)
        for d in docs:
            if d.get("graph_status") == "done":
                d["qa_status"] = "done" if has_qa else "empty"
        write_jsonl(path, docs)

    def _build_manifest(
        self, ctx: PipelineContext, n_final: int, n_human: int
    ) -> Dict[str, Any]:
        corpus_files = {}
        corpus_dir = ctx.path("corpus")
        if os.path.isdir(corpus_dir):
            for fn in sorted(os.listdir(corpus_dir)):
                if fn.endswith(".jsonl"):
                    corpus_files[fn] = len(
                        read_jsonl_list(os.path.join(corpus_dir, fn))
                    )
        return {
            "outputs": {
                "benchmark": {
                    "benchmark.jsonl": n_final,
                    "human_review.jsonl": n_human,
                },
                "advanced_corpus": corpus_files,
                "graph": ["graph/nodes.jsonl", "graph/edges.jsonl", "graph.graphml"],
                "recipe_snapshot": "recipe_snapshot.yaml",
            },
            "benchmark_fields": _BENCH_FIELDS,
        }

    def _build_stats(
        self,
        ctx: PipelineContext,
        qa_items: List[QAItem],
        n_final: int,
        n_human: int,
    ) -> Dict[str, Any]:
        by_type: Dict[str, int] = defaultdict(int)
        by_diff: Dict[str, int] = defaultdict(int)
        for q in qa_items:
            by_type[q.question_type] += 1
            by_diff[q.difficulty] += 1

        graph_stats = {}
        if ctx.artifact_exists("nodes.jsonl"):
            graph_stats["nodes"] = len(read_jsonl_list(ctx.path("nodes.jsonl")))
            graph_stats["edges"] = len(read_jsonl_list(ctx.path("edges.jsonl")))
        docs = (
            read_jsonl_list(ctx.path("documents.jsonl"))
            if ctx.artifact_exists("documents.jsonl")
            else []
        )
        return {
            "documents": len(docs),
            "qa_total": len(qa_items),
            "qa_passed": n_final,
            "qa_human_review": n_human,
            "qa_by_type": dict(by_type),
            "qa_by_difficulty": dict(by_diff),
            "graph": graph_stats,
        }
