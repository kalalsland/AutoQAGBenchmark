"""模块二：PDF 解析与文档结构恢复 (论文 §5.2)。

读取 documents.jsonl，对每篇 PDF：MinerU 主解析 (失败/低质回退 PyMuPDF) → DIR，
计算 parsing_quality_score，写 dir/<paper_id>.json，并更新 documents.jsonl 的状态字段。
低质量 (overall=low 且 fallback_on_low) 的论文标记 parsing_status=low_quality 供人工复核。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from autoqag.common.io import ensure_dir, read_jsonl_list, write_json, write_jsonl
from autoqag.ops.base import BaseStage, PipelineContext
from autoqag.ops.m2_parse.dir_builder import build_dir
from autoqag.ops.m2_parse.mineru_parser import MinerUParser
from autoqag.ops.m2_parse.pymupdf_fallback import parse_pdf_pymupdf
from autoqag.registry import STAGES


@STAGES.register_module("parse")
class ParseStage(BaseStage):
    declared_inputs = ["documents.jsonl"]
    declared_outputs = ["dir/", "documents.jsonl"]

    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        docs = read_jsonl_list(ctx.path("documents.jsonl"))
        if not docs:
            self.log("documents.jsonl 为空，先运行 ingest")
            return {"parsed": 0}

        backend = self.params.get("backend", "pipeline")
        method = self.params.get("method", "auto")
        device = self.params.get("device", "cpu")
        lang = self.params.get("lang")
        fallback = self.params.get("fallback", "pymupdf")
        fallback_on_low = self.params.get("fallback_on_low", True)

        dir_out = ensure_dir(ctx.path("dir"))
        mineru_cache = ensure_dir(ctx.path("mineru"))
        mineru_ok = MinerUParser.is_available()
        if not mineru_ok:
            self.log("MinerU 不可用，全部走 %s 回退解析", fallback)

        parsed = low = failed = 0
        for doc in docs:
            paper_id = doc["paper_id"]
            pdf_path = doc["source_path"]
            content_list = None
            tool = "mineru"

            if mineru_ok:
                try:
                    content_list = MinerUParser.parse_pdf(
                        pdf_path,
                        output_dir=mineru_cache,
                        method=method,
                        backend=backend,
                        device=device,
                        lang=lang,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    self.log("MinerU 解析 %s 失败: %s", paper_id, exc)
                    content_list = None

            if not content_list and fallback == "pymupdf":
                try:
                    content_list = parse_pdf_pymupdf(pdf_path)
                    tool = "pymupdf"
                except Exception as exc:  # pylint: disable=broad-except
                    self.log("回退解析 %s 失败: %s", paper_id, exc)
                    content_list = None

            if not content_list:
                doc["parsing_status"] = "failed"
                failed += 1
                continue

            dir_obj = build_dir(paper_id, doc, content_list, tool)
            write_json(os.path.join(dir_out, f"{paper_id}.json"), dir_obj)

            q = dir_obj["parse_quality"]
            doc["pdf_quality_level"] = q["overall_quality"]
            doc["has_tables"] = len(dir_obj["tables"]) > 0
            doc["has_figures"] = len(dir_obj["figures"]) > 0
            doc["has_equations"] = len(dir_obj["equations"]) > 0
            if q["overall_quality"] == "low" and fallback_on_low:
                doc["parsing_status"] = "low_quality"
                low += 1
            else:
                doc["parsing_status"] = "parsed"
                parsed += 1

        write_jsonl(ctx.path("documents.jsonl"), docs)
        self.log("解析完成: ok=%d low=%d failed=%d", parsed, low, failed)
        return {"parsed": parsed, "low_quality": low, "failed": failed}
