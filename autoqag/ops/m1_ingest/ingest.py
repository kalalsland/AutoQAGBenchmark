"""模块一：科研 PDF 输入与数据管理 (论文 §5.1)。

扫描输入目录下的 PDF，为每篇建立 metadata，输出 documents.jsonl。
metadata 含 parsing_status/graph_status/qa_status 等状态字段，供后续 stage 更新。
"""

from __future__ import annotations

import glob
import hashlib
import os
from typing import Any, Dict, List

from autoqag.common.io import write_jsonl
from autoqag.ops.base import BaseStage, PipelineContext
from autoqag.registry import STAGES
from autoqag.schema import DocumentMeta


@STAGES.register_module("ingest")
class IngestStage(BaseStage):
    declared_outputs = ["documents.jsonl"]

    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        self.ensure_work_dir(ctx)
        input_dir = self.params.get("input_dir", "data/raw")
        domain = self.params.get("domain", ctx.global_params.get("domain", ""))
        subdomain = self.params.get("subdomain", "")
        recursive = self.params.get("recursive", True)

        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdfs = sorted(glob.glob(os.path.join(input_dir, pattern), recursive=recursive))

        docs: List[Dict[str, Any]] = []
        for pdf in pdfs:
            paper_id = _make_paper_id(pdf)
            meta = DocumentMeta(
                paper_id=paper_id,
                title=os.path.splitext(os.path.basename(pdf))[0],
                domain=domain,
                subdomain=subdomain,
                source_path=os.path.abspath(pdf),
            )
            docs.append(meta.to_dict())

        out = ctx.path("documents.jsonl")
        n = write_jsonl(out, docs)
        self.log("发现 %d 篇 PDF (input_dir=%s)", n, input_dir)
        if n == 0:
            self.log("警告：未找到 PDF，请把论文放到 %s", input_dir)
        return {"documents": n}


def _make_paper_id(pdf_path: str) -> str:
    """用文件名 + 内容前缀哈希生成稳定 paper_id。"""
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    h = hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]
    safe = "".join(c if c.isalnum() else "_" for c in stem)[:40]
    return f"{safe}_{h}"
