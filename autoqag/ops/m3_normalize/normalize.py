"""模块三：证据归一化 (论文创新二 §5.2 第二步)。

读取 dir/*.json，把每篇 DIR 转换为统一的 EvidenceBlock 列表 (6 类：
text / table / formula / caption / figure / reference)，每块绑定物理地址、modality、
confidence，并对文本中的数值-单位做归一化记录。输出 evidence_blocks.jsonl。
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List

from autoqag.common.io import read_json, write_jsonl
from autoqag.ops.base import BaseStage, PipelineContext
from autoqag.ops.m3_normalize.units import extract_value_units, normalize_unit
from autoqag.registry import STAGES
from autoqag.schema import Address, EvidenceBlock, Modality


@STAGES.register_module("normalize")
class NormalizeStage(BaseStage):
    declared_inputs = ["dir/"]
    declared_outputs = ["evidence_blocks.jsonl"]

    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        dir_files = sorted(glob.glob(ctx.path("dir/*.json")))
        if not dir_files:
            self.log("dir/ 为空，先运行 parse")
            return {"blocks": 0}

        normalize_units = self.params.get("normalize_units", True)
        blocks: List[EvidenceBlock] = []
        for fp in dir_files:
            dir_obj = read_json(fp)
            blocks.extend(self._convert(dir_obj, normalize_units))

        n = write_jsonl(ctx.path("evidence_blocks.jsonl"), (b.to_dict() for b in blocks))
        by_mod: Dict[str, int] = {}
        for b in blocks:
            by_mod[b.modality] = by_mod.get(b.modality, 0) + 1
        self.log("生成 %d 个证据块: %s", n, by_mod)
        return {"blocks": n, **{f"mod_{k}": v for k, v in by_mod.items()}}

    def _convert(
        self, dir_obj: Dict[str, Any], normalize_units: bool
    ) -> List[EvidenceBlock]:
        paper_id = dir_obj["paper_id"]
        out: List[EvidenceBlock] = []

        # text blocks (来自 chunk)
        for sec in dir_obj.get("sections", []):
            for chunk in sec.get("chunks", []):
                page = chunk["page_range"][0] if chunk.get("page_range") else None
                addr = Address(
                    paper_id=paper_id,
                    section_path=sec["section_path"],
                    chunk_id=chunk["chunk_id"],
                    page=page,
                    bbox=chunk.get("bbox") or None,
                )
                extra: Dict[str, Any] = {
                    "sentence_list": chunk.get("sentence_list", []),
                }
                if normalize_units:
                    vus = extract_value_units(chunk["text"])
                    if vus:
                        extra["value_units"] = [
                            {"value": v, "unit": u, "normalized_unit": normalize_unit(u)}
                            for v, u in vus
                        ]
                out.append(
                    EvidenceBlock(
                        block_id=f"{chunk['chunk_id']}_eb",
                        modality=Modality.TEXT.value,
                        content=chunk["text"],
                        address=addr,
                        figure_refs=chunk.get("figure_refs", []),
                        table_refs=chunk.get("table_refs", []),
                        equation_refs=chunk.get("equation_refs", []),
                        extra=extra,
                    )
                )

        # table blocks (+ caption)
        for t in dir_obj.get("tables", []):
            addr = Address(
                paper_id=paper_id,
                section_path=t.get("section_path", ""),
                chunk_id=t["table_id"],
                page=t.get("page"),
                bbox=t.get("bbox") or None,
            )
            out.append(
                EvidenceBlock(
                    block_id=f"{t['table_id']}_eb",
                    modality=Modality.TABLE.value,
                    content=t.get("html", ""),
                    address=addr,
                    caption=t.get("caption", ""),
                    extra={"label": t.get("label"), "footnote": t.get("footnote", "")},
                )
            )
            if t.get("caption"):
                out.append(self._caption_block(t["table_id"], t["caption"], addr))

        # formula blocks
        for e in dir_obj.get("equations", []):
            addr = Address(
                paper_id=paper_id,
                section_path=e.get("section_path", ""),
                chunk_id=e["equation_id"],
                page=e.get("page"),
                bbox=e.get("bbox") or None,
            )
            out.append(
                EvidenceBlock(
                    block_id=f"{e['equation_id']}_eb",
                    modality=Modality.FORMULA.value,
                    content=e.get("latex", ""),
                    address=addr,
                    extra={"label": e.get("label")},
                )
            )

        # figure blocks (+ caption)
        for f in dir_obj.get("figures", []):
            addr = Address(
                paper_id=paper_id,
                section_path=f.get("section_path", ""),
                chunk_id=f["figure_id"],
                page=f.get("page"),
                bbox=f.get("bbox") or None,
            )
            out.append(
                EvidenceBlock(
                    block_id=f"{f['figure_id']}_eb",
                    modality=Modality.FIGURE.value,
                    content=f.get("caption", ""),  # 图本身无文本，用图注作可检索内容
                    address=addr,
                    caption=f.get("caption", ""),
                    extra={
                        "label": f.get("label"),
                        "img_path": f.get("img_path", ""),
                        "footnote": f.get("footnote", ""),
                    },
                )
            )
            if f.get("caption"):
                out.append(self._caption_block(f["figure_id"], f["caption"], addr))

        # reference blocks
        for i, ref in enumerate(dir_obj.get("references", [])):
            out.append(
                EvidenceBlock(
                    block_id=f"{paper_id}_ref{i}",
                    modality=Modality.REFERENCE.value,
                    content=ref if isinstance(ref, str) else str(ref),
                    address=Address(paper_id=paper_id, section_path="References"),
                )
            )

        return out

    @staticmethod
    def _caption_block(parent_id: str, caption: str, addr: Address) -> EvidenceBlock:
        return EvidenceBlock(
            block_id=f"{parent_id}_cap",
            modality=Modality.CAPTION.value,
            content=caption,
            address=addr,
            extra={"parent": parent_id},
        )
