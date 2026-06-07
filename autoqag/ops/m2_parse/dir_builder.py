"""Document Intermediate Representation (DIR) 构建 + 解析质量门控 (论文 §5.2)。

把 MinerU/PyMuPDF 的 content_list (按阅读顺序的扁平 block 列表) 转换为论文 §5.2 的
统一 DIR：sections → chunks → (sentence_list / figure_refs / table_refs / equation_refs)，
并单独汇总 figures / tables / equations / references。

同时计算 parsing_quality_score (阅读顺序/表格/公式/图注/OCR) → overall high/medium/low。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# 正文中显式引用 Figure/Table/Equation 的正则 (图谱构建.pdf §十 跨模态引用)
_FIG_RE = re.compile(r"\b(?:Fig\.?|Figure)\s*\.?\s*(\d+)", re.IGNORECASE)
_TAB_RE = re.compile(r"\b(?:Tab\.?|Table)\s*\.?\s*(\d+)", re.IGNORECASE)
_EQ_RE = re.compile(r"\b(?:Eq\.?|Equation)\s*\.?\s*\(?(\d+)\)?", re.IGNORECASE)


def _split_sentences(text: str) -> List[str]:
    """轻量分句 (中英文)。"""
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _block_text(block: Dict[str, Any]) -> str:
    return block.get("text") or block.get("content") or ""


def build_dir(
    paper_id: str,
    metadata: Dict[str, Any],
    content_list: List[Dict[str, Any]],
    parse_tool: str,
) -> Dict[str, Any]:
    """content_list → DIR。"""
    sections: List[Dict[str, Any]] = []
    figures: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []
    equations: List[Dict[str, Any]] = []

    # 当前 section 路径栈：按 text_level 维护层级
    section_stack: List[Tuple[int, str]] = []  # (level, title)
    cur_section: Dict[str, Any] = None
    chunk_counter = 0
    fig_n = tab_n = eq_n = 0

    def _new_section(title: str, level: int) -> Dict[str, Any]:
        path = "/".join(t for _, t in section_stack) or title
        sec = {
            "section_id": f"{paper_id}_sec{len(sections)}",
            "section_path": path,
            "title": title,
            "level": level,
            "chunks": [],
        }
        sections.append(sec)
        return sec

    # 默认根 section，避免开头正文无归属
    cur_section = _new_section("ROOT", 0)

    for block in content_list:
        btype = block.get("type", "text")
        text_level = block.get("text_level", 0) or 0
        page_idx = block.get("page_idx")
        bbox = block.get("bbox")

        if btype == "title" or text_level >= 1:
            title = _block_text(block).strip()
            # 维护层级栈
            while section_stack and section_stack[-1][0] >= text_level:
                section_stack.pop()
            section_stack.append((text_level or 1, title))
            cur_section = _new_section(title, text_level or 1)
            continue

        if btype in ("text", "list"):
            text = _block_text(block)
            if btype == "list" and block.get("list_items"):
                text = "\n".join(block["list_items"])
            if not text.strip():
                continue
            chunk_counter += 1
            chunk = {
                "chunk_id": f"{paper_id}_c{chunk_counter}",
                "text": text,
                "page_range": [page_idx] if page_idx is not None else [],
                "bbox": bbox or [],
                "sentence_list": _split_sentences(text),
                "figure_refs": _FIG_RE.findall(text),
                "table_refs": _TAB_RE.findall(text),
                "equation_refs": _EQ_RE.findall(text),
            }
            cur_section["chunks"].append(chunk)

        elif btype in ("image", "figure", "chart"):
            fig_n += 1
            figures.append(
                {
                    "figure_id": f"{paper_id}_fig{fig_n}",
                    "label": str(fig_n),
                    "caption": " ".join(block.get("image_caption", []) or []),
                    "footnote": " ".join(block.get("image_footnote", []) or []),
                    "img_path": block.get("img_path", ""),
                    "page": page_idx,
                    "bbox": bbox or [],
                    "section_path": cur_section["section_path"],
                }
            )

        elif btype == "table":
            tab_n += 1
            tables.append(
                {
                    "table_id": f"{paper_id}_tab{tab_n}",
                    "label": str(tab_n),
                    "caption": " ".join(block.get("table_caption", []) or []),
                    "footnote": " ".join(block.get("table_footnote", []) or []),
                    "html": block.get("table_body", ""),
                    "img_path": block.get("img_path", ""),
                    "page": page_idx,
                    "bbox": bbox or [],
                    "section_path": cur_section["section_path"],
                }
            )

        elif btype == "equation":
            eq_n += 1
            equations.append(
                {
                    "equation_id": f"{paper_id}_eq{eq_n}",
                    "label": str(eq_n),
                    "latex": block.get("text", ""),
                    "text_format": block.get("text_format", "latex"),
                    "img_path": block.get("img_path", ""),
                    "page": page_idx,
                    "bbox": bbox or [],
                    "section_path": cur_section["section_path"],
                }
            )

    # 移除空的 ROOT section
    sections = [s for s in sections if s["chunks"] or s["title"] != "ROOT"]

    quality = compute_parse_quality(content_list, sections, figures, tables, equations)

    return {
        "paper_id": paper_id,
        "metadata": metadata,
        "parse_tool": parse_tool,
        "parse_quality": quality,
        "sections": sections,
        "figures": figures,
        "tables": tables,
        "equations": equations,
        "references": [],
    }


def compute_parse_quality(
    content_list: List[Dict[str, Any]],
    sections: List[Dict[str, Any]],
    figures: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    equations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """计算解析质量分 (论文 §5.2 解析质量门控)。

    采用可解释的启发式分数：有 bbox/阅读顺序、识别出章节、表格有 HTML、公式有 latex、
    图注非空 → 各维度得分越高。无 GPU/OCR 置信度时给中性值。
    """
    n_block = max(1, len(content_list))
    n_text = sum(1 for b in content_list if b.get("type") in ("text", "list"))

    # 阅读顺序：有 page_idx 的 block 比例
    with_page = sum(1 for b in content_list if b.get("page_idx") is not None)
    reading_order = round(with_page / n_block, 3)

    # 章节识别：识别出的 section 数 (>1 即较好)
    n_section = len([s for s in sections if s["title"] != "ROOT"])
    section_score = round(min(1.0, n_section / 5.0), 3)

    # 表格：有 HTML body 的比例
    table_score = (
        round(sum(1 for t in tables if t["html"]) / len(tables), 3) if tables else 1.0
    )
    # 公式：有 latex 的比例
    formula_score = (
        round(sum(1 for e in equations if e["latex"]) / len(equations), 3)
        if equations
        else 1.0
    )
    # 图注：有 caption 的比例
    caption_total = len(figures) + len(tables)
    caption_with = sum(1 for f in figures if f["caption"]) + sum(
        1 for t in tables if t["caption"]
    )
    caption_score = round(caption_with / caption_total, 3) if caption_total else 1.0

    ocr_confidence = 0.9  # 占位：MinerU 未直接给 OCR 置信度，给中性偏高值

    overall_val = (
        reading_order * 0.25
        + section_score * 0.2
        + table_score * 0.2
        + formula_score * 0.15
        + caption_score * 0.2
    )
    if overall_val >= 0.75:
        overall = "high"
    elif overall_val >= 0.5:
        overall = "medium"
    else:
        overall = "low"

    return {
        "reading_order_score": reading_order,
        "section_score": section_score,
        "table_quality_score": table_score,
        "formula_quality_score": formula_score,
        "caption_quality_score": caption_score,
        "ocr_confidence": ocr_confidence,
        "overall_score": round(overall_val, 3),
        "overall_quality": overall,
        "n_text_blocks": n_text,
    }
