"""PyMuPDF 回退解析器 (论文 §5.2 fallback)。

当 MinerU 不可用或解析质量过低时，用 PyMuPDF 抽取纯文本块，
产出与 MinerU content_list 兼容的最小结构 (type=text/title + page_idx + bbox)。
表格/公式/图注质量会较低，parsing_quality 据此降级。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from autoqag.common.logging import logger


def parse_pdf_pymupdf(pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """返回与 MinerU content_list 字段兼容的 block 列表。"""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyMuPDF 未安装，无法回退解析。请 pip install pymupdf"
        ) from exc

    pdf = Path(pdf_path).expanduser().resolve()
    doc = fitz.open(str(pdf))
    blocks: List[Dict[str, Any]] = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
            text = (text or "").strip()
            if not text:
                continue
            # 简单启发式：短而独占一行的可能是标题
            is_title = len(text) < 80 and "\n" not in text and text.isupper()
            blocks.append(
                {
                    "type": "title" if is_title else "text",
                    "text": text,
                    "text_level": 1 if is_title else 0,
                    "page_idx": page_idx,
                    "bbox": [x0, y0, x1, y1],
                }
            )
    doc.close()
    logger.info("PyMuPDF 回退解析 %s: %d blocks", pdf.name, len(blocks))
    return blocks
