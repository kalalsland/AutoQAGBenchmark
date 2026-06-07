"""表格 HTML → 行列网格解析 (图谱构建.pdf §四 same_table_row / same_table_column 的基础)。

MinerU 把表格识别为 HTML (table_body 字段，含 <thead>/<tr>/<td>，可能带 colspan/rowspan)，
但不做"行列语义建图"。本模块把 HTML 解析为规整网格 (展开合并单元格)，供 m4 抽取
表格单元格点并建立同行/同列共现边。纯标准库 html.parser，无额外依赖。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import List, Optional

_NUM_RE = re.compile(r"^[-+±]?\s*\d+(?:\.\d+)?(?:\s*[-–~±]\s*\d+(?:\.\d+)?)?\s*$")
# 表头括号内的单位，如 "PCE (%)" / "Voc (V)"
_UNIT_IN_HEADER_RE = re.compile(r"[（(]\s*([^)）]{1,12}?)\s*[)）]\s*$")


@dataclass
class Cell:
    row: int
    col: int
    text: str
    is_header: bool = False  # 来自 <th> 或 thead/首行


@dataclass
class TableGrid:
    cells: List[Cell] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0

    def header_row(self) -> List[Cell]:
        """列表头：优先 is_header 的最上面一行，否则首行。"""
        header_cells = [c for c in self.cells if c.is_header]
        if header_cells:
            top = min(c.row for c in header_cells)
            return sorted(
                [c for c in header_cells if c.row == top], key=lambda c: c.col
            )
        return sorted([c for c in self.cells if c.row == 0], key=lambda c: c.col)

    def column_header(self, col: int) -> Optional[Cell]:
        for c in self.header_row():
            if c.col == col:
                return c
        return None

    def row_label(self, row: int) -> Optional[Cell]:
        """行标签：该行最左 (col=0) 的单元格 (常为研究对象/样本名)。"""
        candidates = [c for c in self.cells if c.row == row and c.col == 0]
        return candidates[0] if candidates else None

    def body_cells(self) -> List[Cell]:
        """正文数据单元格 (非表头行、非首列标签)。"""
        header_rows = {c.row for c in self.cells if c.is_header} or {0}
        return [
            c
            for c in self.cells
            if c.row not in header_rows and c.col != 0 and c.text.strip()
        ]


class _TableHTMLParser(HTMLParser):
    """把 HTML 表格展开为网格，正确处理 colspan / rowspan。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: List[Cell] = []
        self._row = -1
        self._col_cursor = 0
        self._in_cell = False
        self._cur_is_header = False
        self._cur_text: List[str] = []
        self._cur_colspan = 1
        self._cur_rowspan = 1
        self._in_thead = False
        # 被 rowspan 占用的格子：{(row, col): remaining_rows}
        self._occupied: dict = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "thead":
            self._in_thead = True
        elif tag == "tr":
            self._row += 1
            self._col_cursor = 0
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cur_is_header = tag == "th" or self._in_thead
            self._cur_text = []
            self._cur_colspan = _to_int(a.get("colspan"), 1)
            self._cur_rowspan = _to_int(a.get("rowspan"), 1)

    def handle_endtag(self, tag):
        if tag == "thead":
            self._in_thead = False
        elif tag in ("td", "th") and self._in_cell:
            # 跳过被上方 rowspan 占用的列
            while self._occupied.get((self._row, self._col_cursor), 0) > 0:
                self._occupied[(self._row, self._col_cursor)] -= 1
                self._col_cursor += 1
            col = self._col_cursor
            text = re.sub(r"\s+", " ", "".join(self._cur_text)).strip()
            self.cells.append(
                Cell(row=self._row, col=col, text=text, is_header=self._cur_is_header)
            )
            # 处理跨列/跨行占位
            for dc in range(self._cur_colspan):
                for dr in range(1, self._cur_rowspan):
                    self._occupied[(self._row + dr, col + dc)] = (
                        self._occupied.get((self._row + dr, col + dc), 0) + 1
                    )
            self._col_cursor += self._cur_colspan
            self._in_cell = False

    def handle_data(self, data):
        if self._in_cell:
            self._cur_text.append(data)


def _to_int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def parse_table_html(html: str) -> TableGrid:
    """解析表格 HTML → TableGrid。无法解析返回空网格。"""
    grid = TableGrid()
    if not html or "<" not in html:
        return grid
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
    except Exception:  # pragma: no cover - 容错：损坏 HTML 不致命
        return grid
    cells = [c for c in parser.cells if c.text.strip()]
    grid.cells = cells
    grid.n_rows = (max((c.row for c in cells), default=-1)) + 1
    grid.n_cols = (max((c.col for c in cells), default=-1)) + 1
    return grid


def is_numeric_cell(text: str) -> bool:
    return bool(_NUM_RE.match(text.strip()))


def unit_from_header(text: str) -> Optional[str]:
    """从表头 'PCE (%)' 抽取单位 '%'。"""
    m = _UNIT_IN_HEADER_RE.search(text.strip())
    return m.group(1).strip() if m else None
