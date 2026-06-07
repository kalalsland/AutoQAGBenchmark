"""LLM 点抽取与元组解析 (改编自 GraphGen kg_extraction 解析逻辑)。

调用 POINT_EXTRACTION_PROMPT 抽取科研点与点间关系，解析 GraphGen 风格的
("point"<|>name<|>type<|>content) / ("relation"<|>src<|>tgt<|>label) 元组。
解析与 LLM 调用解耦：parse_extraction 可独立单测。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from autoqag.templates.point_extraction import POINT_EXTRACTION_PROMPT

_FMT = POINT_EXTRACTION_PROMPT["FORMAT"]
_TUPLE_DELIM = _FMT["tuple_delimiter"]
_RECORD_DELIM = _FMT["record_delimiter"]


def build_prompt(text: str, modality: str, section_path: str, lang: str = "en") -> str:
    tpl = POINT_EXTRACTION_PROMPT.get(lang, POINT_EXTRACTION_PROMPT["en"])
    return tpl.format(
        point_types=_FMT["point_types"],
        tuple_delimiter=_TUPLE_DELIM,
        record_delimiter=_RECORD_DELIM,
        completion_delimiter=_FMT["completion_delimiter"],
        modality=modality,
        section_path=section_path,
        input_text=text,
    )


def _clean(s: str) -> str:
    return s.strip().strip('"').strip("'").strip()


def parse_extraction(response: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """解析 LLM 输出为 (points, relations)。

    points: [{name, type, content}]
    relations: [{source, target, relation}]
    """
    points: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []
    if not response:
        return points, relations

    # 按记录分隔符或换行切分，逐条匹配 (...) 元组
    raw = response.replace(_FMT["completion_delimiter"], "")
    chunks = re.split(re.escape(_RECORD_DELIM) + r"|\n", raw)
    for ch in chunks:
        ch = ch.strip()
        m = re.search(r"\((.*)\)", ch, re.DOTALL)
        if not m:
            continue
        fields = [_clean(x) for x in m.group(1).split(_TUPLE_DELIM)]
        if not fields:
            continue
        kind = fields[0].lower()
        if kind == "point" and len(fields) >= 4:
            points.append({"name": fields[1], "type": fields[2], "content": fields[3]})
        elif kind == "relation" and len(fields) >= 4:
            relations.append(
                {"source": fields[1], "target": fields[2], "relation": fields[3]}
            )
    return points, relations
