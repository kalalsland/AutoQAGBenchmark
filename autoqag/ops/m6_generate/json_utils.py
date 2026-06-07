"""JSON 解析辅助：从 LLM 输出中稳健提取 JSON 对象。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    """从可能含 ```json 代码块或多余文字的 LLM 输出中提取首个 JSON 对象。"""
    if not text:
        return None
    # 去掉 ```json ... ``` 围栏
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        # 取第一个 { 到最后一个 } 之间
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # 容错：尝试去掉尾随逗号
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
