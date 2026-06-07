"""单位归一化与数值/单位抽取 (论文 normalize_units / §5.7 单位验证基础)。

基于 pint 提供单位规范化与换算；pint 缺失时回退到内置常用单位别名表。
这些工具被 m3 (证据归一化) 与 m8 (数值/单位验证) 共用。
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

try:
    import pint  # type: ignore

    _UREG = pint.UnitRegistry()
    _HAS_PINT = True
except Exception:  # pragma: no cover
    _UREG = None
    _HAS_PINT = False


# 数值 + 单位 的粗匹配 (含范围、科学计数、百分号、常见科研单位)
_NUM_UNIT_RE = re.compile(
    r"(?P<num>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"\s*(?P<unit>%|°C|℃|K|nm|µm|um|mm|cm|m|kg|mg|g|MPa|kPa|Pa|GPa|eV|meV|"
    r"mol/L|mol|mAh/g|mA|A|V|Hz|GHz|MHz|W|J|s|h|min|wt\.?%|at\.?%)?",
)

# pint 无法识别的科研别名 → 标准写法
_UNIT_ALIAS = {
    "℃": "degC",
    "°C": "degC",
    "wt%": "percent",
    "wt.%": "percent",
    "at%": "percent",
    "at.%": "percent",
    "%": "percent",
    "um": "micrometer",
    "µm": "micrometer",
    "mAh/g": "milliampere_hour / gram",
}


def normalize_unit(unit: str) -> str:
    """把单位字符串规范化为标准形式 (如 ℃→degC)；无法识别则原样返回。"""
    if not unit:
        return ""
    u = unit.strip()
    if u in _UNIT_ALIAS:
        u = _UNIT_ALIAS[u]
    if _HAS_PINT:
        try:
            return str(_UREG(u).units)
        except Exception:
            return u
    return u


def extract_value_units(text: str) -> List[Tuple[str, str]]:
    """从文本抽取 (数值, 单位) 对，供证据块与验证使用。"""
    out: List[Tuple[str, str]] = []
    for m in _NUM_UNIT_RE.finditer(text or ""):
        num = m.group("num")
        unit = m.group("unit") or ""
        if num:
            out.append((num, unit.strip()))
    return out


def convert(value: float, from_unit: str, to_unit: str) -> Optional[float]:
    """单位换算 (论文 §5.7 单位是否可换算)。失败返回 None。"""
    if not _HAS_PINT:
        return None
    try:
        fu = _UNIT_ALIAS.get(from_unit, from_unit)
        tu = _UNIT_ALIAS.get(to_unit, to_unit)
        q = _UREG.Quantity(value, fu)
        return float(q.to(tu).magnitude)
    except Exception:
        return None


def units_compatible(u1: str, u2: str) -> bool:
    """两个单位是否同量纲 (可换算)。"""
    if not u1 or not u2:
        return u1 == u2
    if _HAS_PINT:
        try:
            a = _UREG(_UNIT_ALIAS.get(u1, u1))
            b = _UREG(_UNIT_ALIAS.get(u2, u2))
            return a.dimensionality == b.dimensionality
        except Exception:
            return normalize_unit(u1) == normalize_unit(u2)
    return normalize_unit(u1) == normalize_unit(u2)
