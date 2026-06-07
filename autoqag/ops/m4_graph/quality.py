"""图谱点 / QA 质量过滤 (强化 benchmark 质量)。

依据真实运行观察到的噪声模式过滤：
- 单字符 / 纯符号 / 布尔(Y/N) / 纯 LaTeX 符号的概念与属性 (来自表格符号列、内联公式)
- 通用表头词当属性 ("Value"/"Parameters"/"No."…)
- 元数据混入条件/结论/方法 (Publication/Received/Accepted Date、DOI、IEEE 版权…)
- QA 泄漏内部 node_id、答案退化 (单字符 / 与问题循环 / 空)

所有判定为纯函数，便于单测；m4/m5/m6 共用。
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# 通用表头/占位词，作属性或概念时无信息量
STOPWORD_NAMES = {
    "value", "values", "parameter", "parameters", "param", "params",
    "no", "no.", "num", "number", "item", "items", "index", "id",
    "symbol", "symbols", "quantity", "unit", "units", "name", "type",
    "data", "result", "results", "table", "fig", "figure", "ref", "refs",
    "n/a", "na", "none", "-", "/", "—",
}

# 元数据 / 版权噪声 (出现即判为非科研约束)
_METADATA_RE = re.compile(
    r"\b(publication|received|accepted|revised|submitted|current version|"
    r"copyright|doi|issn|isbn|license|licen[cs]ed|ieee|digital object|"
    r"permission|reprint|all rights reserved|corresponding author|"
    r"manuscript|page \d+|vol\.|volume \d+)\b",
    re.IGNORECASE,
)

# 纯 LaTeX 数学符号 (如 $\mathbf{w}_{1}$, ${\sf C}_{n}$)
_LATEX_ONLY_RE = re.compile(r"^\$.*\$$")

# node_id 形态 (paper_id 带 8 位 hex，或含 :: 坐标)，QA 不应出现
_NODE_ID_RE = re.compile(r"[a-zA-Z0-9_]+_[0-9a-f]{8}\b|::|_c\d+\b|_tab\d+\b|r\d+c\d+")

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _norm(s: str) -> str:
    return (s or "").strip()


def is_symbolic_name(s: str) -> bool:
    """单字符 / 纯标点 / 布尔(Y/N) / 纯 LaTeX 符号 → 视为符号噪声。"""
    s = _norm(s)
    if not s:
        return True
    if s.upper() in {"Y", "N", "YES", "NO", "/", "-", "—", "N/A"}:
        return True
    if not re.search(r"[A-Za-z0-9一-鿿]", s):
        return True
    if _LATEX_ONLY_RE.match(s):
        # 纯 LaTeX：剥离命令(\mathbf 等)与排版符号后，看变量核心长度
        core = re.sub(r"\\[A-Za-z]+", "", s)            # 去 \mathbf \sf \mathrm
        core = re.sub(r"[${}_^\\\s]", "", core)          # 去 $ { } _ ^ \ 空格
        if len(core) <= 3:                               # w1 / Cn / S2 等变量符号
            return True
    if len(s) <= 2 and not _NUM_RE.fullmatch(s):
        # 单/双字符且非数字 (p, n, S., //)
        return True
    return False


def is_metadata(s: str) -> bool:
    return bool(_METADATA_RE.search(_norm(s)))


def is_stopword_name(s: str) -> bool:
    return _norm(s).lower().rstrip(".") in STOPWORD_NAMES


def _looks_like_measurement(s: str) -> bool:
    """形如 '0.8 mm' / '28 mm' / '$80^{\\circ}$' / '110%' 的纯测量值 (不应作 Concept)。

    判定：去 LaTeX 排版后，整体 = 数字 + 至多 5 字符的单位尾。
    这样能滤掉测量值，又不误杀以数字开头的真概念 (如 '5G antenna' 尾部是长单词)。
    """
    core = re.sub(r"[$\\{}^_\s]", "", _norm(s))  # 去 $ \ { } ^ _ 与空格
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?[A-Za-z%°µΩ/·]{0,5}", core))


def is_valid_point(node_type: str, name: str, content: str) -> bool:
    """点是否有信息量，可进入图谱。node_type 为 NodeType.value 字符串。"""
    name = _norm(name) or _norm(content)
    content = _norm(content)
    if not name and not content:
        return False

    # 结构节点 (Paper/Title/Section/Chunk/Figure/Table/Equation/Caption) 不过滤
    structural = {
        "PaperNode", "TitleNode", "SectionNode", "ChunkNode",
        "FigureNode", "TableNode", "EquationNode", "CaptionNode",
    }
    if node_type in structural:
        return True

    # 元数据混入：任何语义点出现版权/卷期/日期等 → 丢弃 (含 AttributeNode 的 "Volume 25")
    if node_type in (
        "ConditionNode", "ClaimNode", "MethodNode", "ConceptNode", "AttributeNode"
    ) and is_metadata(name):
        return False

    # Value/Unit 走专门校验
    if node_type == "ValueNode":
        return bool(_NUM_RE.search(content) or _NUM_RE.search(name))
    if node_type == "UnitNode":
        return len(content) <= 16 and bool(re.search(r"[A-Za-z%°Ω]", content))

    # Concept/Attribute：拒绝符号噪声与通用词
    if node_type in ("ConceptNode", "AttributeNode"):
        if is_symbolic_name(name):
            return False
        if is_stopword_name(name):
            return False
    # Concept 不应是纯测量值 (LLM 常把 "0.8 mm" 误标为概念)
    if node_type == "ConceptNode" and _looks_like_measurement(name):
        return False

    # Condition：保留含范围/数值/温度/频率等的实质条件
    if node_type == "ConditionNode":
        if is_symbolic_name(name):
            return False

    return True


def is_meaningful_attribute(name: str) -> bool:
    """属性是否为真实指标 (供 m5 数值题门控：通用列头如 Value 不出数值题)。"""
    return not (is_symbolic_name(name) or is_stopword_name(name))


# --------------------------- QA 级过滤 ---------------------------
def looks_like_node_id(s: str) -> bool:
    return bool(_NODE_ID_RE.search(s or ""))


def is_valid_qa(
    question: str,
    answer: str,
    question_type: str = "",
    source_names: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """返回 (是否保留, 原因)。过滤退化/泄漏/无信息 QA。"""
    q = _norm(question)
    a = _norm(answer)
    if len(q) < 6:
        return False, "question_too_short"
    if not a:
        return False, "empty_answer"

    # 泄漏内部 node_id / 坐标
    if looks_like_node_id(q) or looks_like_node_id(a):
        return False, "leaks_node_id"

    # 拒答样本单独保留 (由 refusal 通道处理)，这里不算退化
    refusal = a in ("文中无法确定", "无法确定", "Cannot be determined from the text")
    if refusal:
        return True, "refusal"

    # 答案退化：单字符 / 纯符号
    if is_symbolic_name(a):
        return False, "degenerate_answer"

    # 循环：答案就是问题里出现的源实体名 (如问"参数b的符号"答"b")
    if source_names:
        for nm in source_names:
            nm = _norm(nm)
            if nm and a.lower() == nm.lower() and len(a) <= 6:
                return False, "circular_answer"

    # 数值题答案必须含数字
    if question_type == "numerical" and not _NUM_RE.search(a):
        return False, "numerical_without_number"

    return True, "ok"
