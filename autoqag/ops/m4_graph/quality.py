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
    # 对比表里的自指/泛指行标签，作跨文献概念时只会连出噪声
    "this work", "proposed", "proposed work", "ours", "this paper",
    "ref.", "reference", "others",
}

# 引用标记 ([20] / [16,17] / [3]-[5])，常被误抽为概念
_CITATION_RE = re.compile(r"^\[\s*\d+\s*(?:[,\-–]\s*\d+\s*)*\]$")

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
    """单字符 / 纯标点 / 布尔(Y/N) / 纯 LaTeX 符号 / 引用标记 → 视为符号噪声。"""
    s = _norm(s)
    if not s:
        return True
    if s.upper() in {"Y", "N", "YES", "NO", "/", "-", "—", "N/A"}:
        return True
    if _CITATION_RE.match(s):                            # [20] / [16,17] 引用标记
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

    # 元数据混入：任何语义点的名称或内容出现版权/卷期/收稿日期等 → 丢弃
    # (LLM 常把 "Received: 7 January 2026, Revised…" 抽成 ConditionNode，需查 content)
    if node_type in (
        "ConditionNode", "ClaimNode", "MethodNode", "ConceptNode", "AttributeNode"
    ) and (is_metadata(name) or is_metadata(content)):
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
        # 必须含一个长度>=2 的字母词 (英文或中文)，否则是表格数字碎片：
        # "5.02,2.22" / "(34.75%) 3.898" / "N.A." / "0.64~" / 纯数字列头 "0"
        if not re.search(r"[A-Za-z]{2,}|[一-鿿]{2,}", name):
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


# --------------------------- 边极性 / 置信度 (图谱构建.pdf 刀1) ---------------------------
# 纯共现建边的死穴：原文 "B does NOT improve PCE" / "Unlike A, B ..." 会被建成正向边。
# 这里在建边前做句级极性检测，并按共现范围给边置信度，使被否定/对比/假设的边
# 既被标记 (polarity)、又被降权 (confidence)，不进入正向答案证据层。

# 否定线索 (英文)：not / n't / no / without / fail to / cannot / never / neither-nor / lack of / absence of
_NEG_EN_RE = re.compile(
    r"\b(?:not|no|never|without|cannot|none|neither|nor)\b"
    r"|n['’]t\b"
    r"|\bfail(?:s|ed|ing)?\s+to\b"
    r"|\black(?:s|ed|ing)?\s+of\b"
    r"|\babsence\s+of\b",
    re.IGNORECASE,
)
# 对比 / 让步线索 (英文)
_CONTRAST_EN_RE = re.compile(
    r"\b(?:unlike|whereas|however|although|though|but|conversely)\b"
    r"|\bin\s+contrast\b|\bcontrary\s+to\b|\brather\s+than\b|\binstead\s+of\b|\bas\s+opposed\s+to\b",
    re.IGNORECASE,
)
# 假设 / 情态线索 (英文)：非实测、推测性
_HYPO_EN_RE = re.compile(
    r"\b(?:if|would|could|might|may|suppose|assuming|hypothetically|potentially)\b"
    r"|\bexpected\s+to\b|\bis\s+expected\b|\bwere\s+to\b",
    re.IGNORECASE,
)
# 中文线索 (保守集，避免 "无线/不锈" 等词内误命中)
_NEG_ZH_RE = re.compile(r"没有|并非|并不|无法|未能|不能|不会|不是|不再|未(?=能|被|得到)|缺乏")
_CONTRAST_ZH_RE = re.compile(r"然而|尽管|相反|而非|不同于|与之相反|却")
_HYPO_ZH_RE = re.compile(r"假设|如果|若是|倘若|可能会|预计|有望")

# 共现范围 → 基础置信度 (越紧密越可信)
SCOPE_BASE_CONFIDENCE = {
    "same_sentence": 1.0,
    "same_table_row": 0.9,
    "same_table_column": 0.9,
    "same_caption": 0.85,
    "same_paragraph": 0.7,
    "same_chunk": 0.55,
}
# 同块 all-pairs 规则补全 (LLM 未断言该关系) 的基础置信度
RULE_COMPLETION_CONFIDENCE = 0.4
# 极性惩罚系数：非 positive 的边被原文否定/对比/假设，乘以低系数
POLARITY_PENALTY = {
    "positive": 1.0,
    "hypothetical": 0.45,
    "contrastive": 0.35,
    "negative": 0.25,
}
# 证据层门控阈值：confidence >= 该值且 polarity == positive 才可作正向答案证据
EVIDENCE_CONFIDENCE_THRESHOLD = 0.5


def _between_window(text: str, span_a: str, span_b: str) -> str:
    """取两个实体提及之间 (含少量左边距) 的子串，使极性检测局部化。

    左边距 30 字符用于捕获前置对比/否定线索 (如 'Unlike A, ...')；
    定位失败则回退到整段文本。
    """
    t = text or ""
    if not t:
        return ""
    ha = (span_a or "").strip()[:20]
    hb = (span_b or "").strip()[:20]
    ia = t.lower().find(ha.lower()) if ha else -1
    ib = t.lower().find(hb.lower()) if hb else -1
    if ia < 0 or ib < 0:
        return t  # 定位不到则扫全句/全块
    lo, hi = sorted((ia, ib))
    # hi 端补上第二个提及的长度
    end = hi + max(len(ha), len(hb))
    # 左侧留至多 30 字符边距以捕获前置对比/否定线索 (如 'Unlike A, ...')，
    # 但不得跨句子边界 (.!?; 或中文句号)，否则会把上一句的否定误纳入窗口。
    left = max(0, lo - 30)
    margin = t[left:lo]
    bpos = max(
        margin.rfind("."), margin.rfind("!"), margin.rfind("?"),
        margin.rfind(";"), margin.rfind("。"), margin.rfind("；"),
    )
    if bpos >= 0:
        left = left + bpos + 1
    return t[left: end + 5]


def detect_polarity(text: str, span_a: str = "", span_b: str = "") -> str:
    """检测两点之间共现关系的极性。

    返回 positive / negative / contrastive / hypothetical。
    优先级：否定 > 对比 > 假设 > 正向 (否定语义对证据最致命)。
    span_a/span_b 给定时只在两提及之间的窗口检测，降低误命中。
    """
    window = _between_window(text, span_a, span_b)
    if not window:
        return "positive"
    if _NEG_EN_RE.search(window) or _NEG_ZH_RE.search(window):
        return "negative"
    if _CONTRAST_EN_RE.search(window) or _CONTRAST_ZH_RE.search(window):
        return "contrastive"
    if _HYPO_EN_RE.search(window) or _HYPO_ZH_RE.search(window):
        return "hypothetical"
    return "positive"


def edge_confidence(
    scope: str,
    polarity: str = "positive",
    *,
    rule_completion: bool = False,
) -> float:
    """边置信度 = 基础(共现范围或规则补全) × 极性惩罚。

    rule_completion=True 表示该边来自同块 all-pairs 补全 (LLM 未断言)，基础置信度更低，
    只服务召回层，默认不进证据层。
    """
    base = (
        RULE_COMPLETION_CONFIDENCE
        if rule_completion
        else SCOPE_BASE_CONFIDENCE.get(scope, 0.5)
    )
    conf = base * POLARITY_PENALTY.get(polarity, 1.0)
    return round(conf, 3)


def is_evidence_eligible(confidence: float, polarity: str = "positive") -> bool:
    """该边能否作为正向答案证据：置信度达标且极性为正。"""
    return polarity == "positive" and confidence >= EVIDENCE_CONFIDENCE_THRESHOLD


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
