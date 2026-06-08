"""核心数据模型 (改编自 GraphGen graphgen/bases/datatypes.py)。

定义流水线全程流转的数据结构，对应论文 §5 各模块的 JSON 规格：
DIR / EvidenceBlock / PointNode / Edge / QuestionPlan / QAItem / Violation。

所有结构都提供 to_dict / from_dict，便于以 jsonl 形式落盘（论文 §5.9 输出规格）。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 枚举：节点类型、题型、难度、错误类型、验证层
# ---------------------------------------------------------------------------
class NodeType(str, Enum):
    """Schema-Evidence Graph 节点类型 (论文 §4 / 图谱构建.pdf §二)。"""

    PAPER = "PaperNode"  # 文献根节点 (图谱构建.pdf §五 结构层级图根)
    TITLE = "TitleNode"  # 论文/章节标题
    SECTION = "SectionNode"  # 章节层级
    CHUNK = "ChunkNode"  # 文本块
    CONCEPT = "ConceptNode"  # 材料、模型、算法、设备、实验对象
    ATTRIBUTE = "AttributeNode"  # 性能、指标、物理量、评价维度
    VALUE = "ValueNode"  # 数值、范围、比例
    UNIT = "UnitNode"  # 单位
    CONDITION = "ConditionNode"  # 实验条件、边界条件、适用范围
    METHOD = "MethodNode"  # 实验/仿真/制备方法
    EQUATION = "EquationNode"  # 公式
    FIGURE = "FigureNode"  # 图
    TABLE = "TableNode"  # 表
    CAPTION = "CaptionNode"  # 图注、表注
    CLAIM = "ClaimNode"  # 发现、趋势、结论
    EVIDENCE = "EvidenceNode"  # 证据 span


class Modality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"
    CAPTION = "caption"
    REFERENCE = "reference"


class QuestionType(str, Enum):
    """首版 8 类题型 (论文 §5.4)。"""

    ATOMIC = "atomic"
    NUMERICAL = "numerical"
    CONDITION = "condition"
    COMPARATIVE = "comparative"
    TABLE = "table"  # table / figure-grounded
    FORMULA = "formula"
    MULTI_HOP = "multi_hop"
    SUMMARY = "summary"


class VirtualEdgeType(str, Enum):
    """问题级虚拟逻辑边类型 (语义层子图构建.pdf §3.3.2 八类虚拟边)。

    虚拟边只服务于问题规划 (语义覆盖层 Ωq)，不是物理证据边、不是永久边：
    它只在当前问题中有效、连接的仍是 G0 中的原始节点、最终答案仍需回到物理证据验证。
    """

    # 同义/共指：连接同一事物的不同表达 (PCE / power conversion efficiency / 图注 efficiency)
    ALIAS = "virtual_alias"
    COREFERENCE = "virtual_coreference"
    # 可比较：共享同一指标、单位与可对齐条件的两个对象 (比较题/排序题)
    COMPARABLE = "virtual_comparable"
    COMPARE_ON = "virtual_compare_on"
    # 条件转移：同一对象在不同温度/时间/压力/浓度/仿真条件下的结果变化 (条件边界题/趋势题)
    CONDITION_SHIFT = "virtual_condition_shift"
    LIMITED_BY = "virtual_limited_by"
    # 方法—结果 / 机制解释 (机制解释题)
    METHOD_EFFECT = "virtual_method_effect"
    EXPLAIN_EFFECT = "virtual_explain_effect"
    SEEK_MECHANISM = "virtual_seek_mechanism"
    MECHANISM = "virtual_mechanism"
    EXPLAINS = "virtual_explains"
    # 跨文献对齐/比较 (跨文献综合题)
    CROSS_PAPER_ALIGN = "virtual_cross_paper_align"
    CROSS_PAPER_COMPARE = "virtual_cross_paper_compare"
    # 冲突与对比 (批判性/创新性问题)
    CONTRAST = "virtual_contrast"
    CONFLICT = "virtual_conflict"
    # 操作链 (流程推理/操作差异)
    OPERATION_FLOW = "virtual_operation_flow"
    # 跨模态需求 (正文结论需图表证据)
    NEED_VISUAL_EVIDENCE = "virtual_need_visual_evidence"
    # 搜索可比对象 (比较题缺对象时的规划意图)
    SEARCH_COMPARABLE = "virtual_search_comparable"


class Difficulty(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


class ErrorType(str, Enum):
    """负样本/扰动错误类型 (论文 §5.6)。"""

    WRONG_NUMBER = "wrong_number"
    UNIT_MISMATCH = "unit_mismatch"
    BOUNDARY_VIOLATION = "boundary_violation"
    ENTITY_SWAP = "entity_swap"
    UNSUPPORTED_ANSWER = "unsupported_answer"
    MISSING_HOP = "missing_hop"
    TABLE_MISREAD = "table_misread"
    FORMULA_MISUSE = "formula_misuse"
    OVER_GENERALIZATION = "over_generalization"
    EVIDENCE_DRIFT = "evidence_drift"


class VerifyLayer(str, Enum):
    """四层一致性验证 (论文创新四)。"""

    CONSTRAINT = "constraint"
    GRAPH = "graph"
    EVIDENCE = "evidence"
    SEMANTIC = "semantic"


# ---------------------------------------------------------------------------
# 物理地址：图谱构建.pdf 强调的 <文献id, 目录级, chunk_id> 为图谱构建基础
# ---------------------------------------------------------------------------
@dataclass
class Address:
    """带物理地址的点/边的最小定位单元。"""

    paper_id: str = ""
    section_path: str = ""  # 目录级，如 "Results/Stability"
    chunk_id: str = ""
    sentence_id: Optional[str] = None
    paragraph_id: Optional[str] = None
    page: Optional[int] = None
    span: Optional[str] = None  # 文本 span，如 "120:160"
    bbox: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Address":
        return Address(**{k: v for k, v in d.items() if k in Address.__annotations__})


# ---------------------------------------------------------------------------
# 文档元数据 (论文 §5.1)
# ---------------------------------------------------------------------------
@dataclass
class DocumentMeta:
    paper_id: str
    title: str = ""
    authors: str = ""
    year: str = ""
    venue: str = ""
    domain: str = ""  # aerospace / materials
    subdomain: str = ""
    source_path: str = ""
    license_status: str = "unknown"
    pdf_quality_level: str = ""  # high / medium / low
    has_tables: bool = False
    has_figures: bool = False
    has_equations: bool = False
    parsing_status: str = "pending"
    graph_status: str = "pending"
    qa_status: str = "pending"
    annotation_status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DocumentMeta":
        return DocumentMeta(
            **{k: v for k, v in d.items() if k in DocumentMeta.__annotations__}
        )


# ---------------------------------------------------------------------------
# 证据块 (论文 §5.2 DIR + 创新二归一化)
# ---------------------------------------------------------------------------
@dataclass
class EvidenceBlock:
    """归一化后的可建图证据单元 (text/table/formula/caption/figure/reference)。"""

    block_id: str
    modality: str  # Modality 值
    content: str  # 文本 / table HTML / latex
    address: Address = field(default_factory=Address)
    confidence: float = 1.0
    # 引用关系：正文中显式提到的 Figure/Table/Equation 编号
    figure_refs: List[str] = field(default_factory=list)
    table_refs: List[str] = field(default_factory=list)
    equation_refs: List[str] = field(default_factory=list)
    caption: str = ""  # 图/表配套图注
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["address"] = self.address.to_dict()
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EvidenceBlock":
        d = dict(d)
        d["address"] = Address.from_dict(d.get("address", {}))
        return EvidenceBlock(
            **{k: v for k, v in d.items() if k in EvidenceBlock.__annotations__}
        )


# ---------------------------------------------------------------------------
# 图谱：点 + 边 (论文 §5.3 + 图谱构建.pdf)
# ---------------------------------------------------------------------------
@dataclass
class PointNode:
    """带物理地址、带标签、带层级属性的点 (图谱构建.pdf §一)。"""

    node_id: str
    node_type: str  # NodeType 值
    content: str
    normalized_content: str = ""
    address: Address = field(default_factory=Address)
    modality: str = "text"
    confidence: float = 1.0
    domain_schema_tag: str = ""
    canonical_id: str = ""  # 跨文献同类点聚合用

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["address"] = self.address.to_dict()
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PointNode":
        d = dict(d)
        d["address"] = Address.from_dict(d.get("address", {}))
        return PointNode(
            **{k: v for k, v in d.items() if k in PointNode.__annotations__}
        )


@dataclass
class Edge:
    """边：语义类型由端点标签决定，建立依据为物理共现或文章结构 (图谱构建.pdf §三)。"""

    source: str
    target: str
    edge_type: str  # has_attribute / has_value / has_unit / under_condition / supports / derived_from / contains / references ...
    build_reason: str = "physical_cooccurrence"  # physical_cooccurrence / document_structure / cross_paper
    cooccur_scope: str = ""  # same_sentence / same_paragraph / same_chunk / same_table_row / ...
    paper_id: str = ""
    section_path: str = ""
    chunk_id: str = ""
    evidence_span: str = ""
    weight: float = 1.0
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Edge":
        return Edge(**{k: v for k, v in d.items() if k in Edge.__annotations__})


# ---------------------------------------------------------------------------
# 问题级语义覆盖层 (语义规划层方法论.pdf + 语义层子图构建.pdf)
# ---------------------------------------------------------------------------
@dataclass
class VirtualEdge:
    """语义覆盖层 Ωq 中的一条临时虚拟逻辑边 (语义层子图构建.pdf §3.3.1)。

    source/target 仍为 G0 中的原始 node_id；虚拟边只用于规划问题语义路径，
    最终答案须由 backing_evidence_paths (物理证据路径) 支撑，否则该边被拒绝。
    """

    source: str
    target: str
    virtual_type: str  # VirtualEdgeType 值
    question_role: str = ""  # 该边在题型 role schema 中承担的逻辑角色
    score: float = 0.0  # 问题级打分 Score_q(e)
    reason: str = ""
    backing_evidence_paths: List[List[str]] = field(default_factory=list)
    required_physical_nodes: List[str] = field(default_factory=list)
    status: str = "candidate"  # candidate / accepted / rejected

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "VirtualEdge":
        return VirtualEdge(
            **{k: v for k, v in d.items() if k in VirtualEdge.__annotations__}
        )


@dataclass
class QuestionGoal:
    """问题逻辑规划目标 (语义层子图构建.pdf §3.1.1 QuestionGoal)。

    驱动初始节点初始化：先确定题型需要哪些逻辑角色，再从 G0 为这些角色找候选节点。
    """

    question_type: str = QuestionType.ATOMIC.value
    difficulty_level: str = Difficulty.L1.value
    domain: str = ""
    seed_topic: str = ""
    theme: str = ""
    expected_answer_form: str = ""
    expected_reasoning_pattern: str = ""
    required_reasoning_roles: List[str] = field(default_factory=list)
    required_constraints: List[str] = field(default_factory=list)
    required_evidence_granularity: str = ""
    forbidden_generalization: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "QuestionGoal":
        return QuestionGoal(
            **{k: v for k, v in d.items() if k in QuestionGoal.__annotations__}
        )


# ---------------------------------------------------------------------------
# Question Plan (论文创新三 §4 + 语义层子图构建.pdf §3.5)
# ---------------------------------------------------------------------------
@dataclass
class QuestionPlan:
    qid: str
    domain: str = ""
    question_type: str = QuestionType.ATOMIC.value
    difficulty: str = Difficulty.L1.value
    target_subgraph: List[str] = field(default_factory=list)  # node_id 列表
    required_nodes: List[str] = field(default_factory=list)
    required_edges: List[Tuple[str, str]] = field(default_factory=list)
    evidence_spans: List[Dict[str, Any]] = field(default_factory=list)
    constraints: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "number": [],
            "unit": [],
            "condition": [],
            "formula": [],
            "table": [],
        }
    )
    expected_answer_form: str = ""
    forbidden_generalization: List[str] = field(default_factory=list)
    generation_instruction: str = ""
    paper_id_list: List[str] = field(default_factory=list)

    # --- 问题级语义覆盖层字段 (语义规划层方法论.pdf §3.7 / 语义层子图构建.pdf §3.5) ---
    seed_nodes: List[str] = field(default_factory=list)
    theme: str = ""
    # 语义覆盖层 Ωq：当前问题专属的虚拟逻辑边 (规划用，非证据)
    semantic_overlay_edges: List[Dict[str, Any]] = field(default_factory=list)
    # 角色完整性：role -> node_id (题型 role schema 的槽位填充情况)
    role_assignment: Dict[str, str] = field(default_factory=dict)
    required_roles: List[str] = field(default_factory=list)
    # 必需的物理证据路径 (每条为 node_id 列表)，最终答案须落回这些路径
    required_evidence_paths: List[List[str]] = field(default_factory=list)
    # 禁止 shortcut：高难度题须显式要求跨证据整合
    forbidden_shortcuts: List[str] = field(default_factory=list)
    # 综合子图评分 (Subgraph Utility Score) 及其分项，便于诊断与反馈
    utility_score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    # 该 plan 由哪条规划路径产出 (用于长期记忆 overlay pattern)
    overlay_pattern: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "QuestionPlan":
        return QuestionPlan(
            **{k: v for k, v in d.items() if k in QuestionPlan.__annotations__}
        )


# ---------------------------------------------------------------------------
# QA (论文 §5.5)
# ---------------------------------------------------------------------------
@dataclass
class QAItem:
    qid: str
    question: str
    answer: str
    question_type: str = ""
    difficulty: str = ""
    evidence_spans: List[Dict[str, Any]] = field(default_factory=list)
    evidence_path: List[str] = field(default_factory=list)
    source_nodes: List[str] = field(default_factory=list)
    source_edges: List[Tuple[str, str]] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    domain: str = ""
    paper_id_list: List[str] = field(default_factory=list)
    validator_result: Dict[str, Any] = field(default_factory=dict)
    # 标记：是否为 corrupted 负样本及其错误类型
    is_corrupted: bool = False
    error_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "QAItem":
        return QAItem(**{k: v for k, v in d.items() if k in QAItem.__annotations__})


# ---------------------------------------------------------------------------
# Violation Report (论文创新五)
# ---------------------------------------------------------------------------
@dataclass
class Violation:
    qid: str
    layer: str  # VerifyLayer 值
    field: str  # number/unit/condition/entity/evidence/path/answer
    expected: str = ""
    actual: str = ""
    source_node: str = ""
    source_edge: str = ""
    source_address: Dict[str, Any] = field(default_factory=dict)
    severity: str = "major"  # minor / major / critical
    repair_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Violation":
        return Violation(
            **{k: v for k, v in d.items() if k in Violation.__annotations__}
        )


# ---------------------------------------------------------------------------
# 基础 LLM 数据结构 (沿用 GraphGen)
# ---------------------------------------------------------------------------
@dataclass
class Token:
    text: str
    prob: float
    top_candidates: List = field(default_factory=list)
    ppl: Optional[float] = None

    @property
    def logprob(self) -> float:
        return math.log(self.prob)


@dataclass
class Community:
    """采样得到的证据子图。"""

    id: Any
    nodes: List[str] = field(default_factory=list)
    edges: List[tuple] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
