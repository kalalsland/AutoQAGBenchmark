"""科研点抽取 prompt (改编自 GraphGen kg_extraction.py)。

与 GraphGen 通用实体抽取的关键差异：
- 实体类型换成论文定义的 14 类科研点 (Concept/Attribute/Value/Unit/Condition/...);
- 要求每个点标注 modality 与所在 evidence_span，便于绑定物理地址;
- 输出元组格式沿用 GraphGen 的 {tuple_delimiter}/{record_delimiter}/{completion_delimiter}
  分隔符，解析逻辑可直接复用。
"""

FORMAT = {
    "tuple_delimiter": "<|>",
    "record_delimiter": "##",
    "completion_delimiter": "<|COMPLETE|>",
    # 论文 §4 节点类型 (抽取阶段关注的细粒度点，结构点由代码生成)
    "point_types": (
        "Concept, Attribute, Value, Unit, Condition, Method, Claim, "
        "Equation, Figure, Table, Caption"
    ),
}


POINT_EXTRACTION_EN = """You are a scientific-document information-extraction expert. \
Extract typed "points" with physical evidence from the given block of a research paper.

-Point types-
[{point_types}]
- Concept: material/model/algorithm/device/experimental object
- Attribute: performance/metric/physical quantity/evaluation dimension
- Value: a number, range, or ratio
- Unit: the unit of a value (%, eV, MPa, K, ...)
- Condition: experimental/boundary condition, applicable range (temperature, time, ...)
- Method: experimental/simulation/fabrication method
- Claim: a finding, trend, or conclusion
- Equation / Figure / Table / Caption: when present in the block

-Steps-
1. Identify every point. For each output:
   - point_name: a short canonical name (capitalized if English)
   - point_type: one of [{point_types}]
   - point_content: the exact span from the text supporting this point
   Format: ("point"{tuple_delimiter}<point_name>{tuple_delimiter}<point_type>{tuple_delimiter}<point_content>)

2. Identify directly-related point pairs that physically co-occur in this block.
   For each pair output:
   - source, target: point_name from step 1
   - relation: a short label among [has_attribute, has_value, has_unit, under_condition, supports, derived_from, describes, compares]
   Format: ("relation"{tuple_delimiter}<source>{tuple_delimiter}<target>{tuple_delimiter}<relation>)

3. When finished output {completion_delimiter}
Use **{record_delimiter}** between records.

################
-Block (modality={modality}, section={section_path})-
{input_text}
################
Output:
"""


POINT_EXTRACTION_ZH = """你是科研文档信息抽取专家。请从给定的论文片段中抽取带证据的"点"。

-点类型-
[{point_types}]
- Concept: 材料/模型/算法/设备/实验对象
- Attribute: 性能/指标/物理量/评价维度
- Value: 数值/范围/比例
- Unit: 单位 (%、eV、MPa、K ...)
- Condition: 实验/边界条件、适用范围 (温度、时间 ...)
- Method: 实验/仿真/制备方法
- Claim: 发现、趋势、结论
- Equation / Figure / Table / Caption: 片段中出现时

-步骤-
1. 识别每个点，逐个输出：
   - point_name: 简短规范名
   - point_type: [{point_types}] 之一
   - point_content: 支撑该点的原文 span
   格式：("point"{tuple_delimiter}<point_name>{tuple_delimiter}<point_type>{tuple_delimiter}<point_content>)

2. 识别本片段中物理共现且直接相关的点对，逐个输出：
   - source, target: 步骤1的 point_name
   - relation: [has_attribute, has_value, has_unit, under_condition, supports, derived_from, describes, compares] 之一
   格式：("relation"{tuple_delimiter}<source>{tuple_delimiter}<target>{tuple_delimiter}<relation>)

3. 完成后输出 {completion_delimiter}
记录之间用 **{record_delimiter}** 分隔。

################
-片段 (modality={modality}, section={section_path})-
{input_text}
################
输出：
"""


POINT_EXTRACTION_PROMPT = {
    "en": POINT_EXTRACTION_EN,
    "zh": POINT_EXTRACTION_ZH,
    "FORMAT": FORMAT,
}
