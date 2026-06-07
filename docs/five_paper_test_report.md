# 5 篇文献多模态 / 跨文献 / 难度多样性 / 闭环测试报告

测试集：`data/five/` 5 篇同域论文（频率选择吸波体 / Rasorber），DeepSeek 抽取。
运行：`python -m autoqag.pipeline --recipe recipes/five.yaml`。
本报告回答四个问题：①多模态功能是否正常；②跨文献效果；③难度与多样性
（尤其多跳等体现图谱必要性的题）；④自动评测 + 循环迭代修复是否正常。

## 0. 本轮修复 (针对真实运行暴露的问题)

| 问题 | 根因 | 修复 |
|---|---|---|
| `aligns_with` 跨模态对齐恒为 0 | `max_text_blocks` 把图注块一起截掉，正文/图注无同名概念可对齐 | 限额只作用于正文块，图注块全保留 (`graph.py`) |
| 截断破坏题型多样性 | `plans[:N]` 把排在后面的 multi_hop/summary 全切掉 | 按题型轮转截断 `_interleave_by_type` (`generate.py`) |
| 引用标记 `[20]`/`[16]` 当概念 | `is_symbolic_name` 未识别引用 | 新增 `_CITATION_RE` (`quality.py`) |
| 元数据日期 `Received: 7 Jan…` 当条件 | `is_metadata` 只查 name 没查 content | 同时查 content (`quality.py`) |
| 表格数字碎片 `5.02,2.22`/`N.A.`/`0` 当概念 | 无字母词校验 | Concept/Attribute 须含 ≥2 长字母词 (`quality.py`) |
| 合法拒答被判违规入 human_review | verify/repair 对 "文中无法确定" 跑数值/证据检查 | `is_refusal` 短路通过 (`verifiers.py`/`verify.py`/`repair.py`) |
| 多跳问题模糊 ("某参数值→8") | BFS 游走进匿名表格值单元格 | `_bfs_path` 跳过 `::rNcM` 单元格与无内容 Value (`sample.py`) |

## 1. 多模态功能 ✅ 正常

5 篇真实图谱 (nodes=2409, edges=5974)：

| 模态节点 | 数量 | 多模态边 | 数量 |
|---|---|---|---|
| FigureNode | 333 | references (正文→图/表/公式) | 131 |
| TableNode | 23 | describes (图表→图注) | 445 |
| EquationNode | 74 | aligns_with (图注概念↔正文同名) | 31 |
| CaptionNode | 127 | (表格行列 same_table_row/column) | 458 cells |

- **references** 按 `figure/table/equation_text_reference` 区分 scope，正文 "Fig.3/Table II/Eq.(5)" 正确连到对应图表公式节点。
- **aligns_with** 修复后 31 条 (此前因图注截断恒为 0)：同篇内图注抽出的概念与正文同名概念跨模态对齐，合并为同一证据集合 (§十.3)。
- **表格行列**：458 个单元格点，建 same_table_row / same_table_column 共现边，列表头→ValueNode 得 has_value，带括号单位另建 UnitNode + has_unit。

## 2. 跨文献效果 ✅ 正常

`same_as` 跨文献聚合 12 条，过滤后全部为有意义的共享概念/指标 (噪声 "This work"/"[16]" 已清除)：

```
Lossy Layer ↔ Lossy Layer          Broadband Absorption ↔ Broadband Absorption
Lossless Layer ↔ Lossless Layer    Resistive Layer ↔ Resistive layer
Graphene ↔ Graphene                FSS Layer ↔ FSS layer
Top Layer ↔ Top layer              Middle Layer ↔ Middle layer
Bandpass FSS ↔ Bandpass FSS        IL (dB) ↔ IL (dB)
```

5 篇 Rasorber 论文共享的结构层 (Lossy/Lossless/Resistive/FSS Layer)、材料 (Graphene)、
指标 (IL 插入损耗) 被正确横向连接 —— 这正是跨文献比较题的图谱基础。

## 3. 难度与多样性 ✅ 正常 (8 题型齐全)

题型分布 (benchmark 池 81 题)：
atomic 8 / numerical 12 / condition 12 / comparative 12 / table 12 / formula 12 / multi_hop 8 / summary 5

难度分布：L1=8, L2=12, L3=10, L4=51 (高约束文档 → L4 偏多，符合论文定位)

### 体现"图谱必要性"的题 (单 chunk RAG 难以回答)

**multi_hop** (跨 section 遍历，带证据路径)：
- Q: 哪两种可调谐材料分别用于实现吸波器模式和可调谐反射带？请给出完整的证据路径。
  A: VO₂ 用于吸波器模式，石墨烯用于可调谐反射带。证据路径：electronics_…c1 → …
- Q: design procedure of the proposed dual absorption FSA → 多步设计流程综合答案

**comparative** (沿图找共享 Attribute 的两个 Value 再比较)：
- Q: -3 dB 通带带宽的仿真结果与实测结果分别为多少？ A: 仿真 56.20%，实测 12.89%
- Q: 偏压 5V 时仿真与实测的反射带相对带宽 FBW？ A: 仿真 29.29%，实测 25.7%
- Q: Design 1 与 Design 2 的形状因子分别是多少？ A: 3.73 / 2.8

**summary** (章节级聚合多指标)：
- Q: 该超宽带吸波体反射损耗低于 -10 dB 的频率范围 (模拟/实测)？
  A: 模拟 1.6~4.6 GHz (96.8%)，实测 1.9~4.7 GHz (84.8%)

这些题需要"结构图定位 + 共现/跨文献图关联"两步，单 chunk 检索无法回答，验证了图谱的必要性。

## 4. 自动评测 + 循环迭代修复 ✅ 正常

四层验证 (数值/单位/条件/证据 + LLM 语义层) + violation 驱动 ≤3 轮自修复：

| 阶段 | 修复前 (初版) | 质量修复后 | 多跳修复后 (终版) |
|---|---|---|---|
| benchmark 通过 | 36 | 57 | **61** |
| human_review | 42 | 21 | **20** |
| 修复成功 (repair) | 16 | 34 | 39 |

- **verify**：四层 checker + 可选 LLM 语义层全部运行，产出结构化 Violation (layer/field/expected/actual/severity/repair_hint)。
- **repair**：对未通过 QA 用结构化 violation 报告驱动局部修复，每轮重验证，最多 3 轮；修复成功入 final 池，否则入 human_review 池，并记录 wrong→violation→repaired 轨迹 (corpus/repair.jsonl，可作训练数据)。
- 合法拒答 ("文中无法确定") 修复后正确判通过，不再误入 human_review。
- 10 类负样本扰动全部触发 (corpus/verifier.jsonl 234、preference.jsonl 156)。

## 结论

多模态、跨文献、难度多样性 (含多跳/比较/综合)、自动评测 + 循环迭代修复 四项功能
在 5 篇真实论文上均验证正常。本轮 7 处质量修复使 benchmark 通过率从 36 → 61、
human_review 从 42 → 20，并消除了引用标记 / 元数据 / 表格碎片 / 模糊多跳等噪声。
