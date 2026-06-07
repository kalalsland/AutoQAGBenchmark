# 流水线模块记录文档

每个模块的：对应论文章节、输入/输出 artifact、实现要点、复用来源、可修改点。

---

## m1_ingest — PDF 输入与数据管理 (论文 §5.1)

- **代码**: `autoqag/ops/m1_ingest/ingest.py` (`stage: ingest`)
- **输入**: `params.input_dir` 下的 PDF 文件
- **输出**: `documents.jsonl` (DocumentMeta，含 parsing/graph/qa/annotation 状态字段)
- **要点**: 文件名 + md5 生成稳定 `paper_id`；domain 从 recipe `global_params.domain` 继承。
- **可修改点**: metadata 补充 (title/authors/year 可后续接 GROBID)。

## m2_parse — PDF 解析与文档结构恢复 (论文 §5.2 / 创新二)

- **代码**: `autoqag/ops/m2_parse/{parse,mineru_parser,pymupdf_fallback,dir_builder}.py` (`stage: parse`)
- **输入**: `documents.jsonl`
- **输出**: `dir/<paper_id>.json` (统一 DIR)，更新 `documents.jsonl` 状态
- **要点**:
  - `MinerUParser` 改编自 GraphGen，**保留 page_idx/bbox/text_level** (物理地址必需)；
  - MinerU 输出缓存在 `work_dir/mineru/`，重复运行免重解析；
  - `dir_builder.build_dir`: content_list → sections→chunks→sentences + figures/tables/equations；
    用正则抽取正文对 Fig./Table/Eq. 的显式引用 (跨模态边的来源)；
  - `compute_parse_quality`: 阅读顺序/章节/表格/公式/图注 5 维启发式分 → high/medium/low 门控；
  - MinerU 不可用或失败 → PyMuPDF 回退 (`pymupdf_fallback.py`)，质量自然降级。
- **可修改点**: 质量分权重；分句规则；接 middle.json 获取更细粒度 span。

## m3_normalize — 证据归一化 (论文创新二)

- **代码**: `autoqag/ops/m3_normalize/{normalize,units}.py` (`stage: normalize`)
- **输入**: `dir/*.json`
- **输出**: `evidence_blocks.jsonl` (6 类 EvidenceBlock: text/table/formula/caption/figure/reference)
- **要点**: 每块绑定 Address(paper/section/chunk/page/bbox)+modality+confidence；
  `units.py` 基于 pint 做单位归一/换算/量纲兼容判断 (被 m8 复用)；文本块预抽 (数值,单位) 对。
- **可修改点**: `_UNIT_ALIAS` 领域单位别名表 (航天/材料常用单位可继续补)。

## m4_graph — Schema-Evidence Graph (论文创新一/二 + 图谱构建.pdf)

- **代码**: `autoqag/ops/m4_graph/{graph,extractor,edge_rules}.py` (`stage: graph`)
- **输入**: `evidence_blocks.jsonl`
- **输出**: `nodes.jsonl` / `edges.jsonl` / `graph.graphml`
- **要点** (双图耦合):
  1. **结构图 (纵向, §五)**: **Paper 根节点** → Title(标题点) / Section(逐级目录) → Chunk → point 的 contains 边；figure/table/equation 挂到 section；caption 通过 describes 挂到父图表。不调 LLM。
  2. **共现图 (横向, §四)**: 文本块**与图注块**都做 LLM 点抽取 (prompt 改编自 GraphGen kg_extraction，14 类科研点标签)；块内点对建边——LLM 给的 relation 优先，再按 `edge_rules.resolve_edge` (端点标签→语义类型**并规范化方向**，论文 §5.3 表) 规则补全；`cooccur_scope` 标注 same_sentence/same_chunk/same_caption；点尽量落 sentence_id (§一 细粒度地址)。
  2b. **表格行列共现 (§四, `table_parser.py`)**: 解析 MinerU 输出的表格 HTML (标准库 `html.parser`，正确展开 colspan/rowspan)——列表头→AttributeNode、行标签→ConceptNode、数据格→ValueNode、表头括号单位→UnitNode；建 `has_value`(同列 Attribute→Value)、`has_unit`(Value→Unit)、`same_table_row`(行标签→数据格) 等边。**无需 LLM**，且直接喂给数值题采样器。
  3. **跨模态引用边 (§十.1)**: chunk 的 figure_refs/table_refs/equation_refs → 对应节点 (references)。
  4. **跨模态对象对齐 (§十.3)**: 同文献内同 canonical 的 Concept/Attribute 跨模态 (图注↔正文) 连 `aligns_with`，把同对象的跨模态证据合并。
  5. **跨文献聚合 (§八)**: 同 canonical_id 的 Concept/Attribute 跨 paper 连 same_as。
- **可修改点**: `edge_rules._PAIR_RULES` 增删边语义；`extract_points: false` 可只建结构图(含表格行列)零成本调试；`parse_table_cells: false` 关闭表格解析；`max_text_blocks` 控制 token 成本。
- **对照《图谱构建.pdf》覆盖度**: §一~§十 均已实现。`same_paragraph` scope 待补 paragraph_id 后启用 (当前用 same_sentence/same_chunk 近似)。

## m5_sample — 子图采样与 Question Plan (论文创新三 §5.4)

- **代码**: `autoqag/ops/m5_sample/{sample,planner}.py` (`stage: sample`)
- **输入**: `nodes.jsonl` / `edges.jsonl`
- **输出**: `question_plans.jsonl` (论文创新三的 QuestionPlan JSON 格式)
- **要点**: 8 题型各一个 finder (纯图遍历，无 LLM)：
  atomic(单点) / numerical(Concept-Attr-Value-Unit) / condition(Condition→*) /
  comparative(多 Concept 共享同 canonical Attribute) / table(Table/Figure+邻居) /
  formula(Equation→Attr/Value+引用 chunk) / multi_hop(BFS 3-4 跳) / summary(Section 聚合)；
  难度 L1-L4 由 `planner.py` 的结构变量 (evidence_count/path_length/modality_count/section_distance/constraint_count/cross_paper) 计算，非语言长度。
- **可修改点**: `per_type` 数量；难度阈值 (`compute_difficulty`)；新增 finder。

## m6_generate — QA 与高级训练语料 (论文 §5.5 / 创新六)

- **代码**: `autoqag/ops/m6_generate/{generate,json_utils}.py` (`stage: generate`)
- **输入**: `question_plans.jsonl`
- **输出**: `qa.jsonl` + `corpus/{instruction,graph_trace,rag_grounding,refusal}.jsonl`
- **要点**: plan+打包证据 → `QA_GENERATION_PROMPT` (严格 JSON 输出) → QAItem；
  生成原则按论文 §5.5 写进 prompt (绑 span / 数值带单位 / 保留条件 / 多跳给路径 / 不足拒答)；
  4 类训练语料直接从 QA 派生 (verifier/preference 在 m7，repair 在 m9)。
- **可修改点**: prompt 模板 `templates/qa_generation.py`；`max_plans` 控制成本。

## m7_corrupt — 负样本与错误扰动 (论文 §5.6)

- **代码**: `autoqag/ops/m7_corrupt/{corrupt,corruptors}.py` (`stage: corrupt`)
- **输入**: `qa.jsonl` (+ 图节点池)
- **输出**: `corrupted_qa.jsonl` + `corpus/verifier.jsonl` + `corpus/preference.jsonl`
- **要点**: 10 类错误构造器 (纯字符串/图节点替换，零 LLM 成本)；
  每条 QA 随机选 `per_qa` 个构造器；good/bad 成对产 verifier 训练样本与偏好对。
- **可修改点**: 各 corruptor 的扰动强度；`per_qa`/`seed`。

## m8_verify — 四层约束验证 (论文创新四 §5.7)

- **代码**: `autoqag/ops/m8_verify/{verify,verifiers}.py` (`stage: verify`)
- **输入**: `qa.jsonl` (param `target` 可指向其他文件，如 corrupted_qa.jsonl 做实验四)
- **输出**: `violations.jsonl`；qa.jsonl 写回 `validator_result`
- **要点**: MVP 四 checker——数值(证据内匹配±2%) / 单位(pint 量纲兼容) /
  条件(保留+反泛化措辞) / 证据(可追溯+token 重叠)；可选 `semantic_check` 走 LLM 第四层；
  违例输出论文创新五的结构化 Violation (layer/field/expected/actual/severity/repair_hint)。
- **可修改点**: 容差 `rel_tol`；checker 开关在 recipe；新增 checker 登记 `CHECKERS`。

## m9_repair — Violation 驱动自修复 (论文创新五 §5.8)

- **代码**: `autoqag/ops/m9_repair/repair.py` (`stage: repair`)
- **输入**: 验证后的 `qa.jsonl`
- **输出**: 修复后 `qa.jsonl` (pool 标记 final/human_review) + `corpus/repair.jsonl` (修复轨迹)
- **要点**: 结构化 violation → `REPAIR_PROMPT` 局部修复 → 立即用同一组 checker 重验证，
  ≤`max_rounds` 轮；轨迹记录 wrong→violation→repaired 每一步 (训练数据)。
- **可修改点**: `max_rounds`；修复 prompt。

## m10_output — 数据输出与组织 (论文 §5.9)

- **代码**: `autoqag/ops/m10_output/output.py` (`stage: output`)
- **输入**: 前序全部 artifact
- **输出**: `benchmark/{benchmark,human_review}.jsonl`、`graph/`、`dataset_manifest.json`、`stats.json`
- **要点**: 按 validator_result.passed 分池；汇总语料计数与题型/难度分布统计。

---

## Artifact 流向总览

```
documents.jsonl ─m2→ dir/*.json ─m3→ evidence_blocks.jsonl ─m4→ nodes/edges.jsonl
   ─m5→ question_plans.jsonl ─m6→ qa.jsonl + corpus/{instr,trace,rag,refusal}
   ─m7→ corrupted_qa.jsonl + corpus/{verifier,preference}
   ─m8→ violations.jsonl (qa.jsonl 写回 validator_result)
   ─m9→ qa.jsonl (修复) + corpus/repair.jsonl
   ─m10→ benchmark/ + graph/ + stats.json + dataset_manifest.json
```
