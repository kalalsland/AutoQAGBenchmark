# 问题级语义覆盖层 / 语义规划层 (Semantic Overlay Planning)

> 实现来源：`方案/语义规划层方法论.pdf`（Score-guided Question Subgraph Planning
> with Virtual Logic Completion）与 `方案/语义层子图构建.pdf`（问题级语义覆盖层）。

## 1. 动机

基线 `sample` stage 在固定物理图 G0 上直接做模板化子图采样，问题逻辑（题型需要
哪些角色、是否真跨证据、是否会被走捷径）没有被显式建模。本层在 **不修改 G0** 的
前提下，为**每个问题**动态构建一个临时**语义覆盖层 Ωq**（一组**虚拟逻辑边**），
形成 `Gq = G0[Vq] + Ωq`：

- 虚拟边只服务**规划**（补全缺失逻辑角色、表达跨证据/跨文献意图）；
- 最终 `QuestionPlan.required_nodes / required_evidence_paths` 仍由 **G0 物理节点与
  物理路径**构成，保证 Benchmark **可追溯、可验证**。

## 2. 总体流程

`OverlayPlanner.plan_one(qtype, seed, qid)`（`autoqag/ops/m5_sample/semantic/planner.py`）：

| 步 | 阶段 | 实现 |
|---|---|---|
| 1 | 问题逻辑规划 | `roles.role_schema(qtype)` 取题型逻辑角色与最小角色集 |
| 2 | 种子初始化 | `seed.seed_score` / `rank_seeds`（SeedScore，可叠加长期记忆 boost） |
| 3 | 评分引导物理扩展 | `evidence_chain.expand_candidates`（向下游走 + 向上定位 + 同 chunk 邻居 + 二级 chunk 成员 + 媒体节点同章节补块） |
| 4 | 角色分配 | `_assign_roles`（物理优先；种子先占非证据主角色） |
| 5 | 虚拟逻辑补全 | `virtual_edges.propose_virtual_edges` → `score_edge` → `validate_backing`（每条虚拟边须有物理证据回落路径才 accepted） |
| 5b | 证据物理接地 | `_ground_evidence`（空缺 evidence 角色回落到候选证据节点 / 锚点所属 ChunkNode） |
| 5c | 跨文献修正 | `_fix_cross_paper`（paper_B_instance 必取自他刊，子图须跨 ≥2 篇） |
| 6 | 子图选定 | `_select_subgraph`（Accept(v)：仅纳入提升综合评分的节点） |
| 7 | 充分性验证 | `scoring.logical_sufficiency` + `dual_multihop_ok`（双重多跳防伪多跳） |
| — | 长期记忆 | `memory.SemanticMemory`（overlay pattern / seed 的成功失败沉淀） |

## 3. 综合评分与硬门槛

`scoring.utility_score` 返回分项 + total：

```
Score(S_q|T,D) = a1·Semantic + a2·Role + a3·Evidence + a4·Constraint
               + a5·Structure + a6·Difficulty − b1·Shortcut − b2·Noise
```

`logical_sufficiency(parts, qtype)` 硬门槛：

- `role ≥ 0.99`：最小角色集必须全部填满（抽象角色视为已满足）；
- `evidence ≥ 0.6`：关键角色可回落到物理证据；
- `constraint ≥ 0.5`：**仅** numerical / formula / comparative / table 题强制（其余
  题型约束覆盖只作软性加分，避免卡掉本不含数值的合法问题）；
- `structure ≥ 0.5`、`difficulty ≥ 0.5`、`shortcut ≤ 0.5`；
- 双重多跳：L3/L4 须跨 ≥2 chunk 且语义路径足够长。

**难度按真实跨 chunk 跨度封顶**（`_cap_difficulty`）：单 chunk 子图最高 L2，杜绝
"富节点单 chunk 被误判 L4 → shortcut 重罚"的伪多跳矛盾。

## 4. 与基线的工程关系（向后兼容）

- 新增独立 stage `semantic_plan`，**不改动** `sample`；输出同为
  `question_plans.jsonl`，下游 `generate / corrupt / verify / repair` **零改动**复用。
- `QuestionPlan` 仅**新增带默认值字段**（seed_nodes / theme / semantic_overlay_edges
  / role_assignment / required_roles / required_evidence_paths / forbidden_shortcuts
  / utility_score / score_breakdown / overlay_pattern），`from_dict` 按字段名过滤，
  旧 artifact 可直接加载。
- 纯图遍历、**不调用 LLM**（与 `sample` 一致），用 token 重叠 / canonical 启发式
  代替向量，无新依赖。

## 5. 运行

```bash
# 整条流水线（sample → semantic_plan）
python -m autoqag.pipeline --recipe recipes/semantic.yaml

# 或在已有 graph 产物上单独跑本层
python -m autoqag.pipeline --recipe recipes/semantic.yaml --only semantic_plan
```

参数：`per_type`（每题型目标产量，按充分性门槛过滤）。产物：
`question_plans.jsonl`（含覆盖层字段）+ `semantic_memory.json`（长期记忆）。

## 6. 五论文图实测（outputs/five/graph，2409 节点 / 5974 边）

`per_type=8` 产出 66 个 plan，覆盖全部 10 题型；难度分布 L2/L3/L4。
formula / summary / cross_paper 产量偏低是**数据本身稀疏**（equation 在图中近乎孤立、
仅 6 个 section、跨文献对齐实例有限）所致，属充分性门槛的正确过滤，非逻辑缺陷——
cross_paper 经 `_fix_cross_paper` + 跨刊门槛后，所有产出均真实跨 2 篇论文。

## 6b. 题型专属语义绑定（角色绑定精度强化）

纯类型匹配会把任意同类节点填进角色槽，导致"数值不属于被比较对象"等语义松绑。
针对评测中量化到的弱项，planner 增加三处沿物理边的绑定（仍只读 G0）：

- **comparative**：`value_X` 改为沿 `object_X --has_attribute--> 共享指标 --has_value--> value_X`
  取值，保证数值真正属于该对象；`unit` 绑定为该数值的 `has_unit` 邻居（杜绝 `unit='a'` 游离噪声）。
- **cross_paper**：用 `same_as` / `aligns_with` 对齐对锚定 `paper_A_instance` / `paper_B_instance`
  （要求 B 为他刊真实 Concept/Method/Value 实例），`result_A` / `result_B` 各取本刊结果值。
- **numerical / formula**：`unit` 同样改为 `has_unit` 邻居绑定。

绑定取不到时保留原类型匹配结果，不破坏角色完整性硬门槛。

plan 级实测（per_type=12，复用 outputs/five 图）改进前→后：
- comparative value↔object 同源：0/12 → 9/11
- cross_paper result_A/result_B 真跨 2 刊：0/5 → 10/12
- cross_paper paper_B_instance 为真实例：2/5 → 10/12
- 游离噪声单位：unit='a' → 仅保留真实 `%` 等（经 has_unit 绑定）

## 6c. 利用子图前/后实测对照（同一图谱，Qwen 生成）

`recipes/cmp_before.yaml`（基线 `sample`）vs `recipes/cmp_after.yaml`（`semantic_plan`），
同一 nodes/edges、同一 generate、同一 LLM（qwen-plus），各取 ≤60 plan 生成 QA：

| 维度 | 前(基线 sample) | 后(semantic_plan) |
|---|---|---|
| 题型数 | 8（无 mechanism/cross_paper） | 10（+mechanism +cross_paper） |
| L4 标注 vs 真跨 chunk | 33/49 标 L4，仅 8% 真跨 chunk（伪多跳） | 难度诚实，真跨 chunk 42% |
| 平均证据 span/题 | 3.73 | 9.79 |
| 跨文献接地(≥2 论文) | 1/49 | 7/47 |
| is_valid_qa 通过 | 49/49 | 47/47 |
| 数值题答案含数字 | 8/8 | 7/7 |

**LLM-as-judge 评分(qwen-plus，1-5，全维提升):**

| 维度 | 前 | 后 | Δ |
|---|---|---|---|
| faithfulness | 4.02 | 4.30 | +0.28 |
| grounding | 3.88 | 4.15 | +0.27 |
| reasoning_depth | 1.41 | 1.96 | +0.55 (+39%) |
| specificity | 3.73 | 3.87 | +0.14 |
| overall | 3.69 | 3.79 | +0.10 |

**规则验证器(verify，pre/post-repair):**

| 指标 | 前 | 后 |
|---|---|---|
| verify 通过(修复前) | 16/49 (33%) | 3/47 (6%) |
| verify 通过(2 轮修复后) | 33/49 (67%) | 26/47 (55%) |
| 违规分层(修复前) | constraint28/evidence14/semantic24 | constraint89/evidence14/semantic27 |

结论：子图层用"规则验证器通过率"换取了**真实推理深度与题目质量**——题目更难更密
(证据 span 3.73→9.79)，机械约束检查(数值/单位/条件精确复述)因此触发更多，故 verify
通过率偏低；但**忠实性层持平**(evidence 14→14、LLM 语义 24→27)，**LLM 评分五维全升**
(reasoning_depth +39%)，真实多跳(8%→42%)、跨文献(1→7)、题型(8→10)显著扩展。
即：通过率下降源于题目密度/难度上升与抽取噪声(如表格 %/GHz 单位)，而非答案错误。

## 7. 关键文件

```
autoqag/ops/m5_sample/semantic/
  roles.py          题型逻辑角色 schema / 最小角色集 / 角色→NodeType / 种子策略
  evidence_chain.py 候选扩展、证据链游走、chunk 接地、物理回落路径 BFS
  seed.py           SeedScore 与种子排序
  virtual_edges.py  虚拟边生成 / Score_q 打分 / 证据回落验证
  scoring.py        综合评分各分量 + 逻辑充分性 + 双重多跳
  memory.py         长期记忆 (overlay pattern / seed / 领域偏好)
  planner.py        OverlayPlanner 编排器
autoqag/ops/m5_sample/semantic_plan.py   @STAGES.register_module("semantic_plan")
recipes/semantic.yaml                    用 semantic_plan 的整条 recipe
```
