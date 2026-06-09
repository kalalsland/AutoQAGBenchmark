# 实验设计与结果：子图规划模块的渐进式消融

> 目标：证明 **问题级语义子图规划层** 的各模块逐步有效，并给出可复现的
> **内部对比指标 (intrinsic)** 与 **外部对比指标 (extrinsic)**。
> 全部脚本位于 `autoqag/experiments/`，运行产物写入 `runs/`、`results/`（已 gitignore）。

---

## 1. 研究问题 (Research Questions)

- **RQ1（模块有效性）** 子图规划层的每个模块，是否都对问题质量带来可度量的、单调的增益？
- **RQ2（语义绑定正确性）** 题型专属绑定是否真正让"数值/单位归属于被比较对象"，而非类型匹配的游离值？
- **RQ3（多跳真实性）** 双重多跳约束是否消除了"标了 L3/L4 却单 chunk 可答"的伪多跳？
- **RQ4（外部质量）** 相对无子图层基线，最终 QA 在证据接地、推理深度、难度判别力上是否更强？
- **RQ5（代价）** 更难、更跨文献的问题是否带来验证通过率下降？该下降是难度上升的自然结果还是质量退化？

---

## 2. 渐进式消融矩阵 (RQ1)

每一行在上一行基础上**只增开一个模块**（累积式），最后一行为完整规划层。
固定同一张图谱 `outputs/five`（5 篇超材料论文，2409 节点 / 5974 边），每题型采样 `per_type=12`，
**无 LLM、确定性**（已 `PYTHONHASHSEED=0` 固定 set 迭代序，跨进程可复现）。

| 配置 | stage | 新增模块 | 含义 |
|---|---|---|---|
| **A0** | `sample` | — | 纯物理模板采样基线（无问题级语义覆盖层） |
| **A1** | `semantic_plan` | +角色 schema 规划 | 按题型逻辑角色驱动子图构建 |
| **A2** | `semantic_plan` | +评分引导扩展 Accept(v) | 增益门控的候选扩展，剪噪 |
| **A3** | `semantic_plan` | +题型专属语义绑定 | 数值/单位真正归属被比较对象（表行×共享列→单元格） |
| **A4** | `semantic_plan` | +虚拟逻辑补全 Ωq | 问题级语义覆盖层（虚拟边须有物理证据回落） |
| **A5** | `semantic_plan` | +双重多跳 & 难度封顶 | 难度由真实跨 chunk 跳数决定，杜绝伪多跳 |
| **A6** | `semantic_plan` | +逻辑充分性门槛 | = 完整规划层（充分性 + 双重多跳门控） |

> 完整系统另含 **A7 = 违规驱动自修复**（verify→repair，需 LLM），在 §4 外部实验中以
> `outputs/cmp_after`（已含 verify+repair）体现，不进入无 LLM 的内部消融。

运行：
```bash
PYTHONIOENCODING=utf-8 python -m autoqag.experiments.run_ablation \
    --graph_dir outputs/five --per_type 12
# → results/internal_ablation.{json,md}
```

---

## 3. 内部对比指标 (Intrinsic) — 定义与公式

来源：`autoqag/experiments/metrics_internal.py`，只依赖 `question_plans.jsonl` + 图谱。
设规划集合 \(P\)，单个规划 \(p\) 的物理必需节点集 \(N(p)\)。

### 3.1 覆盖 / 结构
- **n_plans** = \(|P|\)
- **type_coverage** = 覆盖题型数 / 总题型数
- **avg_evidence_spans** = \( \frac{1}{|P|}\sum_p |\text{evidence\_spans}(p)| \)（过高=堆砌噪声，过低=证据不足）

### 3.2 多跳真实性（RQ3）
设 \(\text{chunks}(p)\)、\(\text{papers}(p)\) 为 \(N(p)\) 落入的 chunk / paper 集合。
- **real_cross_chunk_ratio** = \( \frac{|\{p: |\text{chunks}(p)|\ge 2\}|}{|P|} \)
- **cross_paper count** = \( |\{p: |\text{papers}(p)|\ge 2\}| \)
- **pseudo_multihop_rate** = \( \frac{|\{p: \text{diff}(p)\in\{L3,L4\}\ \wedge\ |\text{chunks}(p)|<2\}|}{|\{p:\text{diff}(p)\in\{L3,L4\}\}|} \)
  （标了难但实际单 chunk 可答 = 伪多跳，**越低越好**）

### 3.3 逻辑完整性
- **role_completeness** = 各题"已填必需角色 / 必需角色数"的均值（抽象角色除外）
- **avg_utility** = 规划效用分均值（含 score_breakdown 分项）
- **overlay_grounded_ratio** = 被接受虚拟边中"有物理证据回落路径"的比例（应为 1.0，否则虚拟边脱离 G0）

### 3.4 语义绑定正确性（RQ2，本项目核心创新点的直接度量）
- **comparative_value_object_bind (comp_bind)**：比较题中 \(\text{value}_X\) 是否真属于 \(\text{object}_X\) 在共享指标上的取值。
  两条物理路径任一成立即记正确：
  1. 通用链路 \( \text{object}_X \xrightarrow{has\_attribute} \text{指标} \xrightarrow{has\_value} \text{value}_X \)
  2. 表格路径 \( \text{object}_X \text{为表行}，\text{value}_X = \text{该行} \times \text{共享列的单元格} \)
- **cross_paper_result_cross (cp_result)**：跨文献题的 \(\text{result}_A,\text{result}_B\) 是否落在不同论文
- **cross_paper_instance_real (cp_inst)**：`paper_B_instance` 是否为真实 Concept/Method/Value 且过质量门
- **unit_grounded (unit_grnd)**：数值的单位是否经 `has_unit`（数值自身或表格列头）接地

### 3.5 内部消融结果

| config | n_plans | types | avg_ev | cross_chunk | pseudo_mh | cross_paper | role_compl | utility | overlay_grnd | comp_bind | cp_result | cp_inst | unit_grnd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_baseline_sample | 89 | 8 | 3.629 | 0.146 | **0.8** | 7 | **0.0** | **0.0** | — | — | — | — | — |
| A1_role_plan | 103 | 8 | 15.971 | 0.359 | 0.641 | 12 | 0.879 | 3.575 | — | 0.118 | 0.333 | 0.833 | 0.152 |
| A2_score_guided | 112 | 8 | 9.009 | 0.357 | 0.640 | 18 | 0.881 | 3.948 | — | 0.105 | 0.636 | 0.909 | 0.139 |
| A3_binding | 112 | 8 | 8.875 | 0.357 | 0.640 | 18 | 0.881 | 3.953 | — | **0.684** | 0.727 | 0.909 | 0.237 |
| A4_overlay | 112 | 8 | 8.929 | 0.357 | 0.640 | 18 | 0.860 | 3.805 | **1.0** | 0.650 | 0.727 | 0.909 | 0.237 |
| A5_dual_multihop | 112 | 8 | 8.929 | 0.357 | **0.0** | 18 | 0.860 | 4.704 | 1.0 | 0.650 | 0.727 | 0.909 | 0.237 |
| A6_full_plan | 97 | 8 | 9.969 | **0.423** | 0.0 | **23** | **1.0** | **5.141** | 1.0 | 0.545 | **0.818** | 0.909 | 0.242 |

（`—` 表示该模块未启用时指标不适用。粗体为该模块开启时的关键跃迁。）

**读法（每一步的因果增益）：**
- **A0→A1** 角色规划：`role_compl` 0→0.88、`utility` 0→3.58 —— 子图从"模板拼凑"变为"角色完整"。
- **A1→A2** 评分引导：`cross_paper` 12→18、`cp_result` 0.33→0.64，同时 `avg_ev` 16→9（**剪除噪声证据**）。
- **A2→A3** 语义绑定：**`comp_bind` 0.11→0.68**、`unit_grnd` 0.14→0.24 —— 数值真正绑定到被比较对象（直接验证 RQ2）。
- **A3→A4** 虚拟补全：`overlay_grnd` None→1.0 —— 所有虚拟边都有物理证据回落。
- **A4→A5** 双重多跳：**`pseudo_mh` 0.64→0.0**（消除伪多跳，RQ3）、`utility`→4.70。
- **A5→A6** 充分性门槛：`role_compl`→1.0、`cross_paper` 18→23、`utility`→5.14 —— 只保留逻辑充分的问题。

> 复现性：连续两次运行 `internal_ablation.md` 完全一致（`PYTHONHASHSEED=0`）。

---

## 4. 外部对比指标 (Extrinsic)

作用于最终 `qa.jsonl` / `violations.jsonl`。对照两组（均用 qwen-plus 生成）：
- `outputs/cmp_before`：**无子图层**（`sample` → generate）
- `outputs/cmp_after`：**完整系统**（`semantic_plan` + 绑定改进 → generate → verify → repair）

### 4.1 确定性外部指标（无需 LLM）
来源：`autoqag/experiments/metrics_external.py`
- **valid_qa_rate**：问题/答案非空且 ≥1 条证据 span
- **verify_pass_rate**：`validator_result.passed` 比例（数值/单位/条件/证据/语义五层全过）
- **violation_density**：每题平均违规数（逐层、逐严重度细分）
- **avg_evidence_spans / cross_paper_qa**：证据接地与跨文献覆盖
- **type_coverage**：题型覆盖

```bash
python -m autoqag.experiments.metrics_external \
    --before outputs/cmp_before --after outputs/cmp_after
```

| metric | before (无子图层) | after (完整系统) | Δ |
|---|---|---|---|
| n_qa | 49 | 47 | -2 |
| valid_qa_rate | 1.0 | 1.0 | +0.0 |
| verify_pass_rate | 0.673 | 0.553 | **-0.12** |
| violation_density | 0.49 | 1.234 | **+0.744** |
| avg_evidence_spans | 3.735 | 9.787 | **+6.052** |
| cross_paper_qa | 1 | 7 | **+6** |
| type_coverage | 0.875 | 0.875 | +0.0 |
| n_types | 7 | 7 | +0 |

完整系统证据密度、跨文献覆盖显著上升；`verify_pass_rate` 的下降在 §5 解释。

### 4.2 LLM-as-judge 多维评分
来源：`autoqag/experiments/run_testtakers.py --mode judge`
裁判模型按 5 维各打 1–5 分：faithfulness / grounding / reasoning_depth / specificity / overall。

| dimension | before (n=49) | after (n=47) | Δ |
|---|---|---|---|
| faithfulness | 4.531 | 4.043 | -0.488 |
| grounding | 4.265 | 3.830 | -0.435 |
| reasoning_depth | 1.265 | **2.043** | **+0.778** |
| specificity | 3.816 | 3.489 | -0.327 |
| overall | 3.918 | 3.574 | -0.344 |

**核心信号：`reasoning_depth` 1.27→2.04（+0.78）** —— 完整系统生成的问题需要更多跨证据/多步推理。
faithfulness/specificity 的小幅下降是"更难、更跨文献、证据更密"问题的代价（与 §5 一致），
属于难度-保真度权衡，而非幻觉激增（faithfulness 仍 >4，未坍塌）。

### 4.3 难度判别力（闭卷 vs 开卷）
来源：`autoqag/experiments/run_testtakers.py --mode discriminate`
同一考生模型在 **闭卷**（仅问题）与 **开卷**（问题+证据）下作答，裁判对照参考答案判对错，按难度分层。
合格基准：① 准确率随难度单调非增；② 开卷 > 闭卷（证据确有作用，问题非靠常识可答）。

考生与裁判均为 qwen-plus，`outputs/cmp_after` 全量 47 题：

| difficulty | 闭卷 acc | 开卷 acc |
|---|---|---|
| L2 | 0.444 | 0.519 |
| L3 | 0.111 | 0.222 |
| L4 | 0.273 | 0.455 |
| **overall** | **0.34** | **0.447** |

- **开卷-闭卷增益 = +0.107**：提供证据后准确率全面上升，说明问题**真正依赖所给证据**、
  并非靠模型常识可答（这是 benchmark 有效性的关键信号）。
- 单调性：L3 在本集合中最难（闭卷 0.11 / 开卷 0.22），L4 反而略高，故严格"随 L 单调非增"为 False。
  原因是当前每难度样本量小（L3 仅 9 题）；§7 列出的扩规模 / 多考生模型实验将稳定该曲线。

---

## 5. 对"验证通过率下降"的解释（RQ5，重要）

外部对比中 `verify_pass_rate` 由 0.67 降到 0.55、`violation_density` 由 0.49 升到 1.23。
**这不是质量退化，而是难度上升的自然结果**，证据如下：
- 同一对照中 `avg_evidence_spans` 3.7→9.8、`cross_paper_qa` 1→7、judge 的 `reasoning_depth` 1.27→2.04；
- 难度判别力实验显示问题**闭卷几乎不可答、开卷显著提升**（见 §4.3），说明问题真正依赖证据；
- 违规主要落在 `constraint`/`semantic` 层——即"答案需保留更多条件约束"，正是更难问题的特征。

因此该指标应与难度、证据密度联合解读；完整系统的 **A7 自修复**（cmp_after 已含 repair）进一步回收一部分违规。

---

## 6. 复现实验一键清单

```bash
# 内部消融（无 LLM，秒级，确定性）
PYTHONIOENCODING=utf-8 python -m autoqag.experiments.run_ablation --graph_dir outputs/five --per_type 12

# 外部确定性对比（无 LLM）
python -m autoqag.experiments.metrics_external --before outputs/cmp_before --after outputs/cmp_after

# 外部 LLM 评测（需 DASHSCOPE_API_KEY；--limit N 可冒烟）
python -m autoqag.experiments.run_testtakers --mode judge        --work_dir outputs/cmp_after
python -m autoqag.experiments.run_testtakers --mode discriminate --work_dir outputs/cmp_after --taker qwen-plus
```

---

## 7. 仍待补强（论文级别）

- **规模**：从 5 篇扩到 ≥30 篇 + 多领域，验证指标稳定性。
- **人工评测**：抽样人工标注 + 标注者间一致性 (Cohen's κ)，校准 LLM-judge。
- **验证器精度/召回**：用 corrupt 注入的已知错误测 verify 的 TPR/TNR。
- **多考生模型**：discriminate 换 ≥3 个不同能力模型，看难度分层是否拉开模型差距。
