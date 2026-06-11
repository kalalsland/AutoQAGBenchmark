# 路线 A 评审与推进规划（方法侧 AAAI / D&B 方向）

> 文档目的：把多轮讨论的判断**沉淀成可逐条比对的清单**。读法——每节末尾的 ✅/⚠️/❌
> 是"严苛审稿人视角"的当前状态；"参考文献"给出可直接对照的同类工作链接。
> 评审依据：通读 `autoqag/ops/m5_sample/semantic/*`、`autoqag/ops/m4_graph/quality.py`、
> `autoqag/experiments/*`（含 `internal_ablation.md`、`cf_probe.json`、`experiment_design.md`）。
> 撰写日期：2026-06-09。

---

## 0. 一页结论（先行）

- **路线选择**：方法侧（头条 = C1 子图规划方法；C2/C3 降为支撑；benchmark 作为产物）—— **方向可行**。
- **但现在不能开跑头条实验**，原因有三，按severity排序：
  1. ❌ **没有外部基线**（KBQG/GraphRAG/SciQAG 类）——方法论文若无基线对比，"方法更强"无从证明，是 **reject 级**硬伤。
  2. ❌ **零人工评测**——"流水线获专家认可"这句话当前**没有任何数据支撑**，写进论文会被当场击穿。
  3. ⚠️ **指标多为"自定义→自优化→自达标"的 proxy**，未锚定人类判断，继续优化会"越走越偏"。
- **正确顺序**：先钉指标判定规范（§4）→ 补外部基线 + 人工 study（§3 的 ①②）→ 救 C2/防翻盘（③④）→ 最后扩规模正式跑。
- **规模锚点**：≥3 域 × ≥10 篇（舒服线 ≥50 篇）；300–500 题，每"题型×难度"格 ≥20–30 题；人工抽样 50–100 条 × ≥2 标注者 + Cohen's κ。

---

## 1. 三个创新点的严苛复核（基于真实代码）

### C1 · Question-level Semantic Overlay Planning（拟作头条）
- **代码实际做了什么**：role schema 角色填槽 + utility-score 引导的贪心子图扩展（`Accept(v)` 增益门，
  `planner.py:498-506`）+ 虚拟覆盖边（须能回落到物理证据路径才接受，`virtual_edges.py:236-245`）。
- **真正亮点**：虚拟边"无物理证据回落则 reject"——规划层与证据层分离、保可追溯。**这是最该主打的点。**
- **审稿人会攻击**：
  - 新颖性：基于 KG 子图的题目生成是成熟方向（见参考 [1][2][6]），需显式对标，说清"贪心 utility 门"之外的概念新意。
  - utility 权重全手调（`virtual_edges.py:18-29`），无学习、无敏感性 → "weights appear arbitrary"。
  - **循环论证**：`internal_ablation.md` 里 utility 3.575→5.141，但 utility 是自定义目标，优化并报告自己的目标 = 无信息量。
- **状态**：⚠️ 实现扎实、有亮点，但**缺外部基线 + 缺去循环的外部度量**。

### C2 · Traceable Polarity-aware Evidence Graph Substrate（降为支撑）
- **代码实际做了什么**：节点带 `address/chunk_id` 可回溯原文（traceable）；边经 `detect_polarity`
  标 positive/negative/contrastive/hypothetical，再 confidence 门控，使非正向边不进证据层
  （`quality.py:234-275`）。
- **审稿人会攻击**：
  - "Traceable graph grounded in spans" 是 KG-RAG/GraphRAG 标配，单独不构成贡献。
  - **polarity 检测是浅层正则**（`quality.py:234` 在字符窗口里匹配否定/对比/假设关键词），**全文无 P/R 数字**。
  - 魔法数：`POLARITY_PENALTY={1.0/0.45/0.35/0.25}`、阈值 `0.5`（`quality.py:193-200`）未验证。
- **状态**：⚠️ 合理 safeguard，但**不给 P/R 就不能当贡献**；不投入标注则降为"实现细节"。

### C3 · Violation-driven Plan Repair & Benchmark Validity Verification（降为支撑）
- **代码实际做了什么**：缺角色→虚拟补全；sufficiency/shortcut/dual-multihop 门不过→**丢弃 plan
  (return None) + 记录失败模式到记忆**（`planner.py:143-157`）；难度按真实跨 chunk 封顶（`_cap_difficulty`）。
- **审稿人会攻击**：
  - **"Repair" 过卖**：主要是**拒绝** + 缺角色补全，非"诊断→修复→重规划"闭环；`record_pattern` 是日志，无实验证明记忆让后续 plan 变好。
  - **"Validity Verification" 循环论证**：validity 由自定义门定义→强制→报告达标。`pseudo_mh 0.8→0.0`
    只是"我的检测器说没伪多跳"，**不等于人类确认是真多跳**。
- **状态**：❌ 作为头条不成立；补人工 validity study 后可并入 C1。

---

## 2. 对两个核心说法的最严苛审查

### 说法 A："投 AAAI 侧重方法，benchmark 是顺带产出"
- **逻辑自洽**，但前提未满足：AAAI 对"方法"的门槛 = **概念新意 + 受控实验证明强于 baseline**。
- 当前 C1 是已知技术的工程组合，**且全部消融对比的是自家消融、无外部 QG 基线**。
- 审稿人原话级别预测："The authors optimize and report their own objective; without comparison to
  existing QG methods (e.g., SciQAG [3], subgraph-guided QG [1]), the method's advantage is unsubstantiated."
- **结论**：✅ "侧重方法"方向对，但 ❌ **比 benchmark 路线更刚需外部基线**——没有基线，方法论文不成立。

### 说法 B："我们的流水线得到专家认可"
- ❌ **当前不能写进论文**。审稿人只问：认可有数据吗？几个专家、什么协议、κ 多少？现在**零人工评测**。
- 凭开发者/组内口头判断 = anecdotal evidence，在 AAAI 是反作用。
- 反证：`experiment_design.md §4.2` 里 faithfulness/grounding/specificity/overall **全跌**，仅 reasoning_depth 涨；
  审稿人会说"你自己的 judge 都显示质量在降"。
- **结论**："专家认可"必须用真人工 study + κ 兑现（见参考 [1][4][5][7] 的协议），否则**删掉这句**。

> 共性问题：两个说法都把"想要的结论"当成"已有的证据"。前者前提是**外部基线**，后者前提是**人工 study**。

---

## 3. 路线 A 必补四项 + 隐藏拦路石（缺什么 / 是代码还是数据）

| 需求 | 现状 | 缺的是 | 工作量 | 参考 |
|---|---|---|---|---|
| **① 外部基线 KBQG/GraphRAG/SciQAG** | `baselines.py` 中 chunk/rag/vanilla/graphgen **全 TODO**，仅 `ours_wo_*` ready | 真代码：chunk-based/RAG/vanilla 生成 stage + 接 GraphGen/SciQAG | 大 | [1][3] |
| **② 人工 validity study** | **完全没有**（§7 自列待补） | 抽样脚本 + 标注 rubric + 导出 + Cohen's κ | 中 | [1][4][5] |
| **③ polarity 内在 P/R** | `detect_polarity` 有，无金标注集、无评测脚本 | 几百条边 gold polarity + scorer | 小代码+标注 | [4] |
| **④ 权重/阈值敏感性** | `run_ablation.py` 在，未扫 `_SCORE_WEIGHTS`/`POLARITY_PENALTY`/阈值/`tau` | sweep driver（复用现有指标） | 小到中 | — |
| **🚧 数据规模（拦路石）** | 单领域 5 篇、~100 plans、47 QA | 语料：≥30 篇 + 多领域 | 大 | [3][7] |

**为什么"先做实验"是错的顺序**：①②③④ 的所有数字在扩规模后都会变；现在在 47 题上跑人工 study/基线/κ，扩规模后全得重来。**规模是其它一切的前置。**

---

## 4. 指标判定规范（最该立刻做的一步——止住"越走越偏"）

> 你"越走越偏"的担忧是对的：根因是**指标未锚定人类判断就盲目优化**。
> 规则：每个指标必须定死四件事 —— **(a) 精确定义 (b) 判定"对"的充要条件 (c) 它是 proxy 还是
> ground truth (d) 用什么外部信号校准它**。写不出第 (d) 项的指标，不该进论文。

| 指标 | (a) 定义 | (b) 判"对"条件 | (c) 性质 | (d) 校准方式 | 当前问题 |
|---|---|---|---|---|---|
| **comp_bind** | value 经物理边可达 object | 通用链路 or 表行列结构成立 | **proxy** | 人工标 gold「数值是否真属于该对象」，报 proxy↔human 相关 | 只验图结构可达，非题目正确性 |
| **unit_grounded** | 单位经 has_unit 接地 | 数值自身/列头 has_unit 命中 | **proxy** | 同上，人工抽检单位归属 | 同上 |
| **pseudo_multihop** | 标 L3/L4 却单 chunk | 跨 chunk < 2 | **弱 proxy** | 强判定：闭卷不可答∧开卷可答∧去任一跳即错 | 跨 2 chunk ≠ 真多跳 |
| **utility** | 加权效用和 | — | **自定义目标** | ④ 敏感性证明结论不随权重翻盘 | 权重全拍，自优化自报告 |
| **role_completeness** | 已填必需角色比 | 角色非空 | proxy | 人工判角色填得对不对（非仅非空） | 填了≠填对 |
| **judge 五维** | LLM 打 1–5 | — | **有偏 proxy** | 多裁判 + 含强模型 + 对人工 gold 校准 | 用考生之一当裁判、分数饱和 |
| **discriminate 开卷-闭卷增益** | 开卷 acc − 闭卷 acc | 开卷 > 闭卷 | **较硬信号** | 这是少数接近 ground truth 的指标，保留并强化 | 样本小，曲线不稳 |

> 关键认知：comp_bind/unit_grounded/role_completeness 都是"**图结构 proxy**"，证明的是"我的图能连上"，
> **不是**"题目对人类是好题"。论文里必须用人工 gold 把 proxy 和人类判断挂钩，否则审稿人视之为自证。

---

## 5. 推进顺序（顺序别乱）

1. **先停 → 写《指标判定规范》**（§4 的 a/b/c/d 四要素，把每个指标锚定到可被人工校准）。不依赖新数据/新代码，马上能做。
2. **补 ① 外部基线 + ② 人工 study**（50–100 条 / ≥2 人 / κ）——同时支撑"方法更强"和"专家认可"两个说法。
3. **③ polarity P/R、④ 敏感性**——救 C2、防权重翻盘。
4. **扩规模到 ≥30 篇 × 3 域**，正式跑全部实验，一次成稿。

---

## 6. 规模锚点（对照同类论文）

| 工作 | 语料 | 题量 | 人工验证协议 | 定位 |
|---|---|---|---|---|
| SciQAG [3] | 22,743 篇 / 24 域 | 188k QA | 1,200 篇测试集 GPT-4 RACAR 打分过滤 | 大规模数据集 |
| TutorQA (Graphusion) [7] | 单域(NLP) | 1,200 QA | 全部专家验证 | 概念图推理 benchmark |
| Subgraph-Guided KG-QG [1] | WebQuestions/PathQuestions | 标准集 | **50 例/系统 × 6 评分员，1–5 分，盲标** | 子图 QG（与本工作最像）|
| MAKES-QA [4] | 科学文献 | — | **50 三元组/框架 × 2 专家，κ=0.45–0.57，盲标，3 点 Likert** | KG 构建+QA |
| BioGraphletQA [6] | 生物 | 复杂 QA | 单标注者（被作者列为局限）| graphlet 锚定 |

**给本工作的最低线 / 舒服线**：
- 语料：单域不够 → **≥3 域 × 10 篇 = 30 篇**最低；**≥50 篇**舒服。
- 题量：方法论文不靠题海 → **300–500 题**，但每"题型 × 难度"格 **≥20–30 题**（修正当前 L3 仅 9 题导致曲线不可读）。
- 人工：抄 [1][4] → **抽样 50–100 条 × ≥2 标注者 + Cohen's κ**，盲标，维度 = 句法正确/语义正确/与子图相关/答案唯一。

---

## 7. 参考文献（逐一可对照）

- **[1] Toward Subgraph-Guided Knowledge Graph Question Generation with GNNs** — 与本工作"子图规划"最近；含小规模人工评测协议（50 例/系统 × 6 评分员，1–5 分，盲标）。
  https://arxiv.org/pdf/2004.06015
- **[2] Knowledge Graph for Efficient Multi-hop Question Generation (AAAI Student Abstract, 2024)** — AAAI 场 KG 多跳 QG 直接对标。
  https://ojs.aaai.org/index.php/AAAI/article/view/42248
- **[3] SciQAG: A Framework for Auto-Generated Science QA Dataset (2024)** — 科学文献自动 QA 生成的体量与评测标杆（22,743 篇 / 188k QA / RACAR 过滤）。
  https://arxiv.org/abs/2405.09939 ｜ 代码：https://github.com/MasterAI-EAM/SciQAG
- **[4] QA Extraction from Scientific Articles Using KGs and LLMs (2025)** — KG+LLM 科学 QA 抽取，SME 间与 SME-LLM 的 Cohen's κ 协议可直接借鉴。
  https://arxiv.org/pdf/2507.13827
- **[5] MAKES-QA: Multi-agent KG Construction & Enrichment for QA** — 人工验证协议（50 三元组/框架 × 2 专家盲标，κ=0.45–0.57）。
  https://www.sciencedirect.com/science/article/pii/S0957417426010699
- **[6] BioGraphletQA: Knowledge-Anchored Generation of Complex QA Datasets** — graphlet 锚定 QA；其"单标注者"局限是反面教材（提醒你必须 ≥2 标注者）。
  https://arxiv.org/html/2604.26048
- **[7] Graphusion / TutorQA: LLM for Scientific KG Fusion & Construction** — 单域概念图 QA benchmark（1,200 QA，专家验证），规模可对照。
  https://arxiv.org/pdf/2407.10794
- **[8] IRB: Automated Generation of Robust Factuality Benchmarks** — 图驱动生成三步法（KG 构建 → 掩码/变换 → 逐步生成），方法结构可对照。
  https://arxiv.org/pdf/2602.08070
- **[9] CIKQA: Commonsense Inference with Knowledge-in-the-loop QA** — 高一致性人工标注范例（κ=0.83–0.87），可作 κ 目标参照。
  https://arxiv.org/pdf/2210.06246

> 备注：以上链接由联网检索获得，请打开后**自行核对作者、年份、数值**（检索摘要可能有偏差，尤其 [6][8] 的 arXiv 编号请以打开后页面为准）。
