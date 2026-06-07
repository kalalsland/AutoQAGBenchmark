# AutoQAGBenchmark

面向高约束科研文档的**图谱锚定式 Auto-QAG Benchmark 与高级训练语料构建框架**的端到端实现。

把科研 PDF 自动转化为**可追溯、可验证、可评测、可训练、可修复**的科研智能数据资产：

```
PDF → MinerU 解析 → 证据归一化 → Schema-Evidence Graph → 子图采样 & Question Plan
   → QA + 高级训练语料 → 负样本扰动 → 四层约束验证 → violation 驱动自修复 → 验证 QA + 语料
```

## 设计原则

- **高度模块化**：10 个 stage 各自独立，通过工作目录下的命名 artifact (jsonl) 通信，可单独运行任一 stage 局部调试 (`--only`)。
- **最大化复用**：直接复用/改编 data-juicer (registry)、GraphGen (LLM 客户端 / 限流 / networkx 存储 / 抽取 prompt)、MinerU (PDF 解析)。见 [docs/architecture.md](docs/architecture.md)。
- **recipe 驱动**：全流程由 `recipes/mvp.yaml` 声明 (论文 §5.9)，保证可复现。
- **LLM 通用**：默认 OpenAI 兼容 API，`base_url` 可指向 DeepSeek / Qwen / 本地 vLLM。

## 目录结构

```
autoqag/
├── registry.py        # [复用 data-juicer] stage 注册中心
├── config.py          # recipe.yaml 载入
├── pipeline.py        # stage 编排执行器 (CLI 入口)
├── schema.py          # 核心数据模型 (Address/EvidenceBlock/PointNode/Edge/QuestionPlan/QAItem/Violation)
├── common/            # LLM 客户端 / 限流 / 图存储 / IO / 日志
├── templates/         # 各阶段 LLM prompt
├── ops/               # 10 个流水线模块 (m1_ingest ... m10_output)
└── experiments/       # baseline 薄壳 + 指标计算
recipes/mvp.yaml       # 完整 8 模块闭环 recipe
docs/                  # 架构与各模块记录文档
data/raw/              # 放置输入 PDF
```

各模块输入/输出/复用见 [docs/pipeline_modules.md](docs/pipeline_modules.md)；
产物字段规格见 [docs/data_formats.md](docs/data_formats.md)。

## 快速开始 (环境就绪后)

```bash
# 1. 安装依赖
pip install -r requirements.txt
pip install -U "mineru[core]"          # 主解析器；纯 CPU 用 pipeline 后端即可

# 2. 配置 LLM (OpenAI 兼容)
export AUTOQAG_API_KEY=sk-xxx
export AUTOQAG_BASE_URL=https://api.deepseek.com/v1   # 可选，默认 OpenAI
export AUTOQAG_MODEL=deepseek-chat                    # 或在 recipe 里写 model

# 3. 放论文
cp your_papers/*.pdf data/raw/

# 4. 跑完整流水线
python -m autoqag.pipeline --recipe recipes/mvp.yaml

# 5. 看结果
ls outputs/mvp/benchmark/          # benchmark.jsonl + human_review.jsonl
ls outputs/mvp/corpus/             # 高级训练语料
cat outputs/mvp/stats.json         # 统计
python -m autoqag.experiments.metrics --work_dir outputs/mvp
```

### 局部调试 (模块化定位修改)

```bash
python -m autoqag.pipeline --recipe recipes/mvp.yaml --only graph     # 只跑某 stage
python -m autoqag.pipeline --recipe recipes/mvp.yaml --from sample    # 从某 stage 开始
python -m autoqag.pipeline --recipe recipes/mvp.yaml --skip corrupt   # 跳过某 stage
```

无 LLM 时可先验证结构链路：把 recipe 的 `graph.extract_points` 设为 `false`，
跑 `ingest → parse → normalize → graph` 验证解析与结构图。

## 输出 (论文 §5.9)

| 输出 | 路径 |
|---|---|
| Benchmark 数据集 | `outputs/mvp/benchmark/benchmark.jsonl` |
| 高级训练语料 | `outputs/mvp/corpus/{instruction,graph_trace,rag_grounding,refusal,verifier,preference,repair}.jsonl` |
| 图谱数据 | `outputs/mvp/graph/{nodes,edges}.jsonl` + `graph.graphml` |
| Recipe 快照 | `outputs/mvp/recipe_snapshot.yaml` |
| 统计 / 清单 | `outputs/mvp/stats.json` / `dataset_manifest.json` |

## 状态

完整 8 模块闭环代码已实现，**尚未配置环境运行测试**。baseline 对比与完整实验 harness 为薄壳 (见 `autoqag/experiments/`)。
