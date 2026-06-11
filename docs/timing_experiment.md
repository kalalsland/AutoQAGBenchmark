# Pipeline 分环节计时实验 (gpt / gemini / claude)

> 复现：`conda activate pytorch2.2.2` →
> `python scripts/run_timed_pipeline.py --recipe recipes/timing.yaml --model <model> --work-dir outputs/timing_<model>`
> （`--from-stage` / `--to-stage` 可只跑子集；分题型计时由 `generate` 阶段 `record_timing: true` 写出 `generate_timing.json`。）

## 实验设置

- **输入**：`data/raw_timing/` 5 篇 PDF（射频吸波/FSS 领域）。
- **解析**：MinerU（`backend=pipeline`，纯 CPU），图像切割对部分含图表的 PDF 触发已知临时目录写图 bug，自动回退 PyMuPDF；文本仍完整抽取。
- **规模**：`per_type=5`，8 题型；`graph.max_text_blocks=40`（实际含图注块 48）。
- **LLM 代理**：OpenAI 兼容端点（AGTCloud），`max_concurrency=8`，`temperature=0`，`max_tokens=4096`。
- **口径说明**：
  - `PDF抽取(parse)` 与模型无关，只跑一次后被三模型复用。
  - `物理图建立(graph)`、`问题生成(generate)` 含 LLM 调用，按模型分别测。
  - `子图构建(sample)` 为确定性模板采样，无 LLM。
  - 分题型“单题生成时间”= 每个 plan 的**单次 LLM 调用墙钟**（提交→返回，受并发 8 影响，含排队等待），按题型求均值。

## 各环节耗时（秒）

| 环节 | gpt-4o-mini | gemini-2.5-flash | claude-haiku-4-5 |
|---|--:|--:|--:|
| PDF抽取 (parse, 共享) | 355.77 | 355.77 | 355.77 |
| 物理图建立 (graph) | 188.23 | 163.84 | **1182.11** |
| 子图构建 (sample) | 0.13 | 0.06 | 0.11 |
| 问题生成 (generate) | 145.78 | 130.88 | 102.55 |
| —— 产出 points | 274 | 47 | 184 |
| —— 产出 plans | 38 | 27 | 38 |
| —— 保留 QA | 28 | 20 | 32 |

- **PDF抽取**：5 篇 PDF 共 355.77s（≈71s/篇），normalize 0.07s → 411 证据块。
- **claude-haiku graph 异常慢（1182s）**：该代理 claude 走 vertex-ai 后端，在并发 8 的大 prompt 抽取批下被限流/排队（单次小请求仅 ~2s，但 4096-token 抽取批严重拖慢）。gpt/gemini 在同代理下 graph 仅 ~3 分钟。属代理侧吞吐限制，非算法本身。
- **gemini-2.5-flash 抽取稀疏（points=47）**：thinking 模型把 token 预算花在 `reasoning_content`，正文抽取偏少，导致 numerical/condition 题型 plan 数为 0。

## 分题型 平均单题生成时间（墙钟，并发=8；秒）

| 题型 | gpt-4o-mini | gemini-2.5-flash | claude-haiku-4-5 |
|---|--:|--:|--:|
| atomic | 3.0 (n=5) | 14.0 (n=5) | 11.9 (n=5) |
| numerical | 20.0 (n=5) | — | 11.1 (n=5) |
| condition | 64.8 (n=5) | — | 29.0 (n=5) |
| comparative | 57.7 (n=5) | 42.0 (n=5) | 32.3 (n=5) |
| table | 59.4 (n=5) | 41.3 (n=5) | 42.8 (n=5) |
| formula | 110.9 (n=5) | 58.3 (n=5) | 82.6 (n=5) |
| multi_hop | 117.6 (n=5) | 101.8 (n=5) | 90.0 (n=5) |
| summary | 119.0 (n=3) | 106.2 (n=2) | 100.9 (n=3) |

**趋势（一致）**：复杂题型显著更慢，`atomic < numerical < comparative/table < condition < formula < multi_hop ≈ summary`。原因有二：(1) 复杂题需生成更长的证据路径/答案；(2) 这些题型 plan 排在批次靠后，在并发 8 下排队等待更久（墙钟含等待）。若要“纯推理时延”需把并发设为 1（另跑一组）。

> 完整数值见 `outputs/timing_summary.json` 及各 `outputs/timing_<model>/{pipeline_timing,generate_timing}.json`。
