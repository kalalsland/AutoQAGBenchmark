# 架构与复用说明

## 总体架构

轻量 **registry + recipe.yaml 驱动的 stage 流水线**，不引入 Ray / HF-datasets。

- 每个流水线模块 = 一个 `BaseStage` 子类，用 `@STAGES.register_module("name")` 注册。
- stage 之间通过工作目录 (`work_dir`) 下的命名 artifact (jsonl) 通信，因此任一 stage 可独立运行 (`--only`)，满足"局部定位修改"。
- `recipes/*.yaml` 声明 stage 顺序与参数；`autoqag/pipeline.py` 按序执行。
- `PipelineContext` 贯穿全程，持有 `work_dir`、`global_params` 与惰性构建的共享 `LLMClient`。

```
recipe.yaml ──load──▶ RecipeConfig ──▶ pipeline.run_pipeline
                                          │ for each StageSpec
                                          ▼
                         STAGES.get(name)(**params).run(ctx)
                                          │ 读/写
                                          ▼
                         work_dir/*.jsonl (artifact)
```

## 三个参考库的复用映射

| 本项目文件 | 来源 | 复用方式 |
|---|---|---|
| `autoqag/registry.py` | data-juicer `data_juicer/utils/registry.py` | **逐行 copy** (Apache-2.0)，作 stage 注册中心 |
| `autoqag/common/limiter.py` | GraphGen `models/llm/limitter.py` | **copy** RPM/TPM 限流 |
| `autoqag/common/llm.py` | GraphGen `models/llm/api/openai_client.py` + `bases/base_llm_wrapper.py` | **改编瘦身**：AsyncOpenAI + tenacity 重试 + 限流 + token 计数 + `<think>` 过滤；新增 `from_env` / 同步封装 / 并发信号量 |
| `autoqag/common/graph_store.py` | GraphGen `storage/graph/networkx_storage.py` | **改编**：networkx + graphml + nodes/edges.jsonl 导入导出 + 地址/canonical 索引 |
| `autoqag/schema.py` | GraphGen `bases/datatypes.py` | **改编**：保留 Token/Community 思路，新增本项目全部数据模型 |
| `autoqag/ops/m2_parse/mineru_parser.py` | GraphGen `models/reader/pdf_reader.py` 的 `MinerUParser` | **改编**：subprocess 调 `mineru` CLI；**保留 page_idx/bbox/text_level** (原版删除了，本项目地址需要) |
| `autoqag/templates/point_extraction.py` | GraphGen `templates/kg/kg_extraction.py` | **改编**：实体类型→14 类科研点；沿用元组分隔符，解析逻辑可复用 |
| `autoqag/ops/m5_sample/sample.py` | GraphGen `models/partitioner/{bfs,leiden}_partitioner.py` | **参考算法**，重写为按题型的模板化子图采样 |
| MinerU | 整个 MinerU 仓库 | 作**外部 CLI/库** (`pip install mineru`)，不进本仓库；不可用时回退 PyMuPDF |

## 为什么不直接依赖 data-juicer / GraphGen 的执行器

- GraphGen 的 `engine.py` 与存储/LLM 均基于 **Ray Actor**，data-juicer 执行器基于 **HF datasets + Ray**；二者对 MVP 偏重。
- 本项目的处理单元是"论文语料 + 多类 artifact 的阶段流水"，而非行式 dataset.map，故自建轻量 stage 执行器更贴合，同时仍复用它们最有价值的**单文件组件** (registry / LLM client / 图存储 / prompt)。

## 扩展点

- **新增题型**：在 `m5_sample/sample.py` 加一个 finder 并注册到 `finders`。
- **新增验证器**：在 `m8_verify/verifiers.py` 加 checker 并登记到 `CHECKERS`，recipe 里开关。
- **新增错误类型**：在 `m7_corrupt/corruptors.py` 加函数并加入 `ALL_CORRUPTORS`。
- **换 LLM/解析后端**：改 recipe 的 `global_params.llm` / `parse.backend`，无需改代码。
- **换图存储**：`common/graph_store.py` 接口稳定，可替换为 kuzu 等后端。
