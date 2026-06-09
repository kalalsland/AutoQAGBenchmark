# autoqag.experiments

子图规划层的**消融与对比实验**。完整设计、指标公式与已填结果见
[`experiment_design.md`](./experiment_design.md)。

## 模块

| 文件 | 作用 | 需要 LLM |
|---|---|---|
| `metrics_internal.py` | 内部对比指标（覆盖/多跳真实性/逻辑完整性/语义绑定） | 否 |
| `run_ablation.py` | 渐进式消融 A0→A6 runner（确定性，`PYTHONHASHSEED=0`） | 否 |
| `metrics_external.py` | 外部确定性指标（验证通过率/违规密度/证据接地） | 否 |
| `run_testtakers.py` | LLM-judge 多维评分 + 难度判别力（闭卷/开卷） | 是 |
| `metrics.py` / `baselines.py` | 既有基础指标与基线占位 | 否 |

## 快速运行

```bash
# 1) 内部消融（秒级，无 LLM，可复现）
PYTHONIOENCODING=utf-8 python -m autoqag.experiments.run_ablation \
    --graph_dir outputs/five --per_type 12

# 2) 外部确定性对比（无 LLM）
python -m autoqag.experiments.metrics_external \
    --before outputs/cmp_before --after outputs/cmp_after

# 3) 外部 LLM 评测（需 DASHSCOPE_API_KEY；--limit N 可冒烟省 API）
python -m autoqag.experiments.run_testtakers --mode judge        --work_dir outputs/cmp_after
python -m autoqag.experiments.run_testtakers --mode discriminate --work_dir outputs/cmp_after --taker qwen-plus
```

结果写入 `results/`，中间产物写入 `runs/`（均已 gitignore）。

## 关键结论（详见 experiment_design.md）

- **每个模块都有单调增益**：`role_compl` 0→1.0、`utility` 0→5.14、`cross_paper` 7→23。
- **语义绑定有效**：`comp_bind` 在 A3 开启绑定时 0.11→0.68（数值真正归属被比较对象）。
- **伪多跳清零**：`pseudo_multihop_rate` 在 A5 双重多跳约束下 0.64→0.0。
- **外部质量**：LLM-judge `reasoning_depth` 1.27→2.04；开卷比闭卷准确率高 +0.107（问题真依赖证据）。
- **验证通过率下降是难度上升的代价**，非质量退化（见 §5）。
