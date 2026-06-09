"""autoqag.experiments: 消融与对比实验 (论文 §6)。

模块：
- metrics.py           基础指标 (题型覆盖/难度分布/验证器检出率)。
- metrics_internal.py  内部对比指标 (无 LLM，确定性)：覆盖、多跳真实性、
                       逻辑完整性、语义绑定正确性。
- run_ablation.py      渐进式消融 harness (A0 sample → A6 完整规划)，只读源图谱。
- baselines.py         外部对比方法占位 (Vanilla/Chunk/RAG/GraphGen-style)。

运行产物写入 runs/ 与 results/ (已 gitignore，不入库)。
"""
