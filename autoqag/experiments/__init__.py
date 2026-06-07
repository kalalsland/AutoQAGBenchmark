"""autoqag.experiments: baseline 与指标计算 (论文 §6)。

本目录为实验薄壳：
- metrics.py 已实现可直接从流水线产物计算的指标 (题型覆盖/难度分布/验证器检出率)。
- baselines.py 为对比方法占位 (Vanilla/Chunk/RAG/GraphGen-style/Ours w-o-graph)，标注 TODO。
完整实验 harness 待数据与环境就绪后扩展。
"""
