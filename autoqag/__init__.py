"""AutoQAGBenchmark: 面向高约束科研文档的图谱锚定式 Auto-QAG Benchmark 与训练语料构建框架。

端到端流水线: PDF → MinerU解析 → 证据归一化 → Schema-Evidence Graph
→ 子图采样&QuestionPlan → QA+训练语料 → 负样本 → 四层验证 → violation修复 → 输出。
"""

__version__ = "0.1.0"
