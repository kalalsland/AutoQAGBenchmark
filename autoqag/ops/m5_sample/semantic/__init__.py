"""问题级语义覆盖层 / 评分引导的问题子图规划 (Score-guided Question Subgraph
Planning with Virtual Logic Completion)。

实现两份方法论文档的核心创新：
- 语义规划层方法论.pdf —— 评分引导的问题子图规划与虚拟逻辑补全
- 语义层子图构建.pdf   —— 问题级语义覆盖层 (动态子图规划)

核心思想：固定物理证据图 G0 保持不变；针对每个目标问题 q，先做问题逻辑规划 (确定题型
需要哪些逻辑角色)，再在 G0 上做 **评分引导的增量子图扩展**；物理扩展不足时按缺失角色
建立 **临时虚拟逻辑边 (语义覆盖层 Ωq)**，并验证每条虚拟边可回落到物理证据路径；通过
语义完整性 / 证据可回落 / 双重多跳 / 可回答性检查后输出 QuestionPlan。

子模块：
- roles          : 题型 role schema 与逻辑角色 → 节点类型映射
- seed           : 种子节点初始化与评分 (SeedScore)
- evidence_chain : 向下游走 / 向上定位 / 候选角色池构建
- virtual_edges  : 八类虚拟逻辑边生成、问题级打分、证据回落验证
- scoring        : 综合子图评分 (Subgraph Utility Score) 与逻辑充分性判定
- memory         : Overlay Pattern / Failure / Domain Preference 三类长期记忆
- planner        : 端到端编排 (规划 → 扩展 → 补全 → 验证 → QuestionPlan)
"""
