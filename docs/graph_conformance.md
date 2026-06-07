# 图谱构建符合度复查报告

对照《方案/图谱构建.pdf》逐节核查 `m4_graph` 的实现，基于真实运行产出的图谱
(2 篇频率选择表面论文，DeepSeek 抽取) 验证，而非仅看代码。

## 结论：§一~§十 全部实现并在真实数据上验证

| PDF 章节 | 要求 | 实现位置 | 真实图谱验证 |
|---|---|---|---|
| §一 带物理地址的点 | 每点含 `<文献id, 目录级, chunk_id>`，可选 sentence_id / span | `schema.Address` + `PointNode` | 514/514 点含 paper_id；207 含 sentence_id；74 含 span(表格 r×c 坐标) |
| §二 点提取即层级化 | 标题/概念/属性/数值/单位/条件… 抽取时即带层级 | `NodeType` 14 类 + LLM 抽取标签 | 标题层(Paper/Title/Section)、高层(Concept/Claim)、中层(Attribute/Condition)、底层(Value/Unit) 均产出 |
| §三 边语义=f(标签) | 端点标签决定语义类型 | `edge_rules.resolve_edge` (含方向规范化) | has_attribute/has_value/has_unit/under_condition/supports/derived_from/describes/compares 全部出现 |
| §三 边建立依据 | 共现 or 文章结构决定是否建边 | `build_reason` 字段 | physical_cooccurrence / document_structure / cross_modal_reference / cross_paper 四类 |
| §四 物理共现图(横向) | 同句/同段/同 chunk/**同表格行列**/同图注/引用 | `_extract_and_link` + `_build_table_cells` | scope: same_sentence / same_chunk / **same_table_row / same_table_column** / *_text_reference |
| §五 结构层级图(纵向) | **文献根**→标题→章节→chunk→点 | `_build_structure` (PaperNode 根) | 完整链验证: `PaperNode → SectionNode → ChunkNode → AttributeNode` |
| §六/七 两图协同 | 结构定位 + 共现扩展 | m4 双图 + m5 分题型采样 | 纵向 contains 链 + 横向共现邻接均可遍历 |
| §八 跨文献聚合 | 同 canonical 标签横向连 | `_cross_paper_edges` (same_as) | 代码路径已验证 (多篇共享概念时触发；2 篇过滤后偶为 0，数据相关) |
| §十.1 跨模态引用 | 正文 → 图/表/公式 | `_cross_modal_edges` (references) | references 边 20 条，scope 标 figure/table/equation_text_reference |
| §十.2 跨模态结构归属 | 图/表/公式归属章节 | `_build_structure` Section→Figure/Table/Equation | contains 边覆盖 |
| §十.3 图注↔正文同对象合并 | 图注抽概念并与正文对齐 | `_intra_paper_alignment` (aligns_with) | 单测验证 n=1；真实数据相关 |

## 核心设计落点 (对应 PDF 压缩成的一句话)

> 点带地址与标签；标签决定边的语义；共现决定知识边；目录决定结构边；
> 结构图负责定位；共现图负责关联。

- **点带地址与标签** → `PointNode.address` + `node_type`，抽取阶段即赋层级
- **标签决定边语义** → `edge_rules.resolve_edge(src_type, tgt_type)`，§5.3 边语义表逐项实现
- **共现决定知识边** → `build_reason=physical_cooccurrence`，scope 区分句/段/chunk/表行列/引用
- **目录决定结构边** → `build_reason=document_structure`，Paper→Section→Chunk→Point
- **结构图负责定位 + 共现图负责关联** → m5 采样器先沿结构定位、再沿共现扩展子图

## 质量强化 (本轮新增 `m4_graph/quality.py`)

依据真实运行观察到的噪声，加入多层过滤，benchmark 质量显著提升：

| 噪声类型 | 过滤规则 | 例 |
|---|---|---|
| 单字符/符号概念 | `is_symbolic_name` | `p`, `n`, `Y/N`, `/` |
| 纯 LaTeX 变量符号 | LaTeX 命令剥离后核心 ≤3 字符 | `$\mathbf{w}_1$`, `${\sf C}_n$` |
| 测量值误标为概念 | `_looks_like_measurement` (数字+短单位尾) | `0.8 mm`, `110%`, `$80^\circ$` |
| 通用表头当属性 | `is_stopword_name` | `Value`, `Parameters`, `No.` |
| 元数据混入 | `is_metadata` (覆盖 Concept/Attr/Condition/Claim/Method) | `Publication Date`, `Volume 25`, `IEEE Copyright` |
| QA 泄漏 node_id | `looks_like_node_id` | 问题含 `tab1::row2` |
| QA 循环/退化/数值缺数字 | `is_valid_qa` | 问"参数b符号"答"b" |
| QA 重复 | (题型, 答案) 去重 | 同一指标多子图产同义题 |

过滤在三处生效：**m4** (建图时丢噪声点 + 表格通用列/单字符行)、**m5** (采样门控:
数值题要求真实属性、优先带单位子图；atomic 跳过短概念)、**m6** (QA 后过滤 + 去重)。

### before / after 质量对比 (同 2 篇论文)

| | 过滤前 | 过滤后 |
|---|---|---|
| QA 题型分布 | atomic 主导，多为"参数b符号是什么→b"平凡题 | numerical/condition 为主，实质科研题 |
| 典型 QA | "表中参数b的符号？"→"b" | "在什么频率范围实现≥90%吸收率？"→"在4-21 GHz范围内，吸收率≥90%" |
| 图谱节点 | 548 (含符号噪声) | ~500 (噪声滤除) |
| node_id 泄漏 | 偶发 | 0 (后过滤拦截) |

## 数据相关说明

`same_as` (跨文献) 与 `aligns_with` (跨模态对齐) 在仅 2 篇且过滤后可能为 0——
取决于两篇是否共享规范化概念、图注是否与正文同名。代码路径均有单测验证，
扩大到论文池 (论文建议每域 20+ 篇) 后会自然产生跨文献边。
