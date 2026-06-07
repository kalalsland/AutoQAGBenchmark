# 产物字段规格 (论文 §5.9)

所有 artifact 为 jsonl (每行一个 JSON 对象)。下列字段定义见 `autoqag/schema.py`。

## documents.jsonl — DocumentMeta (论文 §5.1)
`paper_id, title, authors, year, venue, domain, subdomain, source_path, license_status,`
`pdf_quality_level, has_tables, has_figures, has_equations, parsing_status, graph_status, qa_status, annotation_status`

## dir/<paper_id>.json — Document Intermediate Representation (论文 §5.2)
```
{ paper_id, metadata, parse_tool, parse_quality:{reading_order_score, table_quality_score,
  formula_quality_score, caption_quality_score, ocr_confidence, overall_quality},
  sections:[{section_id, section_path, title, level,
             chunks:[{chunk_id, text, page_range, bbox, sentence_list,
                      figure_refs, table_refs, equation_refs}]}],
  figures:[{figure_id,label,caption,footnote,img_path,page,bbox,section_path}],
  tables:[{table_id,label,caption,footnote,html,img_path,page,bbox,section_path}],
  equations:[{equation_id,label,latex,text_format,img_path,page,bbox,section_path}],
  references:[] }
```

## evidence_blocks.jsonl — EvidenceBlock
`block_id, modality(text/table/formula/caption/figure/reference), content,`
`address:{paper_id,section_path,chunk_id,sentence_id,paragraph_id,page,span,bbox},`
`confidence, figure_refs, table_refs, equation_refs, caption, extra`

## nodes.jsonl — PointNode (论文 §5.3)
`node_id, node_type(14 类 NodeType), content, normalized_content, address, modality,`
`confidence, domain_schema_tag, canonical_id`

## edges.jsonl — Edge (论文 §5.3 + 图谱构建.pdf)
`source, target, edge_type(has_attribute/has_value/has_unit/under_condition/supports/`
`derived_from/contains/references/describes/compares/same_as/co_occurs_with),`
`build_reason(physical_cooccurrence/document_structure/cross_modal_reference/cross_paper),`
`cooccur_scope, paper_id, section_path, chunk_id, evidence_span, weight, confidence`

## question_plans.jsonl — QuestionPlan (论文创新三 §4)
`qid, domain, question_type(8 类), difficulty(L1-L4), target_subgraph, required_nodes,`
`required_edges, evidence_spans, constraints:{number,unit,condition,formula,table},`
`expected_answer_form, forbidden_generalization, generation_instruction, paper_id_list`

## qa.jsonl — QAItem (论文 §5.5)
`qid, question, answer, question_type, difficulty, evidence_spans, evidence_path,`
`source_nodes, source_edges, constraints, domain, paper_id_list, validator_result,`
`is_corrupted, error_type`

## violations.jsonl — Violation (论文创新五)
`qid, layer(constraint/graph/evidence/semantic), field(number/unit/condition/entity/`
`evidence/path/answer), expected, actual, source_node, source_edge, source_address,`
`severity(minor/major/critical), repair_hint`

## corpus/*.jsonl — 高级训练语料 (论文创新六 §5.5)
| 文件 | 类型 | 产出模块 |
|---|---|---|
| instruction.jsonl | evidence-grounded instruction | m6 |
| graph_trace.jsonl | graph reasoning trace | m6 |
| rag_grounding.jsonl | RAG grounding sample | m6 |
| refusal.jsonl | refusal / insufficient evidence | m6 |
| verifier.jsonl | verifier training (good/bad + error_type) | m7 |
| preference.jsonl | preference pair (chosen/rejected) | m7 |
| repair.jsonl | repair trace (wrong→violation→repaired) | m9 |
