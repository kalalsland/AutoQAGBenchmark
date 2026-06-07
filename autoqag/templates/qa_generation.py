"""QA 生成 prompt (论文 §5.5)。

输入 Question Plan + 打包好的 evidence，要求 LLM 生成严格遵守约束的 QA，
并返回 JSON，便于解析为 QAItem。生成原则见论文 §5.5：
答案绑定 evidence span、数值绑 Value+Unit、条件保留 Condition、多跳输出 evidence_path、
证据不足产 refusal。
"""

QA_GENERATION_PROMPT = """You generate ONE verifiable QA pair from a scientific paper, \
strictly grounded in the provided evidence subgraph. Do NOT use outside knowledge.

-Question Plan-
question_type: {question_type}
difficulty: {difficulty}
expected_answer_form: {expected_answer_form}
forbidden_generalization: {forbidden_generalization}
instruction: {generation_instruction}

-Evidence (each item has node_id, type, content, address)-
{evidence_block}

-Rules-
1. The answer MUST be supported by the evidence above; cite the supporting node_ids.
2. Numerical answers MUST keep the exact value AND its unit from the evidence.
3. Condition-bound answers MUST explicitly keep the condition (do not generalize).
4. For multi_hop, list the ordered evidence_path of node_ids.
5. If the evidence is insufficient to answer, set answer to "文中无法确定" / "Cannot be determined from the text" and set insufficient=true.
6. Write a SELF-CONTAINED, natural question a researcher would ask. Do NOT mention internal identifiers, node ids, coordinates (e.g. "tab1::row2", "r1c0"), or phrases like "according to the node/subgraph". Refer to entities by their real names.
7. The question must be answerable and non-trivial; avoid questions whose answer merely repeats a symbol or single letter.

-Output STRICT JSON-
{{
  "question": "...",
  "answer": "...",
  "evidence_node_ids": ["..."],
  "evidence_path": ["..."],
  "constraints": {{"number": [], "unit": [], "condition": []}},
  "insufficient": false
}}
"""


# 高级训练语料：把已验证 QA 改写为 evidence-grounded instruction (论文创新六)
INSTRUCTION_REWRITE_PROMPT = """Rewrite the following verified QA into an instruction-tuning sample. \
Keep the answer faithful to the evidence; include the evidence as context.

Question: {question}
Answer: {answer}
Evidence: {evidence}

-Output STRICT JSON-
{{"instruction": "...", "input": "...", "output": "..."}}
"""
