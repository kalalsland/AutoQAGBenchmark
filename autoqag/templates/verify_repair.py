"""验证与修复 prompt (论文创新四 / 五，§5.7 / §5.8)。

- SEMANTIC_VERIFY_PROMPT: 第四层语义验证 (忠实/完整/无外部幻觉/无过度泛化)，
  代码侧已做数值/单位/条件/证据的硬验证，这里补语义判断。
- REPAIR_PROMPT: violation 驱动的局部修复，输入结构化 violation，输出修复后的答案。
"""

SEMANTIC_VERIFY_PROMPT = """You are a strict scientific QA verifier (semantic layer). \
Judge whether the ANSWER is faithful to the EVIDENCE only.

Question: {question}
Answer: {answer}
Evidence: {evidence}

Check:
- Is every claim in the answer supported by the evidence? (no outside knowledge)
- Any over-generalization beyond the stated conditions/scope?
- Any hallucinated entity, number, or relation?

-Output STRICT JSON-
{{"faithful": true/false, "issues": ["..."], "severity": "minor|major|critical"}}
"""


# 用于验证器 comprehension-loss 打分 (改编自 GraphGen statement_judgement)
STATEMENT_JUDGEMENT_PROMPT = """Given the statement, answer only "yes" if it is fully \
supported and correct, otherwise "no".
Statement: {statement}
Answer (yes/no):"""


REPAIR_PROMPT = """You repair a scientific QA using a structured violation report. \
Apply ONLY the minimal local fix indicated; keep everything else unchanged. \
Stay grounded in the evidence; do not invent facts.

Question: {question}
Current answer: {answer}
Evidence: {evidence}

Violations (structured):
{violations}

-Repair rules by field-
- number: replace with the correct value from evidence
- unit: fix the unit or convert correctly
- condition: add back the missing limiting condition (avoid generalization)
- evidence: ground the answer in the cited evidence span
- path: complete the missing reasoning hop
- answer: if evidence is insufficient, change to "文中无法确定"

-Output STRICT JSON-
{{"answer": "...", "changed_fields": ["..."]}}
"""
