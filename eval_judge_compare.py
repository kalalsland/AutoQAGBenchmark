"""LLM-as-judge 评分：对 before/after 两组 QA 各按 5 维 1-5 打分并求平均。
仅评测用，不属于流水线。读取 qa.jsonl，用 qwen-plus 评分，输出对比表。"""
import json, os, statistics, sys
from autoqag.common.llm import LLMClient
from autoqag.ops.m8_verify.verifiers import _evidence_text
from autoqag.schema import QAItem
from autoqag.ops.m6_generate.json_utils import parse_json

JUDGE = """你是严格的科研QA质量评审。仅依据EVIDENCE判断，对下面QA在5个维度各打1-5分(整数)。
维度:
- faithfulness 答案是否忠实于证据、无幻觉外推
- grounding 答案是否可由所给证据充分支撑
- reasoning_depth 是否需要跨证据/多步推理(单点查找=低分,多跳整合=高分)
- specificity 是否保留数值/单位/条件等精确约束、不过度泛化
- overall 作为科研benchmark题目的总体质量
只输出JSON: {{"faithfulness":x,"grounding":x,"reasoning_depth":x,"specificity":x,"overall":x}}

QUESTION: {q}
ANSWER: {a}
EVIDENCE:
{e}
"""

def judge(path):
    rows=[json.loads(l) for l in open(path,encoding='utf-8')]
    items=[QAItem.from_dict(r) for r in rows]
    llm=LLMClient(model="qwen-plus", api_key=os.environ["AUTOQAG_API_KEY"],
                  base_url=os.environ["AUTOQAG_BASE_URL"], max_concurrency=6, json_mode=True)
    prompts=[JUDGE.format(q=q.question,a=q.answer,e=_evidence_text(q)[:1800]) for q in items]
    resp=llm.generate_batch(prompts)
    dims=["faithfulness","grounding","reasoning_depth","specificity","overall"]
    acc={d:[] for d in dims}
    for r in resp:
        o=parse_json(r) or {}
        for d in dims:
            v=o.get(d)
            if isinstance(v,(int,float)) and 1<=v<=5: acc[d].append(float(v))
    return {d:(round(statistics.mean(acc[d]),2) if acc[d] else None) for d in dims}, len(items)

b,nb=judge('outputs/cmp_before/qa.jsonl')
a,na=judge('outputs/cmp_after/qa.jsonl')
print(f"{'dimension':16} {'BEFORE(n=%d)'%nb:>14} {'AFTER(n=%d)'%na:>14}  delta")
for d in ["faithfulness","grounding","reasoning_depth","specificity","overall"]:
    dl=round(a[d]-b[d],2) if (a[d] and b[d]) else None
    print(f"{d:16} {b[d]:>14} {a[d]:>14}  {('+' if dl and dl>0 else '')}{dl}")
