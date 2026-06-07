"""模块八：Violation-driven Self-Repair (论文创新五 §5.8)。

读取已验证 qa.jsonl，对未通过的 QA：用结构化 violation 报告驱动局部修复 →
重验证，最多 max_rounds 轮。修复成功进入终池，否则进入 low-quality / human-review 池。
记录 wrong→violation→repaired 修复轨迹 (论文：可作训练数据) → corpus/repair.jsonl。

与"自然语言 self-refine"的关键差异：修复 prompt 输入的是带 field/expected/actual/
source_address 的结构化 violation，提供明确外部锚点，避免自洽幻觉 (论文创新五)。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from autoqag.common.io import read_jsonl_list, write_jsonl
from autoqag.ops.base import BaseStage, PipelineContext
from autoqag.ops.m6_generate.json_utils import parse_json
from autoqag.ops.m8_verify.verifiers import CHECKERS, _evidence_text
from autoqag.registry import STAGES
from autoqag.schema import QAItem
from autoqag.templates.verify_repair import REPAIR_PROMPT


@STAGES.register_module("repair")
class RepairStage(BaseStage):
    declared_inputs = ["qa.jsonl"]
    declared_outputs = ["qa.jsonl", "corpus/repair.jsonl"]

    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        rows = read_jsonl_list(ctx.path("qa.jsonl"))
        if not rows:
            self.log("qa.jsonl 为空，先运行 verify")
            return {"repaired": 0}

        qa_items = [QAItem.from_dict(r) for r in rows]
        max_rounds = int(self.params.get("max_rounds", 3))
        enabled = [n for n in CHECKERS if self.params.get(n, True)]

        # 仅修复未通过的
        pending_idx = [
            i
            for i, q in enumerate(qa_items)
            if not q.validator_result.get("passed", True)
        ]
        if not pending_idx:
            self.log("无需修复 (全部已通过)")
            return {"repaired": 0, "pending": 0}

        self.log("待修复 QA: %d 条，最多 %d 轮", len(pending_idx), max_rounds)
        traces: List[Dict[str, Any]] = []
        # 每条 QA 的修复轨迹累积
        trace_map: Dict[int, Dict[str, Any]] = {
            i: {
                "qid": qa_items[i].qid,
                "question": qa_items[i].question,
                "wrong_answer": qa_items[i].answer,
                "steps": [],
            }
            for i in pending_idx
        }

        active = list(pending_idx)
        for rnd in range(max_rounds):
            if not active:
                break
            prompts = []
            for i in active:
                qa = qa_items[i]
                viol = qa.validator_result.get("violations", [])
                prompts.append(
                    REPAIR_PROMPT.format(
                        question=qa.question,
                        answer=qa.answer,
                        evidence=_evidence_text(qa),
                        violations=json.dumps(viol, ensure_ascii=False, indent=2),
                    )
                )
            self.log("修复第 %d 轮: %d 条", rnd + 1, len(prompts))
            responses = ctx.llm.generate_batch(prompts)

            still_active: List[int] = []
            for i, resp in zip(active, responses):
                qa = qa_items[i]
                old_answer = qa.answer
                data = parse_json(resp)
                new_answer = (data or {}).get("answer", "").strip() or old_answer
                changed = (data or {}).get("changed_fields", [])

                qa.answer = new_answer
                evidence = _evidence_text(qa)
                new_viol = []
                for name in enabled:
                    new_viol.extend(CHECKERS[name](qa, evidence))
                qa.validator_result = {
                    "passed": len(new_viol) == 0,
                    "n_violations": len(new_viol),
                    "violations": [v.to_dict() for v in new_viol],
                    "repaired": True,
                    "repair_rounds": rnd + 1,
                }
                trace_map[i]["steps"].append(
                    {
                        "round": rnd + 1,
                        "before": old_answer,
                        "after": new_answer,
                        "changed_fields": changed,
                        "remaining_violations": len(new_viol),
                    }
                )
                if new_viol:
                    still_active.append(i)
            active = still_active

        # 汇总结果与池
        repaired = 0
        unresolved = 0
        for i in pending_idx:
            qa = qa_items[i]
            tr = trace_map[i]
            tr["final_answer"] = qa.answer
            tr["success"] = qa.validator_result.get("passed", False)
            traces.append(tr)
            if qa.validator_result.get("passed"):
                repaired += 1
                qa.validator_result["pool"] = "final"
            else:
                unresolved += 1
                qa.validator_result["pool"] = "human_review"

        write_jsonl(ctx.path("qa.jsonl"), [q.to_dict() for q in qa_items])
        n_tr = write_jsonl(ctx.path("corpus/repair.jsonl"), traces)

        self.log(
            "修复完成: success=%d unresolved=%d traces=%d", repaired, unresolved, n_tr
        )
        return {
            "repaired": repaired,
            "unresolved": unresolved,
            "repair_traces": n_tr,
        }
