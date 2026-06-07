"""模块七：类型特定约束验证 (论文创新四 §5.7)。

读取 qa.jsonl，对每条 QA 跑四层验证 (MVP：数值/单位/条件/证据，外加可选 LLM 语义层)，
不通过的产出结构化 Violation (论文创新五格式) → violations.jsonl，
并把验证结论写回 qa.jsonl 的 validator_result，区分通过池 / 需修复池。
"""

from __future__ import annotations

from typing import Any, Dict, List

from autoqag.common.io import read_jsonl_list, write_jsonl
from autoqag.ops.base import BaseStage, PipelineContext
from autoqag.ops.m6_generate.json_utils import parse_json
from autoqag.ops.m8_verify.verifiers import CHECKERS, _evidence_text, is_refusal
from autoqag.registry import STAGES
from autoqag.schema import QAItem, Violation, VerifyLayer
from autoqag.templates.verify_repair import SEMANTIC_VERIFY_PROMPT


@STAGES.register_module("verify")
class VerifyStage(BaseStage):
    declared_inputs = ["qa.jsonl"]
    declared_outputs = ["violations.jsonl", "qa.jsonl"]

    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        # 支持验证主 QA 或被修复后的 QA (target 参数)
        target = self.params.get("target", "qa.jsonl")
        rows = read_jsonl_list(ctx.path(target))
        if not rows:
            self.log("%s 为空，先运行 generate", target)
            return {"verified": 0}

        qa_items = [QAItem.from_dict(r) for r in rows]
        enabled = [
            name
            for name in CHECKERS
            if self.params.get(name, True)
        ]
        semantic_check = self.params.get("semantic_check", False)

        all_violations: List[Violation] = []
        passed = failed = 0

        # 语义层批量调用 (可选)
        semantic_results: List[Dict[str, Any]] = []
        if semantic_check:
            prompts = [
                SEMANTIC_VERIFY_PROMPT.format(
                    question=q.question,
                    answer=q.answer,
                    evidence=_evidence_text(q),
                )
                for q in qa_items
            ]
            self.log("LLM 语义验证: %d 条", len(prompts))
            resp = ctx.llm.generate_batch(prompts)
            semantic_results = [parse_json(r) or {} for r in resp]

        for i, qa in enumerate(qa_items):
            # 合法拒答 (证据不足) 直接通过：约束层不适用，避免误入 human_review
            if is_refusal(qa.answer):
                qa.validator_result = {
                    "passed": True,
                    "n_violations": 0,
                    "layers_checked": enabled,
                    "violations": [],
                    "refusal": True,
                }
                passed += 1
                continue

            evidence = _evidence_text(qa)
            violations: List[Violation] = []
            for name in enabled:
                violations.extend(CHECKERS[name](qa, evidence))

            if semantic_check and i < len(semantic_results):
                sr = semantic_results[i]
                if sr and sr.get("faithful") is False:
                    violations.append(
                        Violation(
                            qid=qa.qid,
                            layer=VerifyLayer.SEMANTIC.value,
                            field="answer",
                            expected="faithful to evidence",
                            actual="; ".join(sr.get("issues", [])) or "unfaithful",
                            severity=sr.get("severity", "major"),
                            repair_hint="增强答案忠实性，去除外部幻觉/过度泛化",
                        )
                    )

            qa.validator_result = {
                "passed": len(violations) == 0,
                "n_violations": len(violations),
                "layers_checked": enabled + (["semantic"] if semantic_check else []),
                "violations": [v.to_dict() for v in violations],
            }
            if violations:
                failed += 1
                all_violations.extend(violations)
            else:
                passed += 1

        n_viol = write_jsonl(
            ctx.path("violations.jsonl"), [v.to_dict() for v in all_violations]
        )
        write_jsonl(ctx.path(target), [q.to_dict() for q in qa_items])

        self.log("验证: passed=%d failed=%d violations=%d", passed, failed, n_viol)
        return {"passed": passed, "failed": failed, "violations": n_viol}
