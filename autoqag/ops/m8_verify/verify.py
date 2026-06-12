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
from autoqag.ops.m8_verify.verifiers import (
    CHECKERS,
    _evidence_text,
    answer_recoverable,
    is_refusal,
    make_masking_violation,
    masked_evidence_text,
)
from autoqag.registry import STAGES
from autoqag.schema import QAItem, Violation, VerifyLayer
from autoqag.templates.verify_repair import SEMANTIC_VERIFY_PROMPT

# 遮蔽测试重答提示 (§4.1)：只许用给定 (已删除关键证据的) 证据作答，
# 无法确定须明确拒答——若仍答出原答案则暴露伪多跳/捷径。
_MASKING_PROMPT = (
    "仅根据下面提供的证据回答问题；若证据不足以确定答案，请只回答“无法从文中确定”。\n"
    "不得使用证据之外的任何知识。\n\n"
    "问题：{question}\n\n"
    "证据：\n{evidence}\n\n"
    "答案："
)


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
        masking_check = self.params.get("masking_check", False)

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

        # 遮蔽层批量调用 (可选；行为级反伪多跳 §4.1)：
        # 对每条带 masking_spec 的 QA，删除 drop_operand 证据后让强模型重答，
        # 仍能复现答案则判伪多跳。trials 收集 (qa_index, kind, dropped, prompt)。
        masking_trials: List[Dict[str, Any]] = []
        masking_results: List[str] = []
        if masking_check:
            for i, q in enumerate(qa_items):
                if is_refusal(q.answer) or not q.masking_spec:
                    continue
                for kind in ("drop_operand", "drop_cross_chunk"):
                    dropped = q.masking_spec.get(kind) or []
                    if not dropped:
                        continue
                    masked_ev = masked_evidence_text(q, dropped)
                    prompt = _MASKING_PROMPT.format(question=q.question, evidence=masked_ev)
                    masking_trials.append({"qa_index": i, "kind": kind, "dropped": dropped, "prompt": prompt})
            if masking_trials:
                self.log("遮蔽测试重答: %d 次", len(masking_trials))
                masking_results = ctx.llm.generate_batch([t["prompt"] for t in masking_trials])

        # qa_index -> 该题的遮蔽违规列表
        masking_viol_by_idx: Dict[int, List[Violation]] = {}
        for t, ans in zip(masking_trials, masking_results):
            if answer_recoverable(qa_items[t["qa_index"]].answer, ans or ""):
                masking_viol_by_idx.setdefault(t["qa_index"], []).append(
                    make_masking_violation(qa_items[t["qa_index"]], t["dropped"], t["kind"])
                )

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

            # 遮蔽层违规 (行为级反伪多跳)
            if masking_check and i in masking_viol_by_idx:
                violations.extend(masking_viol_by_idx[i])

            qa.validator_result = {
                "passed": len(violations) == 0,
                "n_violations": len(violations),
                "layers_checked": enabled
                + (["semantic"] if semantic_check else [])
                + (["masking"] if masking_check else []),
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
