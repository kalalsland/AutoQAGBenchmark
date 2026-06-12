"""模块五：QA 与高级训练语料生成 (论文 §5.5 + 创新六)。

输入 question_plans.jsonl，对每个 plan 打包证据 → LLM 按约束生成 QA → 解析为 QAItem。
同时产出可直接从 QA+图谱派生的高级训练语料 (论文创新六)：
  - evidence-grounded instruction
  - graph reasoning trace
  - RAG grounding sample
  - refusal / insufficient-evidence sample
(verifier / repair / preference 样本分别由 m7 / m9 产出，m10 汇总。)

生成原则 (论文 §5.5)：答案绑定 evidence span；数值绑 Value+Unit；条件保留 Condition；
多跳输出 evidence_path；证据不足产 refusal。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Tuple

from autoqag.common.io import read_jsonl_list, write_jsonl
from autoqag.common.llm import _run_sync
from autoqag.common.logging import logger
from autoqag.ops.base import BaseStage, PipelineContext
from autoqag.ops.m6_generate.json_utils import parse_json
from autoqag.ops.m4_graph.quality import is_valid_qa
from autoqag.registry import STAGES
from autoqag.schema import QAItem, QuestionPlan
from autoqag.templates.qa_generation import QA_GENERATION_PROMPT


@STAGES.register_module("generate")
class GenerateStage(BaseStage):
    declared_inputs = ["question_plans.jsonl"]
    declared_outputs = ["qa.jsonl", "corpus/"]

    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        rows = read_jsonl_list(ctx.path("question_plans.jsonl"))
        if not rows:
            self.log("question_plans.jsonl 为空，先运行 sample")
            return {"qa": 0}

        plans = [QuestionPlan.from_dict(r) for r in rows]
        max_plans = self.params.get("max_plans")
        if max_plans and len(plans) > int(max_plans):
            # 按题型轮转截断：plans 按题型分段排列，直接 [:N] 会把排在后面的
            # multi_hop/summary 全部截掉，破坏 benchmark 多样性
            plans = _interleave_by_type(plans, int(max_plans))

        prompts = [self._build_prompt(p) for p in plans]
        self.log("LLM 生成 QA: %d 个 plan", len(prompts))
        if self.params.get("record_timing"):
            responses, durations = _timed_generate_batch(ctx.llm, prompts)
            self._write_timing_report(
                ctx, [p.question_type for p in plans], durations
            )
        else:
            responses = ctx.llm.generate_batch(prompts)

        qa_items: List[QAItem] = []
        refusals: List[Dict[str, Any]] = []
        dropped: Dict[str, int] = {}
        seen: set = set()  # (question_type, 归一化答案) 去重
        for plan, resp in zip(plans, responses):
            data = parse_json(resp)
            if not data or not data.get("question"):
                dropped["no_json"] = dropped.get("no_json", 0) + 1
                continue
            insufficient = bool(data.get("insufficient"))
            question = data.get("question", "")
            answer = data.get("answer", "")
            # 后过滤：丢弃泄漏 node_id / 退化 / 循环 / 数值缺数字的 QA (强化质量)
            src_names = [
                (e.get("content") or "") for e in plan.evidence_spans
            ]
            ok, reason = is_valid_qa(question, answer, plan.question_type, src_names)
            if not ok:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            # 去重：同题型同答案视为重复 (避免 sampler 多子图产同义题)
            key = (plan.question_type, " ".join(answer.lower().split())[:80])
            if key in seen:
                dropped["duplicate"] = dropped.get("duplicate", 0) + 1
                continue
            seen.add(key)
            item = QAItem(
                qid=plan.qid,
                question=question,
                answer=answer,
                question_type=plan.question_type,
                difficulty=plan.difficulty,
                evidence_spans=plan.evidence_spans,
                evidence_path=data.get("evidence_path", []) or plan.required_nodes,
                source_nodes=data.get("evidence_node_ids", []) or plan.required_nodes,
                source_edges=plan.required_edges,
                constraints=data.get("constraints", {}) or plan.constraints,
                domain=plan.domain,
                paper_id_list=plan.paper_id_list,
                masking_spec=plan.masking_spec,
                inference_ops=plan.inference_ops,
            )
            qa_items.append(item)
            if insufficient:
                refusals.append(self._refusal_sample(item))

        if dropped:
            self.log("QA 质量过滤丢弃: %s", dropped)

        n_qa = write_jsonl(ctx.path("qa.jsonl"), [q.to_dict() for q in qa_items])

        # 高级训练语料 (可从 QA+图谱直接派生的几类)
        corpus_dir = "corpus"
        n_instr = write_jsonl(
            ctx.path(f"{corpus_dir}/instruction.jsonl"),
            [self._instruction_sample(q) for q in qa_items],
        )
        n_trace = write_jsonl(
            ctx.path(f"{corpus_dir}/graph_trace.jsonl"),
            [self._graph_trace_sample(q) for q in qa_items],
        )
        n_rag = write_jsonl(
            ctx.path(f"{corpus_dir}/rag_grounding.jsonl"),
            [self._rag_sample(q) for q in qa_items],
        )
        n_ref = write_jsonl(ctx.path(f"{corpus_dir}/refusal.jsonl"), refusals)

        self.log(
            "QA=%d | corpus: instr=%d trace=%d rag=%d refusal=%d",
            n_qa,
            n_instr,
            n_trace,
            n_rag,
            n_ref,
        )
        return {
            "qa": n_qa,
            "instruction": n_instr,
            "graph_trace": n_trace,
            "rag_grounding": n_rag,
            "refusal": n_ref,
        }

    # ----- prompt -----
    @staticmethod
    def _build_prompt(plan: QuestionPlan) -> str:
        return QA_GENERATION_PROMPT.format(
            question_type=plan.question_type,
            difficulty=plan.difficulty,
            expected_answer_form=plan.expected_answer_form,
            forbidden_generalization=plan.forbidden_generalization,
            generation_instruction=plan.generation_instruction,
            evidence_block=_format_evidence(plan.evidence_spans),
        )

    # ----- 训练样本派生 -----
    @staticmethod
    def _instruction_sample(q: QAItem) -> Dict[str, Any]:
        return {
            "type": "evidence_grounded_instruction",
            "instruction": q.question,
            "input": _format_evidence(q.evidence_spans),
            "output": q.answer,
            "domain": q.domain,
            "difficulty": q.difficulty,
            "evidence": q.evidence_spans,
        }

    @staticmethod
    def _graph_trace_sample(q: QAItem) -> Dict[str, Any]:
        trace = " -> ".join(q.evidence_path) if q.evidence_path else ""
        return {
            "type": "graph_reasoning_trace",
            "question": q.question,
            "answer": q.answer,
            "reasoning_trace": trace,
            "source_edges": q.source_edges,
            "question_type": q.question_type,
        }

    @staticmethod
    def _rag_sample(q: QAItem) -> Dict[str, Any]:
        return {
            "type": "rag_grounding",
            "question": q.question,
            "contexts": [e.get("content", "") for e in q.evidence_spans],
            "answer": q.answer,
            "gold_evidence_ids": q.source_nodes,
        }

    @staticmethod
    def _refusal_sample(q: QAItem) -> Dict[str, Any]:
        return {
            "type": "refusal",
            "instruction": q.question,
            "input": _format_evidence(q.evidence_spans),
            "output": q.answer or "文中无法确定",
            "reason": "insufficient_evidence",
        }

    # ----- 分题型生成计时 -----
    def _write_timing_report(
        self,
        ctx: PipelineContext,
        types: List[str],
        durations: List[float],
    ) -> None:
        """按题型汇总单次 LLM 调用墙钟耗时，写 generate_timing.json。

        口径：每个 plan 的单次 agenerate 调用 (提交→返回) 的墙钟时长，
        受 max_concurrency 并发影响。per_type 给出各题型平均/总/计数。
        """
        per_type: Dict[str, List[float]] = {}
        for t, d in zip(types, durations):
            per_type.setdefault(t, []).append(d)

        type_stats: Dict[str, Dict[str, float]] = {}
        for t, ds in per_type.items():
            ds_sorted = sorted(ds)
            type_stats[t] = {
                "count": len(ds),
                "avg_sec": round(sum(ds) / len(ds), 3),
                "min_sec": round(ds_sorted[0], 3),
                "max_sec": round(ds_sorted[-1], 3),
                "total_sec": round(sum(ds), 3),
            }
        report = {
            "max_concurrency": ctx.llm.max_concurrency,
            "model": ctx.llm.model,
            "n_calls": len(durations),
            "wall_sum_sec": round(sum(durations), 3),
            "avg_per_call_sec": round(sum(durations) / len(durations), 3)
            if durations
            else 0.0,
            "by_type": type_stats,
        }
        out = ctx.path("generate_timing.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.log("分题型生成计时已写入 generate_timing.json")
        for t, s in sorted(type_stats.items()):
            self.log(
                "  [%s] n=%d avg=%.2fs (min=%.2f max=%.2f)",
                t,
                s["count"],
                s["avg_sec"],
                s["min_sec"],
                s["max_sec"],
            )


def _timed_generate_batch(
    llm, prompts: List[str]
) -> Tuple[List[str], List[float]]:
    """并发跑 batch，同时记录每次调用的墙钟耗时 (与 generate_batch 等价的并发模型)。"""
    n = len(prompts)
    durations: List[float] = [0.0] * n

    async def _one(i: int, text: str) -> str:
        t0 = time.perf_counter()
        try:
            r = await llm.agenerate(text)
        except Exception as e:  # 与 generate_batch 一致：失败计空串
            logger.error("LLM batch item failed: %s", e)
            r = ""
        durations[i] = time.perf_counter() - t0
        return r

    async def _all():
        return await asyncio.gather(
            *[_one(i, t) for i, t in enumerate(prompts)], return_exceptions=True
        )

    results = _run_sync(_all())
    out: List[str] = []
    for r in results:
        out.append("" if isinstance(r, Exception) else r)
    return out, durations



def _format_evidence(spans: List[Dict[str, Any]]) -> str:
    lines = []
    for e in spans:
        addr = e.get("address", {})
        loc = f"{addr.get('paper_id','')}/{addr.get('section_path','')}/{addr.get('chunk_id','')}"
        lines.append(f"[{e.get('node_id','')}] ({loc}) {e.get('content','')}")
    return "\n".join(lines)


def _interleave_by_type(plans: List[QuestionPlan], limit: int) -> List[QuestionPlan]:
    """按题型轮转抽取前 limit 个 plan，截断后各题型尽量均衡。"""
    buckets: Dict[str, List[QuestionPlan]] = {}
    for p in plans:
        buckets.setdefault(p.question_type, []).append(p)
    out: List[QuestionPlan] = []
    queues = list(buckets.values())
    while len(out) < limit and queues:
        queues = [q for q in queues if q]
        if not queues:
            break
        for q in queues:
            if len(out) >= limit:
                break
            out.append(q.pop(0))
    return out
