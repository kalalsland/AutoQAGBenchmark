"""模块六：负样本与错误扰动构造 (论文 §5.6)。

读取 qa.jsonl + 图谱，为每条 QA 构造若干 corrupted 变体 (10 类错误类型)，
输出 corrupted_qa.jsonl，并派生：
  - corpus/verifier.jsonl   : (qa, label=good/bad, error_type) 验证器训练样本
  - corpus/preference.jsonl : (chosen=原始, rejected=corrupted) 偏好对
不调用 LLM (纯图/字符串扰动，可控且零成本)。
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List

from autoqag.common.graph_store import GraphStore
from autoqag.common.io import read_jsonl_list, write_jsonl
from autoqag.ops.base import BaseStage, PipelineContext
from autoqag.ops.m7_corrupt.corruptors import ALL_CORRUPTORS
from autoqag.registry import STAGES
from autoqag.schema import NodeType, QAItem


@STAGES.register_module("corrupt")
class CorruptStage(BaseStage):
    declared_inputs = ["qa.jsonl"]
    declared_outputs = ["corrupted_qa.jsonl", "corpus/verifier.jsonl", "corpus/preference.jsonl"]

    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        rows = read_jsonl_list(ctx.path("qa.jsonl"))
        if not rows:
            self.log("qa.jsonl 为空，先运行 generate")
            return {"corrupted": 0}

        qa_items = [QAItem.from_dict(r) for r in rows]
        per_qa = int(self.params.get("per_qa", 2))
        seed = int(self.params.get("seed", 42))
        rng = random.Random(seed)

        pools = self._build_pools(ctx)

        corrupted: List[Dict[str, Any]] = []
        verifier: List[Dict[str, Any]] = []
        preference: List[Dict[str, Any]] = []

        for qa in qa_items:
            # 正样本（good）
            verifier.append(self._verifier_sample(qa, label="good", error_type=""))

            corruptors = ALL_CORRUPTORS[:]
            rng.shuffle(corruptors)
            made = 0
            for fn in corruptors:
                if made >= per_qa:
                    break
                result = fn(qa, pools, rng)
                if not result:
                    continue
                new_answer, etype = result
                cq = QAItem.from_dict(qa.to_dict())
                cq.qid = f"{qa.qid}__{etype}"
                cq.answer = new_answer
                cq.is_corrupted = True
                cq.error_type = etype
                corrupted.append(cq.to_dict())
                verifier.append(self._verifier_sample(cq, label="bad", error_type=etype))
                preference.append(
                    {
                        "type": "preference_pair",
                        "question": qa.question,
                        "chosen": qa.answer,
                        "rejected": new_answer,
                        "error_type": etype,
                        "evidence": qa.evidence_spans,
                    }
                )
                made += 1

        n_c = write_jsonl(ctx.path("corrupted_qa.jsonl"), corrupted)
        n_v = write_jsonl(ctx.path("corpus/verifier.jsonl"), verifier)
        n_p = write_jsonl(ctx.path("corpus/preference.jsonl"), preference)

        by_type: Dict[str, int] = defaultdict(int)
        for c in corrupted:
            by_type[c["error_type"]] += 1
        self.log("corrupted=%d verifier=%d preference=%d %s", n_c, n_v, n_p, dict(by_type))
        return {"corrupted": n_c, "verifier": n_v, "preference": n_p}

    def _build_pools(self, ctx: PipelineContext) -> Dict[str, List[str]]:
        """从图谱抽取同类型节点池，供实体/数值替换。"""
        pools: Dict[str, List[str]] = {"concept": [], "value": [], "unit": []}
        if not ctx.artifact_exists("nodes.jsonl"):
            return pools
        store = GraphStore.load_jsonl(ctx.path("nodes.jsonl"), ctx.path("edges.jsonl"))
        for _, d in store.all_nodes():
            nt = d.get("node_type")
            content = (d.get("normalized_content") or d.get("content") or "").strip()
            if nt == NodeType.CONCEPT.value:
                pools["concept"].append(content)
            elif nt == NodeType.VALUE.value:
                pools["value"].append(content)
            elif nt == NodeType.UNIT.value:
                pools["unit"].append(content)
        return pools

    @staticmethod
    def _verifier_sample(qa: QAItem, label: str, error_type: str) -> Dict[str, Any]:
        return {
            "type": "verifier_training",
            "question": qa.question,
            "answer": qa.answer,
            "evidence": qa.evidence_spans,
            "label": label,  # good / bad
            "error_type": error_type,
        }
