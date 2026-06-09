"""模块四(下)·语义规划层：评分引导的问题子图规划与虚拟逻辑补全。

实现 语义规划层方法论.pdf / 语义层子图构建.pdf 的核心创新，作为 `sample` 阶段的
升级替代 (stage 名 `semantic_plan`)。与 `sample` 接口一致：
  输入  nodes.jsonl + edges.jsonl
  输出  question_plans.jsonl (字段向后兼容 m6_generate，可直接复用下游)

与 `sample` 的差异：不再做纯物理模板采样，而是按题型 role schema 驱动:
  问题逻辑规划 → 种子初始化(SeedScore) → 评分引导扩展 → 角色分配 →
  虚拟逻辑补全(Ωq) → 逻辑充分性/双重多跳验证 → QuestionPlan。

虚拟边写入 plan.semantic_overlay_edges (规划用)；required_nodes / evidence 仍为
G0 物理节点，保证 Benchmark 可追溯。长期记忆落盘 semantic_memory.json。
"""

from __future__ import annotations

from typing import Any, Dict, List

from autoqag.common.graph_store import GraphStore
from autoqag.common.io import write_jsonl
from autoqag.ops.base import BaseStage, PipelineContext
from autoqag.ops.m5_sample.sample import _GraphView, _node_ok
from autoqag.ops.m5_sample.semantic import roles as R
from autoqag.ops.m5_sample.semantic import seed as seedmod
from autoqag.ops.m5_sample.semantic.memory import SemanticMemory
from autoqag.ops.m5_sample.semantic.planner import OverlayPlanner
from autoqag.registry import STAGES
from autoqag.schema import QuestionPlan, QuestionType

# 该层支持的题型 (含方法论新增的 mechanism / cross_paper)
_DEFAULT_TYPES = [
    "atomic",
    "numerical",
    "condition",
    "comparative",
    "mechanism",
    "table",
    "formula",
    "multi_hop",
    "cross_paper",
    "summary",
]


@STAGES.register_module("semantic_plan")
class SemanticPlanStage(BaseStage):
    declared_inputs = ["nodes.jsonl", "edges.jsonl"]
    declared_outputs = ["question_plans.jsonl", "semantic_memory.json"]

    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        store = GraphStore.load_jsonl(ctx.path("nodes.jsonl"), ctx.path("edges.jsonl"))
        if store.node_count() == 0:
            self.log("图谱为空，先运行 graph")
            return {"plans": 0}

        view = _GraphView(store)
        per_type = int(self.params.get("per_type", 12))
        types = self.params.get("types", _DEFAULT_TYPES)
        domain = ctx.global_params.get("domain", "")
        tau = float(self.params.get("tau", 0.0))
        max_expand = int(self.params.get("max_expand", 12))
        seed_pool = int(self.params.get("seed_pool", 0)) or per_type * 4

        # 长期记忆 (跨运行沉淀；§3.9)
        mem_path = ctx.path("semantic_memory.json")
        memory = SemanticMemory.load(mem_path)
        seed_boost = memory.seed_boost()

        planner = OverlayPlanner(
            view, domain=domain, memory=memory, node_ok=_node_ok,
            tau=tau, max_expand=max_expand,
            use_score_guided=bool(self.params.get("use_score_guided", True)),
            use_binding=bool(self.params.get("use_binding", True)),
            use_overlay=bool(self.params.get("use_overlay", True)),
            use_dual_multihop=bool(self.params.get("use_dual_multihop", True)),
            use_sufficiency=bool(self.params.get("use_sufficiency", True)),
        )

        plans: List[QuestionPlan] = []
        counter = 0
        stats: Dict[str, int] = {}
        attempted: Dict[str, int] = {}
        for qtype in types:
            seeds = seedmod.rank_seeds(
                view, R.seed_types(qtype), top_k=seed_pool,
                node_ok=_node_ok, memory_boost=seed_boost,
            )
            made = 0
            tried = 0
            for s in seeds:
                if made >= per_type:
                    break
                tried += 1
                counter += 1
                plan = planner.plan_one(qtype, s, f"sp{counter:06d}")
                if plan is None:
                    continue
                plans.append(plan)
                made += 1
            stats[qtype] = made
            attempted[qtype] = tried

        # 去重 (同题型 + 相同必需节点集合)
        plans = _dedup(plans)

        n = write_jsonl(ctx.path("question_plans.jsonl"), [p.to_dict() for p in plans])
        memory.save(mem_path)
        self.log("语义规划生成 %d 个 question plan: %s (尝试 %s)", n, stats, attempted)
        return {"plans": n, **{f"plan_{k}": v for k, v in stats.items()}}


def _dedup(plans: List[QuestionPlan]) -> List[QuestionPlan]:
    seen = set()
    out: List[QuestionPlan] = []
    for p in plans:
        key = (p.question_type, tuple(sorted(p.required_nodes)))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
