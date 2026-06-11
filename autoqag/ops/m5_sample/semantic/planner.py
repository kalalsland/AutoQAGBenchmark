"""评分引导的问题子图规划编排器 (语义规划层方法论.pdf §2.3 总体流程)。

OverlayPlanner 把一个 (题型 T, 难度 D, 种子) 三元组规划为一个 QuestionPlan：

  1. Question Logic Planning   —— 确定题型 role schema 与必需约束
  2. Seed Initialization       —— 按 SeedScore 选种子，建立初始角色分配
  3. Score-guided Expansion    —— 在 G0 上增量扩展，只纳入提升综合评分的节点
  4. Role Assignment           —— 把候选节点填入题型逻辑角色槽位
  5. Virtual Logic Completion  —— 物理扩展不足时按缺失角色建虚拟边并验证证据回落
  6. Logical Sufficiency Check —— 角色/证据/约束/结构/难度/shortcut 验证
  7. Question Plan Finalization —— 通过则产出 QuestionPlan

虚拟边只进入 QuestionPlan.semantic_overlay_edges (规划用)，required_nodes /
required_evidence_paths 仍由 G0 物理节点与路径构成，保证 Benchmark 可追溯。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from autoqag.ops.m5_sample.planner import compute_difficulty, difficulty_features
from autoqag.ops.m5_sample.semantic import evidence_chain as ec
from autoqag.ops.m5_sample.semantic import roles as R
from autoqag.ops.m5_sample.semantic import scoring, seed
from autoqag.ops.m5_sample.semantic import virtual_edges as ve
from autoqag.ops.m5_sample.semantic.memory import SemanticMemory
from autoqag.schema import (
    NodeType,
    QuestionGoal,
    QuestionPlan,
    QuestionType,
    VirtualEdge,
)


class OverlayPlanner:
    """问题级语义覆盖层规划器。view 为 m5_sample.sample._GraphView。"""

    def __init__(
        self,
        view,
        domain: str = "",
        memory: Optional[SemanticMemory] = None,
        node_ok: Optional[Callable] = None,
        tau: float = 0.0,
        max_expand: int = 12,
        use_score_guided: bool = True,
        use_binding: bool = True,
        use_overlay: bool = True,
        use_dual_multihop: bool = True,
        use_sufficiency: bool = True,
    ):
        self.view = view
        self.domain = domain
        self.memory = memory or SemanticMemory()
        self.node_ok = node_ok
        self.tau = tau  # 纳入新节点所需的最小评分增益 (§3.4 Accept(v))
        self.max_expand = max_expand
        # 消融开关 (默认全开=完整系统)；逐个关闭以度量各模块边际贡献 (见 exeriments/)。
        self.use_score_guided = use_score_guided    # 评分引导扩展 Accept(v)
        self.use_binding = use_binding              # 题型专属语义绑定
        self.use_overlay = use_overlay              # 虚拟逻辑补全 Ωq
        self.use_dual_multihop = use_dual_multihop  # 双重多跳 + 难度封顶
        self.use_sufficiency = use_sufficiency      # 逻辑充分性硬门槛

    # ----------------------------------------------------------------- #
    def plan_one(self, qtype: str, seed_node: str, qid: str) -> Optional[QuestionPlan]:
        """围绕一个种子节点规划一个 QuestionPlan，失败返回 None。"""
        view = self.view

        # 1. 问题逻辑规划
        goal = self._plan_logic(qtype, seed_node)

        # 2 & 3. 种子初始化 + 评分引导的物理扩展 (得到候选节点集与角色池)
        candidates = ec.expand_candidates(view, [seed_node], max_total=80)
        candidates.add(seed_node)
        role_pool = ec.build_role_pool(view, candidates)

        # 4. 角色分配 (物理优先)
        role_assignment = self._assign_roles(qtype, seed_node, candidates, role_pool)

        # 4b. 题型专属语义绑定：把"值"沿物理边绑回"对象"，确保数值真正属于该对象
        #     (comparative: value_X 经 object_X 的 has_attribute/has_value 链；
        #      cross_paper: paper_B_instance / result_B 取自 same_as 对齐的他刊实例)
        if self.use_binding:
            if qtype == "comparative":
                self._bind_comparative(role_assignment, candidates)
            elif qtype == "cross_paper":
                self._bind_cross_paper(role_assignment, candidates)
            elif qtype in ("numerical", "formula"):
                # 单位绑定为数值真正的 has_unit 邻居，杜绝游离噪声单位 (如 unit='a')
                u = ec.unit_of_value(view, role_assignment.get("value", ""))
                role_assignment["unit"] = u if u else role_assignment.get("unit", "")
                if u:
                    candidates.add(u)

        # 5. 虚拟逻辑补全 (针对缺失角色)
        overlay: List[VirtualEdge] = []
        missing = scoring.missing_roles(qtype, role_assignment)
        if self.use_overlay and missing:
            overlay = self._virtual_completion(
                qtype, goal.difficulty_level, role_assignment, missing, role_pool
            )
            # 接受的虚拟边把其 target 填入角色 + 把 backing path 节点纳入子图
            for e in overlay:
                if e.status == "accepted" and e.question_role and not role_assignment.get(e.question_role):
                    role_assignment[e.question_role] = e.target
                    candidates.update(e.required_physical_nodes)

        # 5b. 证据角色物理回落：仍空缺的 evidence 角色用锚点所属 ChunkNode 接地
        #     (chunk_id == ChunkNode.node_id，ChunkNode 即可追溯的物理证据 span)
        self._ground_evidence(qtype, role_assignment, seed_node, candidates)

        # 5c. 跨文献题：paper_B_instance 必须取自与 paper_A 不同的论文
        if qtype == "cross_paper":
            self._fix_cross_paper(role_assignment, candidates)
            if not self._spans_two_papers(role_assignment):
                return None

        # 6. 评分引导地选定最终子图节点集 (Accept(v): 仅保留提升评分的节点)
        node_ids = self._select_subgraph(
            qtype, goal.difficulty_level, seed_node, role_assignment, candidates, goal.theme
        )

        # 难度由结构变量决定 (复用既有 planner)，再按真实跨 chunk 跨度封顶，
        # 避免单 chunk 富节点子图被误判为 L3/L4 造成伪多跳 (§3.4.4 双重多跳)。
        datas = view.datas(node_ids)
        path_len = max(0, len(node_ids) - 1)
        feats = difficulty_features(datas, path_len)
        difficulty = compute_difficulty(feats)
        if self.use_dual_multihop:
            difficulty = _cap_difficulty(view, node_ids, difficulty, path_len)
        goal.difficulty_level = difficulty

        edges = self._subgraph_edges(node_ids)
        parts = scoring.utility_score(
            view, qtype, difficulty, node_ids, edges, role_assignment,
            theme=goal.theme, path_length=path_len,
        )

        overlay_pattern = self._overlay_pattern(qtype, role_assignment)

        # 7. 逻辑充分性 + 双重多跳验证 (消融时可分别关闭)
        suff_ok = scoring.logical_sufficiency(parts, qtype) if self.use_sufficiency else True
        dual_ok = (
            scoring.dual_multihop_ok(view, node_ids, difficulty, path_len)
            if self.use_dual_multihop else True
        )
        ok = suff_ok and dual_ok
        if not ok:
            # 记录失败模式 (长期记忆)，丢弃该问题
            self.memory.record_pattern(overlay_pattern, success=False)
            return None

        # 通过：沉淀正向记忆
        self.memory.record_pattern(overlay_pattern, success=True)
        self.memory.record_seed(seed_node)

        return self._finalize(qid, qtype, difficulty, goal, seed_node, role_assignment,
                              node_ids, edges, overlay, parts, overlay_pattern)

    # ----------------------------------------------------------------- #
    def _plan_logic(self, qtype: str, seed_node: str) -> QuestionGoal:
        theme = (
            self.view.nodes.get(seed_node, {}).get("normalized_content")
            or self.view.nodes.get(seed_node, {}).get("content", "")
        )[:80]
        return QuestionGoal(
            question_type=qtype,
            domain=self.domain,
            seed_topic=seed_node,
            theme=theme,
            required_reasoning_roles=R.role_schema(qtype),
            expected_answer_form=_answer_form(qtype),
        )

    def _assign_roles(
        self, qtype: str, seed_node: str, candidates, role_pool
    ) -> Dict[str, str]:
        """把候选节点按 NodeType 匹配填入题型逻辑角色 (物理优先；§3.1.1)。"""
        view = self.view
        assignment: Dict[str, str] = {}
        used: set = set()
        schema = R.role_schema(qtype)

        # 种子先占其最匹配的"主"角色 (优先非证据角色，避免占用 evidence 槽位
        # 导致 concept/object 等主角色空缺；§3.1.1 角色由问题逻辑驱动)
        seed_type = view.nodes.get(seed_node, {}).get("node_type", "")
        seed_role = None
        for role in schema:
            if role in R.ABSTRACT_ROLES or R.is_evidence_role(role):
                continue
            if seed_type in R.node_types_for_role(role):
                seed_role = role
                break
        if seed_role is None:  # 退而求其次：允许证据角色
            for role in schema:
                if role not in R.ABSTRACT_ROLES and seed_type in R.node_types_for_role(role):
                    seed_role = role
                    break
        if seed_role:
            assignment[seed_role] = seed_node
            used.add(seed_node)

        # 其余角色从候选集中按类型匹配；同类角色 (A/B) 不重复占用
        cand_list = [c for c in candidates if c != seed_node]
        for role in schema:
            if role in assignment or role in R.ABSTRACT_ROLES:
                continue
            allowed = R.node_types_for_role(role)
            if not allowed:
                continue
            for c in cand_list:
                if c in used:
                    continue
                if view.nodes.get(c, {}).get("node_type", "") in allowed:
                    if self.node_ok and not self.node_ok(view, c) and view.nodes.get(c, {}).get("node_type") in (
                        NodeType.CONCEPT.value, NodeType.ATTRIBUTE.value, NodeType.METHOD.value, NodeType.CLAIM.value
                    ):
                        continue
                    assignment[role] = c
                    used.add(c)
                    break
        return assignment

    def _val_ok(self, nid: str) -> bool:
        """节点是否通过质量门 (过滤符号/噪声值与单位，如 unit='a')。"""
        if not nid:
            return False
        if self.node_ok is None:
            return True
        return self.node_ok(self.view, nid)

    def _bind_comparative(self, role_assignment: Dict[str, str], candidates) -> None:
        """比较题语义绑定：让 value_A/value_B 真正属于 object_A/object_B 在同一指标上的取值。

        类型匹配会把任意 Value 填进 value_A，导致"数值不属于被比较对象"。两条物理路径：
          表格路径 (本语料主导)：对象=表行，指标=共享列，值=该行该列单元格
              (列 --has_value--> 单元格；行 --compares/co_occurs_with--> 单元格)；
          通用链路：object_X --has_attribute--> 共享指标 --has_value--> value_X。
        单位绑定为该数值的 has_unit，或表格列头的 has_unit (杜绝游离噪声单位如 'a')。
        """
        view = self.view
        objA = role_assignment.get("object_A")
        objB = role_assignment.get("object_B")
        if not objA or not objB:
            return

        # 对象规范化：表行标签单元格与行节点同为 ConceptNode，统一回溯到行节点，
        # 否则 shared_metric 取不到行的单元格映射 (绑定退化为类型匹配的错误值)。
        objA = ec.as_table_row(view, objA)
        objB = ec.as_table_row(view, objB)
        if objA != role_assignment.get("object_A"):
            role_assignment["object_A"] = objA
            candidates.add(objA)
        if objB != role_assignment.get("object_B"):
            role_assignment["object_B"] = objB
            candidates.add(objB)

        # (1) 表格路径优先：两行共享列上的单元格才是"同一指标下的可比值"
        sm = ec.shared_metric(view, objA, objB, prefer=role_assignment.get("shared_attribute", ""))
        if sm:
            col, cellA, cellB = sm
            role_assignment["shared_attribute"] = col
            candidates.add(col)
            if self._val_ok(cellA):
                role_assignment["value_A"] = cellA
                candidates.add(cellA)
            if self._val_ok(cellB):
                role_assignment["value_B"] = cellB
                candidates.add(cellB)
            cu = ec.column_unit(view, col)
            if cu:
                role_assignment["unit"] = cu
                candidates.add(cu)
        else:
            # (2) 通用链路：object--has_attribute-->共享指标-->has_value-->value
            attr = role_assignment.get("shared_attribute", "")
            attrsA, attrsB = ec.attributes_of(view, objA), ec.attributes_of(view, objB)
            attrA = attr if attr in attrsA else ""
            attrB = attr if attr in attrsB else ""
            if not (attrA and attrB) and attrsA and attrsB:
                for x in attrsA:
                    xt = ec._tok(ec._content(view, x))
                    for y in attrsB:
                        if xt and (xt & ec._tok(ec._content(view, y))):
                            attrA, attrB = x, y
                            break
                    if attrA and attrB:
                        break
                if attrA:
                    role_assignment["shared_attribute"] = attrA
                    candidates.add(attrA)
            valsA = [v for v in ec.values_via_attribute(view, objA, attrA) if self._val_ok(v)]
            valsB = [v for v in ec.values_via_attribute(view, objB, attrB) if self._val_ok(v)]
            if valsA:
                role_assignment["value_A"] = valsA[0]
                candidates.add(valsA[0])
            if valsB:
                role_assignment["value_B"] = valsB[0]
                candidates.add(valsB[0])

        # 单位绑定 (两路径通用)：先看数值自身 has_unit，再退化到列头单位
        if not role_assignment.get("unit"):
            for vrole in ("value_A", "value_B"):
                u = ec.unit_of_value(view, role_assignment.get(vrole, ""))
                if u:
                    role_assignment["unit"] = u
                    candidates.add(u)
                    break

    def _real_instance(self, nid: str) -> bool:
        """是否为可作"论文实例"的真实节点 (Concept/Method/Value，且非噪声标签)。"""
        nt = self.view.nodes.get(nid, {}).get("node_type", "")
        return nt in (
            NodeType.CONCEPT.value, NodeType.METHOD.value, NodeType.VALUE.value,
        ) and self._val_ok(nid)

    def _nearby_result(self, nid: str) -> Optional[str]:
        """取与 nid 同论文的一个结果节点 (Value 优先，其次同 chunk 的 Value/Claim)。"""
        view = self.view
        pid = self._paper_of(nid)
        for v in ec.values_via_attribute(view, nid):
            if self._paper_of(v) == pid and self._val_ok(v):
                return v
        for m in ec.chunk_mates(view, nid, limit=40):
            nt = view.nodes.get(m, {}).get("node_type")
            if nt in (NodeType.VALUE.value, NodeType.CLAIM.value) and \
                    self._paper_of(m) == pid and self._val_ok(m):
                return m
        return None

    def _bind_cross_paper(self, role_assignment: Dict[str, str], candidates) -> None:
        """跨文献语义绑定：用 same_as 对齐对锚定 A/B 实例，并各取本刊结果值。

        旧逻辑靠类型匹配，常使 paper_B_instance 退化为属性标签、result_A/result_B
        落在同一篇。这里改为：找一个有他刊 same_as 对齐实例的锚点作 paper_A_instance，
        其对齐实例作 paper_B_instance，再分别在各自论文内取 result_A / result_B。
        """
        view = self.view
        anchors = [
            role_assignment.get("paper_A_instance"),
            role_assignment.get("canonical_concept"),
        ] + [v for v in role_assignment.values() if v]
        pair = None
        for anchor in anchors:
            if not anchor:
                continue
            peers = [p for p in ec.same_as_peers(view, anchor, other_paper=True)
                     if self._real_instance(p)]
            if peers:
                pair = (anchor, peers[0])
                break
        if not pair:
            return
        a, b = pair
        role_assignment["paper_A_instance"] = a
        role_assignment["paper_B_instance"] = b
        if not role_assignment.get("canonical_concept"):
            role_assignment["canonical_concept"] = a
        candidates.update(pair)
        ra, rb = self._nearby_result(a), self._nearby_result(b)
        if ra:
            role_assignment["result_A"] = ra
            candidates.add(ra)
        if rb:
            role_assignment["result_B"] = rb
            candidates.add(rb)

    def _ground_evidence(
        self, qtype: str, role_assignment: Dict[str, str], seed_node: str, candidates
    ) -> None:
        """把仍空缺的 evidence 角色回落到锚点所属 ChunkNode (物理证据接地)。

        每个节点 address.chunk_id 即其 ChunkNode 的 node_id，ChunkNode 是可
        追溯到原文的物理证据 span，因此可作为 evidence 角色的合法物理回落。
        evidence_A / evidence_B 尽量取各自比较对象所在 chunk，保证证据可区分。
        """
        view = self.view
        used = set(role_assignment.values())
        evidence_types = {
            NodeType.CHUNK.value, NodeType.EVIDENCE.value,
            NodeType.FIGURE.value, NodeType.TABLE.value,
        }
        for role in R.min_roles(qtype):
            if role in R.ABSTRACT_ROLES or role_assignment.get(role):
                continue
            if not R.is_evidence_role(role):
                continue
            # (a) 候选集中现成的物理证据型节点 (ChunkNode/Evidence/Figure/Table)
            picked = None
            for c in candidates:
                if c in used:
                    continue
                if view.nodes.get(c, {}).get("node_type") in evidence_types:
                    picked = c
                    break
            # (b) 退而求其次：锚点所属 ChunkNode (chunk_id == ChunkNode.node_id)
            if picked is None:
                anchors = [role_assignment.get(a) for a in _EV_ANCHOR.get(role, [])]
                anchors = [a for a in anchors if a]
                anchors += [v for v in role_assignment.values() if v] + [seed_node]
                for src in anchors:
                    cid = view.nodes.get(src, {}).get("address", {}).get("chunk_id", "")
                    if cid and cid in view.nodes and cid not in used:
                        picked = cid
                        break
            if picked:
                role_assignment[role] = picked
                used.add(picked)
                candidates.add(picked)

    def _paper_of(self, nid: str) -> str:
        return self.view.nodes.get(nid, {}).get("address", {}).get("paper_id", "")

    def _fix_cross_paper(self, role_assignment: Dict[str, str], candidates) -> None:
        """跨文献题：确保 paper_B_instance 与 paper_A_instance 来自不同论文。

        若当前 B 实例与 A 同源，则在候选集中按 canonical_id / 同类型寻找
        另一篇论文中的对齐实例替换 (§3.3 CROSS_PAPER_ALIGN 的物理落地)。
        """
        view = self.view
        a = role_assignment.get("paper_A_instance")
        b = role_assignment.get("paper_B_instance")
        pa = self._paper_of(a)
        if not pa or (b and self._paper_of(b) != pa):
            return
        canon = role_assignment.get("canonical_concept")
        cid = view.nodes.get(canon, {}).get("canonical_id", "") if canon else ""
        a_type = view.nodes.get(a, {}).get("node_type", "")
        used = set(role_assignment.values())
        # 优先同 canonical_id 的他刊实例，其次同类型他刊节点
        best = None
        for c in candidates:
            if c in used or self._paper_of(c) == pa:
                continue
            ct = view.nodes.get(c, {})
            if cid and ct.get("canonical_id") == cid:
                best = c
                break
            if best is None and ct.get("node_type") == a_type:
                best = c
        if best is None and cid:
            for nid, d in view.nodes.items():
                if d.get("canonical_id") == cid and self._paper_of(nid) != pa and nid not in used:
                    best = nid
                    break
        if best:
            role_assignment["paper_B_instance"] = best
            candidates.add(best)

    def _spans_two_papers(self, role_assignment: Dict[str, str]) -> bool:
        papers = {self._paper_of(n) for n in role_assignment.values() if n}
        papers.discard("")
        return len(papers) >= 2

    def _virtual_completion(
        self, qtype, difficulty, role_assignment, missing, role_pool
    ) -> List[VirtualEdge]:
        """生成候选虚拟边 → 打分 → 证据回落验证，返回已验证的虚拟边 (按分降序)。"""
        cands = ve.propose_virtual_edges(
            self.view, qtype, role_assignment, missing, role_pool
        )
        validated: List[VirtualEdge] = []
        for e in cands:
            ve.validate_backing(self.view, e)  # 找物理证据回落路径
            e.score = ve.score_edge(self.view, e, qtype, difficulty)
            # 只有能回落到物理证据且评分为正的虚拟边被保留 (§3.6 末)
            if e.status == "accepted" and e.score > 0:
                validated.append(e)
        validated.sort(key=lambda x: x.score, reverse=True)
        return validated

    def _select_subgraph(
        self, qtype, difficulty, seed_node, role_assignment, candidates, theme
    ) -> List[str]:
        """评分引导的增量子图选择 (Accept(v): 只保留提升综合评分的节点；§3.4)。"""
        view = self.view
        # 基底：已分配角色的节点 (问题逻辑必需) + 种子
        base = [seed_node] + [n for n in role_assignment.values() if n]
        base = list(dict.fromkeys(base))

        def score_of(nodes: List[str]) -> float:
            edges = self._subgraph_edges(nodes)
            return scoring.utility_score(
                view, qtype, difficulty, nodes, edges, role_assignment,
                theme=theme, path_length=max(0, len(nodes) - 1),
            )["total"]

        current = list(base)
        cur_score = score_of(current)
        # 候选按是否带物理地址 / 是否证据节点排序，优先纳入高价值证据节点
        extra = [c for c in candidates if c not in current]
        extra.sort(key=lambda c: _cand_priority(view, c), reverse=True)
        if not self.use_score_guided:
            # 消融：无评分引导，按优先级朴素纳入至上限 (不做 Accept(v) 增益门)
            return current + extra[: self.max_expand]
        for c in extra:
            if len(current) >= len(base) + self.max_expand:
                break
            trial = current + [c]
            s = score_of(trial)
            if s - cur_score > self.tau:  # Accept(v)
                current = trial
                cur_score = s
        return current

    def _subgraph_edges(self, node_ids: List[str]) -> List[Tuple[str, str]]:
        nodeset = set(node_ids)
        out = []
        for n in node_ids:
            for t, _ in self.view.out.get(n, []):
                if t in nodeset:
                    out.append((n, t))
        return out

    def _overlay_pattern(self, qtype: str, role_assignment: Dict[str, str]) -> str:
        """overlay pattern 签名：题型 + 已填角色对应的节点类型链 (供长期记忆)。"""
        types = []
        for role, nid in role_assignment.items():
            nt = self.view.nodes.get(nid, {}).get("node_type", "?")
            types.append(nt.replace("Node", ""))
        return f"{qtype}:" + "→".join(sorted(set(types)))

    def _finalize(
        self, qid, qtype, difficulty, goal, seed_node, role_assignment,
        node_ids, edges, overlay, parts, overlay_pattern,
    ) -> QuestionPlan:
        view = self.view
        evidence_spans = [
            {
                "node_id": nid,
                "content": (view.nodes[nid].get("content", "") or "")[:200],
                "address": view.nodes[nid].get("address", {}),
            }
            for nid in node_ids if nid in view.nodes
        ]
        constraints = {"number": [], "unit": [], "condition": [], "formula": [], "table": [], "value_units": []}
        bound_units: List[str] = []
        for nid in node_ids:
            d = view.nodes.get(nid, {})
            nt = d.get("node_type")
            c = d.get("content", "")
            if nt == NodeType.VALUE.value:
                constraints["number"].append(c)
                # 单位绑定：取该数值真正的 has_unit；无则退化到所属表列头的 has_unit。
                # 杜绝把子图里游离的单位节点 (如别的值/表头的 GHz) 张冠李戴到本数值。
                uid = ec.resolve_value_unit(view, nid)
                u = view.nodes.get(uid, {}).get("content", "") if uid else ""
                constraints["value_units"].append({"value": c, "unit": u})
                if u and u not in bound_units:
                    bound_units.append(u)
            elif nt == NodeType.CONDITION.value:
                constraints["condition"].append(c)
            elif nt == NodeType.EQUATION.value:
                constraints["formula"].append(c[:80])
            elif nt == NodeType.TABLE.value:
                constraints["table"].append(c[:80])
        # unit 列表只收与子图内某个数值真正绑定的单位，剔除游离单位节点 (修复 value↔unit 错标)
        constraints["unit"] = bound_units

        papers = list({view.nodes.get(n, {}).get("address", {}).get("paper_id", "") for n in node_ids})
        required_paths = [e.backing_evidence_paths[0] for e in overlay if e.backing_evidence_paths]

        forbidden_gen = []
        if constraints["condition"]:
            forbidden_gen.append("不得省略条件限定，不得泛化到所有场景")
        if constraints["unit"]:
            forbidden_gen.append("不得更改或省略单位")

        forbidden_shortcuts = []
        if difficulty in ("L3", "L4"):
            forbidden_shortcuts.append("不得只基于单一片段作答，必须跨证据/跨chunk整合")

        return QuestionPlan(
            qid=qid,
            domain=self.domain,
            question_type=qtype,
            difficulty=difficulty,
            target_subgraph=node_ids,
            required_nodes=node_ids,
            required_edges=[list(e) for e in edges],
            evidence_spans=evidence_spans,
            constraints=constraints,
            expected_answer_form=goal.expected_answer_form,
            forbidden_generalization=forbidden_gen,
            generation_instruction=_instruction(qtype, role_assignment, difficulty),
            paper_id_list=papers,
            # --- 语义覆盖层字段 ---
            seed_nodes=[seed_node],
            theme=goal.theme,
            semantic_overlay_edges=[e.to_dict() for e in overlay],
            role_assignment=role_assignment,
            required_roles=R.min_roles(qtype),
            required_evidence_paths=required_paths,
            forbidden_shortcuts=forbidden_shortcuts,
            utility_score=parts.get("total", 0.0),
            score_breakdown=parts,
            overlay_pattern=overlay_pattern,
        )


# evidence 角色 → 优先取其所在 chunk 的"主角"角色 (用于证据物理接地)
_EV_ANCHOR = {
    "evidence_A": ["value_A", "object_A"],
    "evidence_B": ["value_B", "object_B"],
    "evidence_span": ["claim", "attribute_or_result", "condition_boundary"],
    "supporting_evidence": ["observed_result", "intermediate_mechanism", "method_or_intervention"],
    "evidence": ["value", "attribute", "concept", "equation", "figure_or_table", "claim"],
}


def _cap_difficulty(view, node_ids: List[str], difficulty: str, path_len: int) -> str:
    """按真实跨 chunk 跨度封顶难度：单 chunk 子图不得标为 L3/L4 (避免伪多跳)。"""
    chunks = {view.nodes.get(n, {}).get("address", {}).get("chunk_id", "") for n in node_ids}
    chunks.discard("")
    n_chunk = len(chunks)
    order = ["L1", "L2", "L3", "L4"]
    if n_chunk <= 1:
        cap = "L1" if path_len <= 1 else "L2"
    elif n_chunk == 2:
        cap = "L3"
    else:
        cap = "L4"
    return cap if order.index(difficulty) > order.index(cap) else difficulty


def _cand_priority(view, nid: str) -> float:
    d = view.nodes.get(nid, {})
    nt = d.get("node_type", "")
    pr = {
        NodeType.VALUE.value: 0.9, NodeType.UNIT.value: 0.85,
        NodeType.CONDITION.value: 0.95, NodeType.EVIDENCE.value: 0.9,
        NodeType.FIGURE.value: 0.7, NodeType.TABLE.value: 0.7,
        NodeType.CLAIM.value: 0.6, NodeType.ATTRIBUTE.value: 0.6,
    }.get(nt, 0.4)
    addr = d.get("address", {})
    if addr.get("chunk_id"):
        pr += 0.05
    return pr


def _answer_form(qtype: str) -> str:
    return {
        QuestionType.NUMERICAL.value: "数值 + 单位",
        QuestionType.CONDITION.value: "结论 + 显式条件限定",
        QuestionType.COMPARATIVE.value: "比较结论 (含对象、指标、对齐条件)",
        "mechanism": "方法→机制→结果 的因果链结论",
        QuestionType.MULTI_HOP.value: "结论 + evidence_path",
        QuestionType.FORMULA.value: "公式相关结论 (保留适用条件)",
        "cross_paper": "跨文献对齐比较结论",
    }.get(qtype, "简短事实结论")


def _instruction(qtype: str, role_assignment: Dict[str, str], difficulty: str) -> str:
    base = {
        QuestionType.NUMERICAL.value: "生成数值敏感问答，答案必须包含原文数值与单位。",
        QuestionType.CONDITION.value: "生成条件边界问答，答案必须显式保留实验/边界条件，不得泛化。",
        QuestionType.COMPARATIVE.value: "生成比较问答，比较对象与指标需一致、条件需对齐。",
        "mechanism": "生成机制解释问答，须串起方法、中间机制与结果，并给出支撑证据。",
        QuestionType.TABLE.value: "生成图表证据问答，答案需引用正确图/表并与图注一致。",
        QuestionType.FORMULA.value: "生成公式依赖问答，需引用正确公式并保留适用条件。",
        QuestionType.MULTI_HOP.value: "生成多跳综合问答，答案需给出完整 evidence_path。",
        "cross_paper": "生成跨文献综合问答，对齐同一概念在不同论文中的指标与条件。",
        QuestionType.SUMMARY.value: "生成章节级综合问答，整合该区域的多个结论。",
    }.get(qtype, "生成问答。")
    if difficulty in ("L3", "L4"):
        base += " (高难度：必须跨多个证据点，避免单 chunk 即可回答。)"
    return base
