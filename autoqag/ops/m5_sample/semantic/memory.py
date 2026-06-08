"""长期记忆：保存模式，而不是修改底图 (语义规划层方法论.pdf §3.9；语义层子图构建.pdf §3.6.5)。

三类记忆 (均为问题级经验沉淀，不污染固定物理证据图 G0):
- Overlay Pattern Memory : 哪些子图结构/虚拟边组合容易生成高质量问题
- Failure Memory         : 哪些模式容易导致不可回答 / 条件缺失 / 伪多跳
- Domain Preference Memory: 专家偏好的重点指标、条件、题型

持久化为单个 JSON 文件 (work_dir/semantic_memory.json)，供下一轮规划初始化使用。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SemanticMemory:
    # overlay_pattern 字符串 -> {"success": int, "fail": int}
    overlay_patterns: Dict[str, Dict[str, int]] = field(default_factory=dict)
    failures: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # domain -> {"attributes": [...], "conditions": [...], "question_types": [...]}
    domain_preference: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    # node_id -> historical_success 计数 (供 SeedScore 加权)
    seed_success: Dict[str, int] = field(default_factory=dict)

    # ---- 记录 ----
    def record_pattern(self, pattern: str, success: bool) -> None:
        slot = self.overlay_patterns.setdefault(pattern, {"success": 0, "fail": 0})
        slot["success" if success else "fail"] += 1
        if not success:
            f = self.failures.setdefault(pattern, {"count": 0})
            f["count"] += 1

    def record_seed(self, node_id: str) -> None:
        self.seed_success[node_id] = self.seed_success.get(node_id, 0) + 1

    def add_domain_preference(self, domain: str, key: str, value: str) -> None:
        d = self.domain_preference.setdefault(domain, {})
        lst = d.setdefault(key, [])
        if value and value not in lst:
            lst.append(value)

    # ---- 查询 ----
    def pattern_weight(self, pattern: str) -> float:
        """overlay pattern 的历史成功率 (0.5 为先验)。"""
        slot = self.overlay_patterns.get(pattern)
        if not slot:
            return 0.5
        total = slot["success"] + slot["fail"]
        return slot["success"] / total if total else 0.5

    def is_known_failure(self, pattern: str, min_count: int = 2) -> bool:
        return self.failures.get(pattern, {}).get("count", 0) >= min_count

    def seed_boost(self) -> Dict[str, float]:
        """node_id -> [0,1] 的历史成功加权 (供 seed.rank_seeds)。"""
        if not self.seed_success:
            return {}
        mx = max(self.seed_success.values())
        return {k: v / mx for k, v in self.seed_success.items()} if mx else {}

    # ---- 持久化 ----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overlay_patterns": self.overlay_patterns,
            "failures": self.failures,
            "domain_preference": self.domain_preference,
            "seed_success": self.seed_success,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SemanticMemory":
        return SemanticMemory(
            overlay_patterns=d.get("overlay_patterns", {}),
            failures=d.get("failures", {}),
            domain_preference=d.get("domain_preference", {}),
            seed_success=d.get("seed_success", {}),
        )

    @staticmethod
    def load(path: str) -> "SemanticMemory":
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return SemanticMemory.from_dict(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
        return SemanticMemory()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
