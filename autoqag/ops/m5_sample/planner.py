"""难度计算 (论文 §5.4：难度由结构变量决定，而非语言长度)。

难度变量：evidence_count / path_length / modality_count / section_distance /
constraint_count / cross_paper → L1-L4。
"""

from __future__ import annotations

from typing import Any, Dict, List

from autoqag.schema import Difficulty, NodeType


def difficulty_features(node_datas: List[Dict[str, Any]], path_length: int) -> Dict[str, Any]:
    modalities = {d.get("modality", "text") for d in node_datas}
    sections = {d.get("address", {}).get("section_path", "") for d in node_datas}
    papers = {d.get("address", {}).get("paper_id", "") for d in node_datas}
    constraints = sum(
        1
        for d in node_datas
        if d.get("node_type") in (NodeType.UNIT.value, NodeType.CONDITION.value)
    )
    return {
        "evidence_count": len(node_datas),
        "path_length": path_length,
        "modality_count": len(modalities),
        "section_distance": max(0, len(sections) - 1),
        "constraint_count": constraints,
        "cross_paper": len(papers) > 1,
    }


def compute_difficulty(feats: Dict[str, Any]) -> str:
    """把难度变量映射到 L1-L4 (论文 §5.4 难度等级)。"""
    score = (
        feats["evidence_count"]
        + feats["path_length"]
        + feats["modality_count"]
        + feats["section_distance"] * 2
        + feats["constraint_count"]
        + (3 if feats["cross_paper"] else 0)
    )
    # L4: 跨章节/跨模态/跨文献; L3: 多证据多条件; L2: 单段组合; L1: 单点单边
    if feats["cross_paper"] or feats["section_distance"] >= 1 or feats["modality_count"] >= 2 or score >= 9:
        return Difficulty.L4.value
    if score >= 6:
        return Difficulty.L3.value
    if score >= 3:
        return Difficulty.L2.value
    return Difficulty.L1.value
