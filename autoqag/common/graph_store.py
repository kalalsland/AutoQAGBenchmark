"""Schema-Evidence Graph 存储 (改编自 GraphGen networkx_storage.py)。

基于 networkx，支持：
- upsert 点/边、查询邻居/度/连通分量
- graphml 持久化
- nodes.jsonl / edges.jsonl 导入导出 (论文 §5.9 图谱输出规格)
- 物理地址索引：按 (paper_id, section_path, chunk_id) 快速定位点 (图谱构建.pdf 纵向定位)
- canonical_id 索引：跨文献同类点聚合 (图谱构建.pdf 横向关联)
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from autoqag.common.io import read_jsonl, write_jsonl
from autoqag.schema import Edge, PointNode


class GraphStore:
    def __init__(self):
        import networkx as nx

        self._nx = nx
        self._graph = nx.DiGraph()

    # ----- 点 -----
    def upsert_node(self, node: PointNode) -> None:
        self._graph.add_node(node.node_id, **node.to_dict())

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self._graph.nodes.get(node_id)

    def all_nodes(self) -> List[Tuple[str, Dict[str, Any]]]:
        return list(self._graph.nodes(data=True))

    def nodes_by_type(self, node_type: str) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            (n, d) for n, d in self._graph.nodes(data=True) if d.get("node_type") == node_type
        ]

    # ----- 边 -----
    def upsert_edge(self, edge: Edge) -> None:
        if not (self.has_node(edge.source) and self.has_node(edge.target)):
            return
        self._graph.add_edge(edge.source, edge.target, **edge.to_dict())

    def has_edge(self, s: str, t: str) -> bool:
        return self._graph.has_edge(s, t)

    def all_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        return list(self._graph.edges(data=True))

    def neighbors(self, node_id: str) -> List[str]:
        if not self.has_node(node_id):
            return []
        # DiGraph：取双向邻居 (无向语义)
        return list(set(self._graph.successors(node_id)) | set(self._graph.predecessors(node_id)))

    def degree(self, node_id: str) -> int:
        return int(self._graph.degree(node_id)) if self.has_node(node_id) else 0

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def connected_components(self) -> List[Set[str]]:
        g = self._graph.to_undirected()
        return [set(c) for c in self._nx.connected_components(g)]

    # ----- 索引 -----
    def index_by_address(self) -> Dict[Tuple[str, str, str], List[str]]:
        """(paper_id, section_path, chunk_id) → node_id 列表 (纵向定位)。"""
        idx: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        for n, d in self._graph.nodes(data=True):
            addr = d.get("address", {})
            key = (addr.get("paper_id", ""), addr.get("section_path", ""), addr.get("chunk_id", ""))
            idx[key].append(n)
        return idx

    def index_by_canonical(self) -> Dict[str, List[str]]:
        """canonical_id → node_id 列表 (跨文献同类点聚合)。"""
        idx: Dict[str, List[str]] = defaultdict(list)
        for n, d in self._graph.nodes(data=True):
            cid = d.get("canonical_id")
            if cid:
                idx[cid].append(n)
        return idx

    # ----- 持久化 -----
    def save_graphml(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # graphml 不支持嵌套 dict/list，导出时序列化为字符串
        g = self._nx.DiGraph()
        for n, d in self._graph.nodes(data=True):
            g.add_node(n, **{k: _flatten(v) for k, v in d.items()})
        for s, t, d in self._graph.edges(data=True):
            g.add_edge(s, t, **{k: _flatten(v) for k, v in d.items()})
        self._nx.write_graphml(g, path)

    def export_jsonl(self, nodes_path: str, edges_path: str) -> Tuple[int, int]:
        n = write_jsonl(nodes_path, (d for _, d in self._graph.nodes(data=True)))
        e = write_jsonl(edges_path, (d for _, _, d in self._graph.edges(data=True)))
        return n, e

    @classmethod
    def load_jsonl(cls, nodes_path: str, edges_path: str) -> "GraphStore":
        store = cls()
        for row in read_jsonl(nodes_path):
            store.upsert_node(PointNode.from_dict(row))
        for row in read_jsonl(edges_path):
            store.upsert_edge(Edge.from_dict(row))
        return store


def _flatten(v: Any) -> Any:
    """graphml 仅支持标量，dict/list 转 JSON 字符串。"""
    if isinstance(v, (dict, list, tuple)):
        import json

        return json.dumps(v, ensure_ascii=False)
    if v is None:
        return ""
    return v
