"""IO 辅助：jsonl / json 读写与工作目录管理。"""

import json
import os
from typing import Any, Dict, Iterable, Iterator, List


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    """写 jsonl，返回写入条数。"""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl_list(path: str) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
