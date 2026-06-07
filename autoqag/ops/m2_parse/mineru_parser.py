"""MinerU 调用封装 (改编自 GraphGen graphgen/models/reader/pdf_reader.py)。

关键差异：GraphGen 版本在加载 content_list 时删除了 page_idx/bbox/text_level，
而本项目的物理地址 (page/bbox/section 层级) 是图谱构建的基础，必须保留。
因此这里完整返回 content_list 的所有字段，并把相对 img 路径解析为绝对路径。
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from autoqag.common.logging import logger


class MinerUParser:
    """通过 mineru CLI 解析 PDF，返回保留物理地址的 content_list。"""

    @staticmethod
    def is_available() -> bool:
        try:
            subprocess.run(
                ["mineru", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def parse_pdf(
        pdf_path: Union[str, Path],
        output_dir: Union[str, Path],
        method: str = "auto",
        backend: str = "pipeline",
        device: str = "cpu",
        lang: Optional[str] = None,
        **kw: Any,
    ) -> Optional[List[Dict[str, Any]]]:
        """返回 content_list（含 page_idx/bbox/text_level/img_path 等）。失败返回 None。"""
        pdf = Path(pdf_path).expanduser().resolve()
        if not pdf.is_file():
            raise FileNotFoundError(pdf)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 优先复用已有解析结果 (缓存)
        cached = MinerUParser._load_content_list(str(out), pdf.stem, method, backend)
        if cached is not None:
            return cached

        MinerUParser._run_cli(pdf, out, method, backend, device, lang, **kw)
        return MinerUParser._load_content_list(str(out), pdf.stem, method, backend)

    @staticmethod
    def _content_list_path(out_dir: str, stem: str, method: str, backend: str) -> str:
        # backend=vlm-* 时 method 目录名为 vlm
        m = "vlm" if backend.startswith("vlm-") else method
        return os.path.join(out_dir, stem, m, f"{stem}_content_list.json")

    @staticmethod
    def _load_content_list(
        out_dir: str, stem: str, method: str, backend: str
    ) -> Optional[List[Dict[str, Any]]]:
        json_file = MinerUParser._content_list_path(out_dir, stem, method, backend)
        if not os.path.exists(json_file):
            return None
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("加载 MinerU 结果失败: %s", exc)
            return None

        base = os.path.dirname(json_file)
        for it in data:
            for key in ("img_path", "table_img_path", "equation_img_path"):
                rel = it.get(key)
                if rel:
                    it[key] = str(Path(base).joinpath(rel).resolve())
        return data

    @staticmethod
    def _run_cli(
        pdf: Path,
        out: Path,
        method: str,
        backend: str,
        device: str,
        lang: Optional[str],
        **kw: Any,
    ) -> None:
        cmd = ["mineru", "-p", str(pdf), "-o", str(out), "-m", method, "-b", backend]
        if device:
            cmd += ["-d", device]
        if lang:
            cmd += ["-l", lang]
        for k, v in kw.items():
            if v is None:
                continue
            cmd += [f"--{k}", str(v).lower() if isinstance(v, bool) else str(v)]

        logger.info("MinerU 解析: %s", pdf.name)
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"MinerU 失败: {proc.stderr or proc.stdout}")
