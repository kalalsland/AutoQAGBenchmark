"""统一日志 (基于 loguru，失败回退标准 logging)。

统一用 %-style 占位 (`logger.info("x=%s", v)`)：标准 logging 原生支持；
loguru 默认用 {}-style，故用适配器在有 args 时先做 `msg % args` 再输出，
保证两种后端下日志都能正确填充 (修复早期 %s 不被替换的问题)。
"""

import sys


class _LoguruAdapter:
    """把 %-style 调用适配到 loguru。"""

    def __init__(self, logger):
        self._logger = logger

    def _fmt(self, msg, args):
        if args:
            try:
                return str(msg) % args
            except Exception:
                return " ".join([str(msg)] + [str(a) for a in args])
        return str(msg)

    def debug(self, msg, *args):
        self._logger.opt(depth=1).debug(self._fmt(msg, args))

    def info(self, msg, *args):
        self._logger.opt(depth=1).info(self._fmt(msg, args))

    def warning(self, msg, *args):
        self._logger.opt(depth=1).warning(self._fmt(msg, args))

    def error(self, msg, *args):
        self._logger.opt(depth=1).error(self._fmt(msg, args))

    def exception(self, msg, *args):
        self._logger.opt(depth=1).exception(self._fmt(msg, args))


try:
    from loguru import logger as _loguru_logger  # type: ignore

    _loguru_logger.remove()
    _loguru_logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    )
    logger = _LoguruAdapter(_loguru_logger)
except ImportError:  # pragma: no cover - loguru 未安装时回退
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("autoqag")  # type: ignore
