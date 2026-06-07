"""OpenAI 兼容异步 LLM 客户端 (改编自 GraphGen openai_client.py + base_llm_wrapper.py)。

- 默认走 OpenAI 兼容 API (base_url 可指向 DeepSeek/Qwen/本地 vLLM 等)。
- 保留 tenacity 重试、RPM/TPM 限流、token 计数、<think> 过滤。
- 额外提供同步便捷封装 (generate / generate_batch)，供 stage 内非 async 代码直接调用。
- from_env() 从环境变量构建，便于 recipe 不写密钥。
"""

from __future__ import annotations

import asyncio
import math
import os
import re
from typing import Any, Dict, List, Optional

from autoqag.common.limiter import RPM, TPM
from autoqag.common.logging import logger
from autoqag.schema import Token

try:  # tiktoken 用于 token 计数与限流估算，缺失时回退到字符估算
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text))

except Exception:  # pragma: no cover

    def _count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


def _filter_think_tags(text: str, think_tag: str = "think") -> str:
    """移除 <think>...</think> 推理标签 (沿用 GraphGen 逻辑)。"""
    paired = re.compile(rf"<{think_tag}>.*?</{think_tag}>", re.DOTALL)
    filtered = paired.sub("", text)
    orphan = re.compile(rf"^.*?</{think_tag}>", re.DOTALL)
    filtered = orphan.sub("", filtered).strip()
    return filtered if filtered else text.strip()


class LLMClient:
    """OpenAI 兼容异步客户端。"""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        seed: Optional[int] = None,
        json_mode: bool = False,
        topk_per_token: int = 5,
        request_limit: bool = True,
        rpm: int = 1000,
        tpm: int = 50000,
        max_concurrency: int = 8,
        **kwargs: Any,
    ):
        self.model = model
        self.api_key = api_key or "dummy"
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = float(top_p)
        self.seed = seed
        self.json_mode = json_mode
        self.topk_per_token = topk_per_token
        self.request_limit = request_limit
        self.rpm = RPM(rpm)
        self.tpm = TPM(tpm)
        self.max_concurrency = int(max_concurrency)
        # 信号量按事件循环惰性创建：每个 stage 经 asyncio.run 新建 loop，
        # 在 __init__ 绑定的信号量会因 "bound to a different event loop" 失效。
        self._sem = None
        self._sem_loop = None
        self.token_usage: List[Dict[str, int]] = []
        self._client = None  # 懒加载，避免无 openai 包时 import 失败
        self._client_loop = None

    # ----- 工厂 -----
    @classmethod
    def from_env(cls, **override: Any) -> "LLMClient":
        """从环境变量构建：
        AUTOQAG_MODEL / AUTOQAG_API_KEY / AUTOQAG_BASE_URL，
        回退到 OPENAI_MODEL / OPENAI_API_KEY / OPENAI_BASE_URL。
        """
        cfg = {
            "model": os.getenv("AUTOQAG_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini",
            "api_key": os.getenv("AUTOQAG_API_KEY") or os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("AUTOQAG_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        }
        cfg.update({k: v for k, v in override.items() if v is not None})
        return cls(**cfg)

    @property
    def client(self):
        """返回绑定当前事件循环的 AsyncOpenAI 客户端 (内部 httpx 绑 loop，跨 loop 重建)。"""
        from openai import AsyncOpenAI

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if self._client is None or self._client_loop is not loop:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            self._client_loop = loop
        return self._client

    def _get_sem(self) -> "asyncio.Semaphore":
        """返回绑定到当前事件循环的信号量；跨 loop 自动重建。"""
        loop = asyncio.get_event_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(self.max_concurrency)
            self._sem_loop = loop
        return self._sem

    # ----- 内部 -----
    def _build_messages(
        self, text: str, system: Optional[str], history: Optional[List[str]]
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        sys_prompt = system if system is not None else self.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        if history:
            assert len(history) % 2 == 0, "history 必须为偶数个元素"
            for i, h in enumerate(history):
                messages.append(
                    {"role": "user" if i % 2 == 0 else "assistant", "content": h}
                )
        messages.append({"role": "user", "content": text})
        return messages

    # ----- 异步核心 -----
    async def agenerate(
        self,
        text: str,
        *,
        system: Optional[str] = None,
        history: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **extra: Any,
    ) -> str:
        from openai import (  # 延迟 import
            APIConnectionError,
            APIError,
            APITimeoutError,
            RateLimitError,
        )
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        try:
            from openai import InternalServerError  # 5xx (含服务过载)
        except ImportError:  # pragma: no cover
            InternalServerError = APIError  # type: ignore

        messages = self._build_messages(text, system, history)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        est = sum(_count_tokens(m["content"]) for m in messages) + kwargs["max_tokens"]

        @retry(
            stop=stop_after_attempt(6),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(
                (
                    RateLimitError,
                    APIConnectionError,
                    APITimeoutError,
                    InternalServerError,  # 瞬时 5xx / 服务过载
                )
            ),
        )
        async def _call():
            async with self._get_sem():
                if self.request_limit:
                    await self.rpm.wait(silent=True)
                    await self.tpm.wait(est, silent=True)
                return await self.client.chat.completions.create(**kwargs)

        completion = await _call()
        if getattr(completion, "usage", None):
            self.token_usage.append(
                {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }
            )
        return _filter_think_tags(completion.choices[0].message.content or "")

    async def agenerate_topk_per_token(
        self, text: str, *, system: Optional[str] = None
    ) -> List[Token]:
        """返回下一个 token 的 top-k 概率，用于验证器 comprehension-loss 打分。"""
        messages = self._build_messages(text, system, None)
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=self.topk_per_token,
        )
        token_logprobs = completion.choices[0].logprobs.content
        tokens: List[Token] = []
        for tp in token_logprobs:
            candidates = [Token(t.token, math.exp(t.logprob)) for t in tp.top_logprobs]
            tokens.append(
                Token(tp.token, math.exp(tp.logprob), top_candidates=candidates)
            )
        return tokens

    # ----- 同步便捷封装 (stage 内同步代码使用) -----
    def generate(self, text: str, **kw: Any) -> str:
        return _run_sync(self.agenerate(text, **kw))

    def generate_batch(self, texts: List[str], **kw: Any) -> List[str]:
        async def _all():
            return await asyncio.gather(
                *[self.agenerate(t, **kw) for t in texts], return_exceptions=True
            )

        results = _run_sync(_all())
        out: List[str] = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("LLM batch item failed: %s", r)
                out.append("")
            else:
                out.append(r)
        return out


def _run_sync(coro):
    """在同步上下文运行协程；若已在事件循环中则新开线程跑。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)
