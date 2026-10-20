"""Thin parallel client for OpenAI-compatible (vLLM) endpoints."""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demos", ".env"))

DEFAULT_MAX_TOKENS = 1024


@dataclass(frozen=True)
class Endpoint:
    """One model endpoint: where it listens and what it is called.

    Local vLLM endpoints need no key and accept the Qwen3 chat-template switch; remote (OpenRouter)
    endpoints read their key from the environment and take provider-specific extra parameters."""
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None
    extra_body: tuple = ()  # items of a dict, kept as a tuple so the dataclass stays hashable

    @property
    def remote(self) -> bool:
        return self.api_key_env is not None

    def client(self) -> OpenAI:
        key = os.environ[self.api_key_env] if self.remote else "EMPTY"
        return OpenAI(base_url=self.base_url, api_key=key, timeout=600, max_retries=0)

    def body(self) -> dict:
        return dict(self.extra_body) if self.remote else {"chat_template_kwargs": {"enable_thinking": False}}


OPENROUTER = "https://openrouter.ai/api/v1"


TARGET = Endpoint("qwen3-1.7b", os.environ.get("TARGET_URL", "http://localhost:8000/v1"), "Qwen/Qwen3-1.7B")
REFINERS = {
    "1.7b": TARGET,
    "4b": Endpoint("qwen3-4b-awq", os.environ.get("REFINER_4B_URL", "http://localhost:8001/v1"), "Qwen/Qwen3-4B-AWQ"),
    "8b": Endpoint("qwen3-8b-awq", os.environ.get("REFINER_8B_URL", "http://localhost:8002/v1"), "Qwen/Qwen3-8B-AWQ"),
    # Remote refiners (OpenRouter). Reasoning is switched off so billed output is the answer text only.
    "gpt6luna": Endpoint("gpt-6-luna", OPENROUTER, "openai/gpt-6-luna", "OPENROUTER_API_KEY",
                         (("reasoning", {"effort": "minimal"}),)),
    "qwen235": Endpoint("qwen3-235b-a22b-2507", OPENROUTER, "qwen/qwen3-235b-a22b-2507", "OPENROUTER_API_KEY"),
    "llama70": Endpoint("llama-3.3-70b", OPENROUTER, "meta-llama/llama-3.3-70b-instruct", "OPENROUTER_API_KEY"),
}

_CLIENTS: dict[str, OpenAI] = {}


def _client_for(endpoint: Endpoint) -> OpenAI:
    if endpoint.base_url not in _CLIENTS:
        _CLIENTS[endpoint.base_url] = endpoint.client()
    return _CLIENTS[endpoint.base_url]


# Per-process cap on in-flight requests. Hundreds of simultaneous sockets exhausted the WSL2 localhost
# relay's file descriptors and silently killed the connection to the server.
_INFLIGHT = threading.BoundedSemaphore(int(os.environ.get("MAX_INFLIGHT", "40")))


@retry(stop=stop_after_attempt(6), wait=wait_fixed(5) + wait_random(0, 5), reraise=True)
def chat(endpoint: Endpoint, messages: Sequence[dict], temperature: float = 0.0,
         max_tokens: int = DEFAULT_MAX_TOKENS, seed: Optional[int] = None,
         top_p: Optional[float] = None) -> str:
    """One chat completion with Qwen3 thinking disabled; raises after retries (never returns a sentinel)."""
    kwargs: dict = {"extra_body": endpoint.body()}
    if seed is not None and not endpoint.remote:
        kwargs["seed"] = seed
    if top_p is not None:
        kwargs["top_p"] = top_p
    with _INFLIGHT:
        response = _client_for(endpoint).chat.completions.create(
            model=endpoint.model, messages=list(messages), temperature=temperature,
            max_tokens=max_tokens, **kwargs)
    return response.choices[0].message.content or ""


def chat_many(endpoint: Endpoint, batch: Sequence[Sequence[dict]], workers: int = 48, **kwargs) -> list[str]:
    """Run many independent greedy/sampled requests concurrently; order is preserved."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda messages: chat(endpoint, messages, **kwargs), batch))
