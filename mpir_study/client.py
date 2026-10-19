"""Thin parallel client for OpenAI-compatible (vLLM) endpoints."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random

DEFAULT_MAX_TOKENS = 1024


@dataclass(frozen=True)
class Endpoint:
    """One served model: where it listens and what it is called."""
    name: str
    base_url: str
    model: str

    def client(self) -> OpenAI:
        return OpenAI(base_url=self.base_url, api_key="EMPTY", timeout=600, max_retries=0)


TARGET = Endpoint("qwen3-1.7b", os.environ.get("TARGET_URL", "http://localhost:8000/v1"), "Qwen/Qwen3-1.7B")
REFINERS = {
    "1.7b": TARGET,
    "4b": Endpoint("qwen3-4b-awq", os.environ.get("REFINER_4B_URL", "http://localhost:8001/v1"), "Qwen/Qwen3-4B-AWQ"),
    "8b": Endpoint("qwen3-8b-awq", os.environ.get("REFINER_8B_URL", "http://localhost:8002/v1"), "Qwen/Qwen3-8B-AWQ"),
}

_CLIENTS: dict[str, OpenAI] = {}


def _client_for(endpoint: Endpoint) -> OpenAI:
    if endpoint.base_url not in _CLIENTS:
        _CLIENTS[endpoint.base_url] = endpoint.client()
    return _CLIENTS[endpoint.base_url]


@retry(stop=stop_after_attempt(4), wait=wait_fixed(2) + wait_random(0, 3), reraise=True)
def chat(endpoint: Endpoint, messages: Sequence[dict], temperature: float = 0.0,
         max_tokens: int = DEFAULT_MAX_TOKENS, seed: Optional[int] = None,
         top_p: Optional[float] = None) -> str:
    """One chat completion with Qwen3 thinking disabled; raises after retries (never returns a sentinel)."""
    kwargs: dict = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    if seed is not None:
        kwargs["seed"] = seed
    if top_p is not None:
        kwargs["top_p"] = top_p
    response = _client_for(endpoint).chat.completions.create(
        model=endpoint.model, messages=list(messages), temperature=temperature,
        max_tokens=max_tokens, **kwargs)
    return response.choices[0].message.content or ""


def chat_many(endpoint: Endpoint, batch: Sequence[Sequence[dict]], workers: int = 48, **kwargs) -> list[str]:
    """Run many independent greedy/sampled requests concurrently; order is preserved."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda messages: chat(endpoint, messages, **kwargs), batch))
