"""Thin Anthropic wrapper that tracks spend so every candidate reports a true cost."""
from __future__ import annotations

import json
import re
from typing import Any

from .config import get_settings

# $ per million tokens (input, output)
PRICING = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}

_client = None


def available() -> bool:
    return bool(get_settings().anthropic_api_key)


def client():
    global _client
    if _client is None:
        from anthropic import AsyncAnthropic

        _client = AsyncAnthropic(api_key=get_settings().anthropic_api_key)
    return _client


def cost_of(model: str, in_tok: int, out_tok: int) -> float:
    pi, po = PRICING.get(model, (3.0, 15.0))
    return (in_tok * pi + out_tok * po) / 1_000_000


class Usage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0

    def add(self, model: str, u: Any) -> None:
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.cost += cost_of(model, u.input_tokens, u.output_tokens)


async def complete(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    max_tokens: int = 1024,
    usage: Usage | None = None,
    temperature: float | None = None,
) -> str:
    model = model or get_settings().fast_model
    kwargs: dict[str, Any] = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    if system:
        kwargs["system"] = system
    if temperature is not None and "haiku" in model:
        kwargs["temperature"] = temperature
    resp = await client().messages.create(**kwargs)
    if usage:
        usage.add(model, resp.usage)
    return "".join(b.text for b in resp.content if b.type == "text")


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of a model response."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON in response")
    return json.JSONDecoder().raw_decode(text[start:])[0]


async def complete_json(prompt: str, **kw) -> Any:
    text = await complete(prompt + "\n\nRespond with JSON only.", **kw)
    return extract_json(text)
