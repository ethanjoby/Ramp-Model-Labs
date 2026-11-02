"""Thin Anthropic wrapper that tracks spend so every candidate reports a true cost."""
from __future__ import annotations

import json
import re
from typing import Any

from .config import get_settings

# $ per million tokens (input, output)
PRICING = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}

# Default output caps per purpose. The 5.5 models think before they answer and thinking counts against
# max_tokens, so a cap sized for the visible label alone returns nothing.
MAX_TOKENS = {"classify": 512, "plan": 4096}

_client = None


class Refused(RuntimeError):
    """The model declined the request (stop_reason "refusal"); there is no answer to parse."""


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


def request_kwargs(model: str, purpose: str = "plan", max_tokens: int | None = None) -> dict[str, Any]:
    """Shape a Messages request for the model it goes to.

    Sampling parameters are rejected by the current API, so effort and thinking are the only cost/quality knobs.
    Sonnet and Opus 5.5 cannot turn thinking off: low effort for classification, medium for planning and chat,
    and a floor on max_tokens so the thinking does not eat the answer. Haiku 5.5 accepts thinking disabled at
    effort high or below, which is the cheapest honest classifier. Anything else gets neither field: Haiku 4.5
    rejects effort, and we do not know the rules for models we have not verified.
    """
    classify = purpose == "classify"
    effort = "low" if classify else "medium"
    floor = MAX_TOKENS["classify" if classify else "plan"]
    kw: dict[str, Any] = {"max_tokens": max_tokens or floor}
    if "haiku-5-5" in model:
        kw["thinking"] = {"type": "disabled"}
        kw["output_config"] = {"effort": effort}
        kw["max_tokens"] = max(kw["max_tokens"], floor)
    elif "sonnet-5-5" in model or "opus-5-5" in model:
        kw["output_config"] = {"effort": effort}
        kw["max_tokens"] = max(kw["max_tokens"], floor)
    return kw


async def complete(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    max_tokens: int | None = None,
    purpose: str = "plan",
    usage: Usage | None = None,
    temperature: float | None = None,
) -> str:
    """One user turn in, the text of the reply out. `purpose` is "classify" or "plan" (see request_kwargs).

    `temperature` is accepted and ignored: the API rejects it now and one older call site still passes it.
    """
    model = model or get_settings().fast_model
    kwargs: dict[str, Any] = dict(model=model, messages=[{"role": "user", "content": prompt}],
                                  **request_kwargs(model, purpose, max_tokens))
    if system:
        kwargs["system"] = system
    text = ""
    for attempt in range(2):
        resp = await client().messages.create(**kwargs)
        if usage:
            usage.add(model, resp.usage)
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            why = getattr(details, "explanation", None) or getattr(details, "category", None) or "no reason given"
            raise Refused(f"{model} declined the request: {why}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        if resp.stop_reason != "max_tokens":
            break
        kwargs["max_tokens"] *= 2  # cut off mid-answer: one retry with room to finish
    return text


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
