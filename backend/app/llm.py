"""One client for every model the lab talks to, with spend tracked so every candidate reports a true cost.

Two providers: Anthropic (Claude) and Together AI (an OpenAI-compatible endpoint serving DeepSeek, Kimi,
MiniMax, GLM, Qwen and others). A model id picks its provider: `claude-*` goes to Anthropic, anything else to
Together. Either key alone is enough for the agents; with both, Claude runs the agents and the Together
models compete as prompted candidates.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from .config import get_settings

# $ per million tokens (input, output). Together prices are the catalog's list prices.
PRICING = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": (0.30, 1.20),
    "deepseek-ai/DeepSeek-V4.1-Flash": (0.30, 1.20),
    "deepseek-ai/DeepSeek-V4-Pro-0813": (1.32, 3.96),
    "moonshotai/Kimi-K3.1-Pro": (2.00, 10.00),
    "moonshotai/Kimi-K3": (3.00, 15.00),
    "MiniMaxAI/MiniMax-M3.1-Flash": (0.15, 0.60),
    "MiniMaxAI/MiniMax-M3": (0.30, 1.20),
    "zai-org/GLM-5.3": (1.40, 4.40),
    "zai-org/GLM-5.3-Flash": (0.15, 0.50),
    "zai-org/GLM-5.2": (1.40, 4.40),
    "meta-models/Muse-Spark-Open-816B-A42B": (1.25, 4.25),
    "meta-models/Muse-Glimmer-30B": (0.35, 1.50),
    "reflection/Beam-1.0-501B-A23B": (0.40, 1.20),
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL": (0.15, 0.50),
    "ricursive-ai/Ripple-65B-Preview": (0.20, 0.60),
    "Qwen/Qwen3.8-Flash": (0.15, 0.47),
    "Qwen/Qwen3.7-Max": (2.50, 7.50),
    "Qwen/Qwen3.5-9B": (0.17, 0.25),
    "together/Tev1-4B-experimental": (0.04, 0.0),
}
CACHED_INPUT = {  # $ per million cached input tokens, for pricing a frontier model that serves its instruction from cache
    "claude-sonnet-5-5": 0.20, "claude-haiku-5-5": 0.01, "claude-opus-5-5": 0.40,
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": 0.006, "deepseek-ai/DeepSeek-V4.1-Flash": 0.006, "moonshotai/Kimi-K3.1-Pro": 0.20,
    "moonshotai/Kimi-K3": 0.30, "MiniMaxAI/MiniMax-M3.1-Flash": 0.015, "zai-org/GLM-5.3-Flash": 0.03, "zai-org/GLM-5.3": 0.26,
    "Qwen/Qwen3.5-9B": 0.17, "Qwen/Qwen3.7-Max": 0.50, "deepseek-ai/DeepSeek-V4-Pro-0813": 0.13,
}
UNPRICED = (1.0, 3.0)  # a model missing from the table is billed at this rate and flagged, never priced at zero

# Default output caps per purpose. The 5.5 models think before they answer and thinking counts against
# max_tokens, so a cap sized for the visible label alone returns nothing.
MAX_TOKENS = {"classify": 512, "plan": 4096}
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)

_client = None
_http = None


class Refused(RuntimeError):
    """The model declined the request (stop_reason "refusal"); there is no answer to parse."""


# ------------------------------------------------------------------ providers

def anthropic_available() -> bool:
    return bool(get_settings().anthropic_api_key)


def together_available() -> bool:
    return bool(get_settings().together_api_key)


def available() -> bool:
    """Any provider that can run the agents (planner, synthesizer, teacher, prompted candidates)."""
    return anthropic_available() or together_available()


def provider(model: str) -> str:
    return "anthropic" if model.startswith("claude-") else "together"


def provider_available(model: str) -> bool:
    return anthropic_available() if provider(model) == "anthropic" else together_available()


def _pick(claude: str, together: str) -> str:
    """Claude when configured; Together's model when only that key is set; Claude's name (for display and list-price
    estimates) when nothing is configured."""
    if anthropic_available() or not together_available():
        return claude
    return together


def agent_model() -> str:
    """The model that plans, synthesizes data, labels rows and explains picks."""
    s = get_settings()
    return _pick(s.agent_model, s.together_agent_model)


def fast_model() -> str:
    s = get_settings()
    return _pick(s.fast_model, s.together_agent_model)


def reference_model() -> str:
    """The frontier yardstick every lab is measured against."""
    s = get_settings()
    return _pick(s.agent_model, s.together_reference_model)


def together_models() -> list[str]:
    return [m.strip() for m in get_settings().together_models.split(",") if m.strip()]


_DISPLAY = {
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": "DeepSeek V4.1 Flash", "deepseek-ai/DeepSeek-V4.1-Flash": "DeepSeek V4.1 Flash",
    "deepseek-ai/DeepSeek-V4-Pro-0813": "DeepSeek V4 Pro", "moonshotai/Kimi-K3.1-Pro": "Kimi K3.1 Pro", "moonshotai/Kimi-K3": "Kimi K3",
    "MiniMaxAI/MiniMax-M3.1-Flash": "MiniMax M3.1 Flash", "MiniMaxAI/MiniMax-M3": "MiniMax M3", "zai-org/GLM-5.3": "GLM 5.3",
    "zai-org/GLM-5.3-Flash": "GLM 5.3 Flash", "zai-org/GLM-5.2": "GLM 5.2", "Qwen/Qwen3.8-Flash": "Qwen3.8 Flash",
    "Qwen/Qwen3.7-Max": "Qwen3.7 Max", "Qwen/Qwen3.5-9B": "Qwen3.5 9B", "openai/gpt-oss-120b": "GPT-OSS 120B",
}


def pretty(model_id: str) -> str:
    """claude-haiku-5-5 -> Claude Haiku 5.5; moonshotai/Kimi-K3.1-Pro -> Kimi K3.1 Pro."""
    if model_id in _DISPLAY:
        return _DISPLAY[model_id]
    if model_id.startswith("claude-"):
        parts = model_id.replace("claude-", "").split("-")
        name, ver = parts[0].capitalize(), ".".join(p for p in parts[1:] if p.isdigit())
        return f"Claude {name} {ver}".strip()
    tail = model_id.split("/")[-1]
    tail = re.sub(r"-\d{4}$", "", tail)  # a dated snapshot suffix is not part of the name
    return tail.replace("-", " ")


def slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_id.split("/")[-1].lower()).strip("_")


def client():
    global _client
    if _client is None:
        from anthropic import AsyncAnthropic

        _client = AsyncAnthropic(api_key=get_settings().anthropic_api_key)
    return _client


def _http_client():
    global _http
    if _http is None:
        import httpx

        _http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0))
    return _http


# ------------------------------------------------------------------ cost

def priced(model: str) -> bool:
    return model in PRICING


def cached_input_price(model: str) -> float:
    return CACHED_INPUT.get(model, PRICING.get(model, UNPRICED)[0])


def cost_of(model: str, in_tok: int, out_tok: int) -> float:
    pi, po = PRICING.get(model, (3.0, 15.0) if model.startswith("claude-") else UNPRICED)
    return (in_tok * pi + out_tok * po) / 1_000_000


class Usage:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0

    def add(self, model: str, u: Any) -> None:
        if isinstance(u, dict):  # OpenAI-style usage block
            i, o = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        else:
            i, o = int(getattr(u, "input_tokens", 0) or 0), int(getattr(u, "output_tokens", 0) or 0)
        self.input_tokens += i
        self.output_tokens += o
        self.cost += cost_of(model, i, o)


# ------------------------------------------------------------------ requests

def request_kwargs(model: str, purpose: str = "plan", max_tokens: int | None = None) -> dict[str, Any]:
    """Shape a request for the model it goes to.

    Anthropic: sampling parameters are rejected by the current API, so effort and thinking are the only
    cost/quality knobs. Sonnet and Opus 5.5 cannot turn thinking off: low effort for classification, medium for
    planning and chat, and a floor on max_tokens so the thinking does not eat the answer. Haiku 5.5 accepts
    thinking disabled at effort high or below, which is the cheapest honest classifier. Anything else gets
    neither field: Haiku 4.5 rejects effort, and we do not know the rules for models we have not verified.

    Together: the reasoning models (DeepSeek, Kimi, GLM, MiniMax) think by default and bill the trace against
    the same completion budget, so thinking is switched off (`reasoning.enabled=false` is documented as
    generating and billing no thinking tokens). A classifier answers; it does not deliberate.
    """
    classify = purpose == "classify"
    if provider(model) == "together":
        return {"max_tokens": max_tokens or (64 if classify else MAX_TOKENS["plan"]), "reasoning": {"enabled": False}}
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


def parse_together(data: dict) -> tuple[str, str, dict]:
    """(answer text, finish reason, usage) from a chat-completions reply. Reasoning comes back in its own field
    (`reasoning_content` / `reasoning`) or, from some templates, inside <think> tags; neither is the answer."""
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content")
    if isinstance(text, list):  # content parts
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    text = _THINK.sub("", text or "").strip()
    return text, str(choice.get("finish_reason") or ""), data.get("usage") or {}


async def _together(prompt: str, system: str, model: str, kwargs: dict[str, Any], usage: Usage | None) -> str:
    key = get_settings().together_api_key
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    payload: dict[str, Any] = {"model": model, "messages": messages, **kwargs}
    text = ""
    for attempt in range(4):
        r = await _http_client().post(TOGETHER_URL, json=payload, headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 400 and "reasoning" in payload:
            payload.pop("reasoning", None)  # a model that predates the switch does not need it
            continue
        if r.status_code in (429, 500, 502, 503, 529) and attempt < 3:
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"Together {r.status_code} for {model}: {r.text[:200]}")
        text, finish, u = parse_together(r.json())
        if usage:
            usage.add(model, u)
        if finish == "length" and attempt == 0:
            payload["max_tokens"] = int(payload.get("max_tokens") or 64) * 4  # cut off mid-answer: one retry with room
            continue
        break
    return text


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

    `temperature` is accepted and ignored: the Anthropic API rejects it now and one older call site still passes it.
    """
    model = model or fast_model()
    if provider(model) == "together":
        return await _together(prompt, system, model, request_kwargs(model, purpose, max_tokens), usage)
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
