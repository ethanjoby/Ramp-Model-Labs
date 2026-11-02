from app import llm
from app.trainers.prompted import _normalize


def test_request_shaping_per_model():
    haiku = llm.request_kwargs("claude-haiku-5-5", "classify")
    assert haiku.get("thinking") == {"type": "disabled"} and haiku["output_config"]["effort"] == "low"
    sonnet = llm.request_kwargs("claude-sonnet-5-5", "classify")
    assert "thinking" not in sonnet and sonnet["output_config"]["effort"] == "low" and sonnet["max_tokens"] >= 512
    plan = llm.request_kwargs("claude-sonnet-5-5", "plan")
    assert plan["max_tokens"] >= 4096
    for kw in (haiku, sonnet, plan):
        assert not {"temperature", "top_p", "top_k"} & set(kw)


def test_prices_cover_the_current_models():
    assert llm.cost_of("claude-haiku-5-5", 1_000_000, 0) == 0.10
    assert llm.cost_of("claude-sonnet-5-5", 0, 1_000_000) == 10.0


def test_label_normalization_never_invents_a_label():
    labels = ["fraud", "legitimate"]
    assert _normalize("Fraud", labels) == "fraud"
    assert _normalize("legit", labels) in ("legitimate", None)
    assert _normalize("banana", labels) is None
    assert _normalize("", labels) is None
