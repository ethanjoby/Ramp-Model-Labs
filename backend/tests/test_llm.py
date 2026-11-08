"""Provider routing, pricing and the Together reply parser; nothing here touches the network."""
from app import llm
from app.config import get_settings
from app.trainers.registry import strategies


def test_provider_routing_and_pretty_names():
    assert llm.provider("claude-haiku-5-5") == "anthropic" and llm.provider("moonshotai/Kimi-K3.1-Pro") == "together"
    assert llm.pretty("deepseek-ai/DeepSeek-V4.1-Flash-1017") == "DeepSeek V4.1 Flash"
    assert llm.pretty("claude-sonnet-5-5") == "Claude Sonnet 5.5" and llm.pretty("org/Some-Model-0813") == "Some Model"
    assert llm.slug("deepseek-ai/DeepSeek-V4.1-Flash-1017") == "deepseek_v4_1_flash_1017"


def test_together_requests_switch_thinking_off_and_price_from_the_catalog():
    kw = llm.request_kwargs("moonshotai/Kimi-K3.1-Pro", "classify")
    assert kw == {"max_tokens": 64, "reasoning": {"enabled": False}}
    assert llm.request_kwargs("deepseek-ai/DeepSeek-V4.1-Flash-1017", "plan")["max_tokens"] == 4096
    assert llm.cost_of("deepseek-ai/DeepSeek-V4.1-Flash-1017", 1_000_000, 0) == 0.30
    assert llm.cost_of("moonshotai/Kimi-K3.1-Pro", 0, 1_000_000) == 10.0
    assert llm.cost_of("unknown-org/mystery", 1_000_000, 0) == llm.UNPRICED[0]  # never priced at zero
    u = llm.Usage()
    u.add("Qwen/Qwen3.5-9B", {"prompt_tokens": 2_000_000, "completion_tokens": 1_000_000})
    assert round(u.cost, 4) == 0.59 and u.input_tokens == 2_000_000
    assert llm.cached_input_price("moonshotai/Kimi-K3.1-Pro") == 0.20 and llm.cached_input_price("together/Tev1-4B-experimental") == 0.04


def test_together_reply_parser_ignores_reasoning_and_think_tags():
    data = {"choices": [{"message": {"content": "<think>it is a card decline</think>\ncard_declined", "reasoning_content": "..."},
                        "finish_reason": "stop"}], "usage": {"prompt_tokens": 50, "completion_tokens": 9}}
    text, finish, usage = llm.parse_together(data)
    assert text == "card_declined" and finish == "stop" and usage["completion_tokens"] == 9
    text, finish, _ = llm.parse_together({"choices": [{"message": {"content": [{"type": "text", "text": "fraud"}]}, "finish_reason": "length"}]})
    assert text == "fraud" and finish == "length"


def test_offline_defaults_pick_claude_names_and_together_candidates_are_listed_but_skipped(monkeypatch):
    s = get_settings()
    assert not llm.available() and llm.agent_model() == s.agent_model and llm.reference_model() == s.agent_model
    keys = [st.key for st in strategies()]
    assert {"committee", "embedding_specialist", "finetuned_encoder", "finetuned_modernbert", "frontier_reference"} <= set(keys)
    together = [st for st in strategies() if st.key.startswith("zero_shot_") and st.key != "zero_shot_haiku"]
    assert len(together) == 5 and all(not st.available()[0] and "TOGETHER_API_KEY" in st.available()[1] for st in together)
    assert "DeepSeek V4.1 Flash" in together[0].display_name
    assert "finetuned_local_llm" not in keys  # disabled in the test environment
    monkeypatch.setattr(s, "together_api_key", "tk-test")
    try:
        assert llm.available() and not llm.anthropic_available()
        assert llm.agent_model() == s.together_agent_model and llm.reference_model() == s.together_reference_model
        ref = next(st for st in strategies() if st.key == "frontier_reference")
        assert ref.model == s.together_reference_model and ref.available()[0]
        assert next(st for st in strategies() if st.key == "zero_shot_haiku").available()[0] is False
    finally:
        monkeypatch.setattr(s, "together_api_key", "")
