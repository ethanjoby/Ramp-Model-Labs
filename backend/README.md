# Ascent Labs backend

FastAPI service behind [Ascent Labs](https://github.com/ethanjoby/ascentlabsfrontend): one plain-English prompt in, a
cheap specialist model out. Agents plan the task, build the data, train nineteen candidates in parallel, score every one
on the same held-out rows, and deploy the cheapest that clears the bar behind an API key. Winners can be downloaded
(weights, spec and scorecard) and run anywhere.

This repository is a snapshot of the `backend/` folder of the main project; the frontend lives in its own repository.

## Run

```bash
cp .env.example .env            # FIREBASE_PROJECT_ID is required; ANTHROPIC_API_KEY turns the Claude agents on
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

The frontend proxies `/api` and `/v1` here. Google sign-in is verified against Firebase; set `FIREBASE_PROJECT_ID` to the
same project the frontend's web config uses.

## Optional engines

| Setting | Unlocks |
| --- | --- |
| `ANTHROPIC_API_KEY` | planner for open-ended tasks, synthetic data, rephrased variants, teacher labels, prompted candidates, the frontier yardstick |
| `TOGETHER_API_KEY` | DeepSeek V4.1 Flash, Kimi K3.1 Pro, MiniMax M3.1 Flash, GLM 5.3 Flash and Qwen3.5 9B as zero-shot competitors |
| `LOCAL_LLM_MODEL` | small decoder LLMs fine-tuned on this machine (default Qwen3-0.6B; add `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`) |
| Modal (`uv run modal token new && uv run modal deploy modal_app.py`, then `MODAL_ENABLED=true`) | LoRA/QLoRA fine-tunes on cloud GPUs: Qwen3-1.7B, DeepSeek-R1-Distill 1.5B/7B, Moonlight-16B by default (`MODAL_MODELS=org/model[@GPU][:4bit]`) |

Everything else runs offline: the procedural data engine, the scikit-learn specialists, the embedding specialist, and the
MiniLM / ModernBERT fine-tunes on CPU or Apple GPU.

## Tests

```bash
uv run pytest                   # ~130 tests; the full offline labs in test_api / test_workflows take ~4 min each
```

## API

Cookie-authenticated `/api/*` serves the frontend. The public contract is `/v1`: create an API key in the app, then
`POST /v1/models/{slug}/predict` with `{"inputs": [...]}` returns a label, confidence and top alternatives per row.
`GET /api/deployments/{slug}/artifact` hands the owner a zip of the trained model with its spec, scorecard and loading notes.
