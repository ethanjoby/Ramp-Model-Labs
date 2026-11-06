# Ramp Model Labs

One prompt in, a cheap specialist model out.

Describe a financial workflow in plain English (expense categorization, fraud screening, invoice triage, support routing, transaction reconciliation, or anything you can label). A team of agents writes the spec, builds a dataset, trains competing candidates in parallel, scores every one on the same held-out set, and deploys the cheapest model that clears the bar behind an API key.

## Run it

```bash
# backend (FastAPI, port 8000)
cd backend
cp .env.example .env            # FIREBASE_PROJECT_ID is required; ANTHROPIC_API_KEY turns the Claude agents on
uv sync
uv run uvicorn app.main:app --reload --port 8000

# frontend (Next.js 16, port 3000)
cd frontend
cp .env.example .env.local      # paste the Firebase web config
pnpm install
pnpm dev
```

Open http://localhost:3000. The frontend proxies `/api` and `/v1` to the backend, so the session cookie never leaves the page's origin.

### Sign-in (Google, through Firebase)

Sign-in is Google only. The browser gets a Firebase ID token from Google, the backend verifies it against Google's public certificates (audience, issuer, expiry, verified email) and issues its own signed httpOnly session cookie. No Firebase Admin SDK or service account is needed.

One-time setup in the [Firebase console](https://console.firebase.google.com/):

1. Create a project and register a web app; copy its config into `frontend/.env.local` and the project ID into `backend/.env`.
2. Authentication → Get started → Sign-in method → **Google** → Enable → Save.
3. For a deployed domain, add it under Authentication → Settings → Authorized domains (`localhost` is already there).

To test the flow without real Google accounts, run `firebase emulators:start --only auth`, set `NEXT_PUBLIC_FIREBASE_AUTH_EMULATOR=http://127.0.0.1:9099` for the frontend and `FIREBASE_AUTH_EMULATOR_HOST=127.0.0.1:9099` for the backend.

### Optional engines

Everything works with no other keys: the procedural data engine and the local specialist models run offline. Keys unlock more candidates.

| Key | Unlocks |
| --- | --- |
| `torch` + `transformers` (installed by `uv sync`) | The local encoder fine-tunes: MiniLM-L6 (22M) and ModernBERT-base (150M, after a short masked-LM warm-up on your own rows); without them both candidates are listed as skipped |
| `ANTHROPIC_API_KEY` | Planner for open-ended tasks, synthetic data, prompted candidates (zero-shot Haiku, optimized few-shot program), the frontier reference, the cascade's LLM level, the copilot's natural-language tool use |
| Modal (`uv run modal token new && uv run modal deploy modal_app.py`, then `MODAL_ENABLED=true`) | LoRA fine-tune of Qwen3-1.7B on an L4 GPU (`MODAL_BASE_MODEL` to change it, `MODAL_HEAVY_MODEL` to add a second, e.g. `Qwen/Qwen3-4B-Instruct-2507`), plus serving for deployed adapters. `uv run modal run modal_app.py` is a smoke test that prints accuracy, timing and GPU cost (cents) |

## What a lab does

1. **Planner** turns the prompt into a spec: labels, target accuracy, latency budget, what to optimize for. Uploaded examples define the labels when they exist.
2. **Curator** builds the dataset. Real examples are reserved for evaluation first, then the procedural generators and Claude fill in training data across several styles. Small uploads get label-preserving variants of their training rows (never of test or dev rows). Near-duplicates are dropped, splits are stratified.
3. **Trainer** runs every available strategy in parallel against the same splits:
   - local specialists: Naive Bayes, tuned linear, SVM, tiny neural net
   - real language models, fine-tuned locally: MiniLM-L6 (22M parameters) and ModernBERT-base (150M) with a new classification head, trained end to end on the lab's split on Apple-silicon GPU or CPU (10-20 s and 1-3 min), evaluated and served on CPU (about 7 ms and 100 ms a call). ModernBERT first spends 45 s on a masked-LM warm-up over the task's own inputs, the laptop-sized version of task-adaptive continued pretraining. DeBERTa-v3 was tried and dropped: it produces NaN losses on Apple GPUs
   - cascade: learned rules → small model → LLM → human review
   - prompted: zero-shot Haiku, optimized few-shot prompt program
   - fine-tuned on a GPU: LoRA (rank 8, every linear layer) on Qwen3-1.7B on a Modal L4, with label-only targets and thinking switched off, because chain-of-thought hurts models under ~10B parameters; serving cost is priced from the batched throughput measured in the same container, and labelled an estimate
   - frontier reference (Sonnet), used as the yardstick, never deployed
4. **Judge** keeps candidates that meet the bar, takes the cheapest within tolerance of the best, and explains the pick. Accuracy, latency and cost per 1k calls are all measured, not estimated, except the frontier cost when no key is present.
5. **Deployer** publishes `POST /v1/models/{slug}/predict`. Corrections made in the playground become real labeled data; retraining re-runs the lab with them.

The homepage replays one real run from its stored events (`frontend/src/components/marketing/demo-lab.json`), through the same reducer the product uses.

### Teacher labels

Uploaded rows without a label are not thrown away. With an Anthropic key, Sonnet labels them by asking three ways (plain instruction, label meanings, labeled examples) and keeping only the labels a majority agrees on; disagreements are dropped, never guessed. Teacher rows train the model and never score it, so a teacher's mistakes cannot become the test. Capped by `TEACHER_MAX_ROWS` (default 1,500, about $1 per 1,000 rows).

## Workflows: a file in, decisions out

A workflow takes what a finance team actually has and runs it through the best model:

- **Inputs**: `.xlsx`/`.csv`/`.tsv` (text and label columns are inferred from headers, then from the shape of the data; an amount column is appended to the text), `.eml` or pasted email text (newest message only, quoted threads dropped, CSV/XLSX attachments parsed), PDFs (one row for short documents, one row per line for statements), JSON/JSONL and plain text. Parsing is pure Python and read-only: no formulas, macros or network.
- **Model choice**: a model you name, else the live model whose task best matches your prompt and the labels in the file, else a new lab is built on the spot when the file is labeled or the prompt matches a built-in workflow.
- **Run**: every row is classified in batches; rows under the confidence floor, or routed to review by the cascade, land in a review queue. A person's decision is saved as feedback for the model's next retrain.
- **Outputs**: a summary (distribution, automatic vs review, mean confidence, measured cost against the frontier estimate, accuracy on any labeled rows in the file), an Excel or CSV export with the original columns plus `prediction`, `confidence`, `needs_review`, `runner_up`, and a live activity stream.
- **Teach**: a finished workflow becomes training data with one click. A person's review decisions and the file's own label column are labels; everything else goes to the teacher model. The lab that starts keeps the deployed model's task, so the new model is a drop-in successor.

API: `POST /api/workflows` (multipart `file` or `text`, optional `prompt`, `deployment`, `confidence_floor`), `GET /api/workflows/{id}`, `/rows?review=true`, `POST /rows/{row_id}/review`, `GET /export?format=xlsx|csv`, `GET /events` (SSE).

## Where the ideas came from

- `mshumer/gpt-llm-trainer`: describe the task, synthesize pairs, fine-tune a small model. That is the curator plus the LoRA trainer; the held-out evaluation on real examples is the defense against synthetic blind spots.
- `stanfordnlp/dspy`: prompt optimization as a first-class candidate, so "prompting is enough" competes on equal footing with fine-tuning.
- Pioneer Agent and AutoML-Agent: a state machine of task analysis, data curation, training and evaluation driven only by the task description. That is the orchestrator.
- BARRED: synthetic data from a task description plus a few examples.
- `drake69/spendif-ai`: the rules → model → LLM → review cascade.
- `lottiefiles/motion-design-skill` and `AThevon/genjutsu`: one motion personality (corporate: 200/300/450 ms, one easing family), reduced-motion support, the three.js rules the hero follows (shader-driven particles, selective bloom, no per-frame allocation, render only while visible), and the "tells" checklist the marketing page was audited against.

## Tests

```bash
scripts/test-all.sh        # backend pytest, frontend lint + types, Playwright end-to-end
cd backend && uv run pytest
cd frontend && pnpm test:e2e   # starts the Auth emulator, a fresh backend and a production build by itself
```

CI runs the same three jobs on every push (`.github/workflows/ci.yml`). The end-to-end suite signs in through the real Google button against the Firebase Auth emulator, builds a model from a starter prompt and from an uploaded CSV, exercises the playground, API keys and sign-out.

## Security

Google sign-in with server-side ID-token verification, signed httpOnly session cookies with server-side revocation, same-origin CSRF checks, per-IP and per-user rate limits, hashed API keys shown once, upload size caps, strict security headers, and ownership checks on every lab, model and key.

## Layout

```
backend/app/agents      planner, data curator, orchestrator, judge, copilot
backend/app/trainers    classical, cascade, prompted, finetuned strategies
backend/app/datasets    procedural generators for the five financial workflows
backend/app/firebase.py Google sign-in token verification
backend/modal_app.py    GPU fine-tune + serving
frontend/src/app        marketing site, auth, the /app product
frontend/src/components lab visualizations, copilot, 3D scene, recorded-run replay
```
