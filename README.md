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
| `ANTHROPIC_API_KEY` | Planner for open-ended tasks, synthetic data, prompted candidates (zero-shot Haiku, optimized few-shot program), the frontier reference, the cascade's LLM level, the copilot's natural-language tool use |
| Modal (`uv run modal token new && uv run modal deploy modal_app.py`, then `MODAL_ENABLED=true`) | LoRA fine-tune of Qwen2.5-0.5B on an L4 GPU, plus serving for deployed adapters |

## What a lab does

1. **Planner** turns the prompt into a spec: labels, target accuracy, latency budget, what to optimize for. Uploaded examples define the labels when they exist.
2. **Curator** builds the dataset. Real examples are reserved for evaluation first, then the procedural generators and Claude fill in training data across several styles. Small uploads get label-preserving variants of their training rows (never of test or dev rows). Near-duplicates are dropped, splits are stratified.
3. **Trainer** runs every available strategy in parallel against the same splits:
   - local specialists: Naive Bayes, tuned linear, SVM, tiny neural net
   - cascade: learned rules → small model → LLM → human review
   - prompted: zero-shot Haiku, optimized few-shot prompt program
   - fine-tuned: LoRA on Modal
   - frontier reference (Sonnet), used as the yardstick, never deployed
4. **Judge** keeps candidates that meet the bar, takes the cheapest within tolerance of the best, and explains the pick. Accuracy, latency and cost per 1k calls are all measured, not estimated, except the frontier cost when no key is present.
5. **Deployer** publishes `POST /v1/models/{slug}/predict`. Corrections made in the playground become real labeled data; retraining re-runs the lab with them.

The homepage replays one real run from its stored events (`frontend/src/components/marketing/demo-lab.json`), through the same reducer the product uses.

## Where the ideas came from

- `mshumer/gpt-llm-trainer`: describe the task, synthesize pairs, fine-tune a small model. That is the curator plus the LoRA trainer; the held-out evaluation on real examples is the defense against synthetic blind spots.
- `stanfordnlp/dspy`: prompt optimization as a first-class candidate, so "prompting is enough" competes on equal footing with fine-tuning.
- Pioneer Agent and AutoML-Agent: a state machine of task analysis, data curation, training and evaluation driven only by the task description. That is the orchestrator.
- BARRED: synthetic data from a task description plus a few examples.
- `drake69/spendif-ai`: the rules → model → LLM → review cascade.
- `lottiefiles/motion-design-skill` and `AThevon/genjutsu`: one motion personality (corporate: 200/300/450 ms, one easing family), reduced-motion support, the three.js rules the hero follows (shader-driven particles, selective bloom, no per-frame allocation, render only while visible), and the "tells" checklist the marketing page was audited against.

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
