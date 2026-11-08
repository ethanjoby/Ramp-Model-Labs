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
| `torch` + `transformers` (installed by `uv sync`) | The local language-model candidates: an embedding specialist (frozen bge-small + linear head), fine-tuned MiniLM-L6 (22M) and ModernBERT-base (150M, after a short masked-LM warm-up on your own rows), and Qwen3-0.6B with its top four blocks trained on the Mac's GPU (`LOCAL_LLM_MODEL=` to skip it; it adds about three minutes to a lab) |
| `ANTHROPIC_API_KEY` | Planner for open-ended tasks, synthetic data, rephrased variants of your rows, teacher labels, prompted candidates (zero-shot Haiku, optimized few-shot program), the frontier reference, the cascade's LLM level, the copilot's natural-language tool use |
| `TOGETHER_API_KEY` | Open-weight frontier models as zero-shot competitors, priced from the catalog list with thinking switched off: DeepSeek V4.1 Flash, Kimi K3.1 Pro, MiniMax M3.1 Flash, GLM 5.3 Flash and Qwen3.5 9B (`TOGETHER_MODELS` to change the set). With no Anthropic key, DeepSeek V4.1 Flash runs the agents and Kimi K3.1 Pro is the yardstick |
| Modal (`uv run modal token new && uv run modal deploy modal_app.py`, then `MODAL_ENABLED=true`) | LoRA and QLoRA fine-tunes of open-weight models, each on the GPU its size needs: Qwen3-1.7B and DeepSeek-R1-Distill-Qwen-1.5B on an L4, DeepSeek-R1-Distill-Qwen-7B on an A100-40GB, Moonshot's Moonlight-16B-A3B (the Kimi lab's open MoE) on an A100-80GB, plus serving for deployed adapters. `MODAL_MODELS` takes `org/model[@GPU][:4bit]` entries. Inference scores every label's likelihood instead of generating text, so there are no off-list answers and every prediction carries a confidence. `uv run modal run modal_app.py --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --gpu A100-40GB` is a smoke test that prints accuracy, calibration, timing and GPU cost |

## What a lab does

1. **Planner** turns the prompt into a spec: labels, target accuracy, latency budget, what to optimize for. Uploaded examples define the labels when they exist.
2. **Curator** builds the dataset. Real examples are reserved for evaluation first, then the procedural generators and the LLM fill in training data across several styles. Every generator keeps two disjoint pools of merchants, vendors, parties and phrasings: training rows only ever use the train pool, and an *unseen* evaluation slice is built from the other one, so the score includes merchants the model has never met, the way production does every week. Uploaded rows are weighted up with rewritten variants (truncated names, other casings and separators, typos, amount formats, processor prefixes), and with a model key each row is also rephrased twice in another register and kept only if the model still reads the same label from the rewrite (the Kimi K2 finding: rephrased data beats repeating the same tokens). Near-duplicates are dropped, splits are stratified, and nothing that mirrors a held-out row once digits are folded may train.
3. **Trainer** runs every available strategy in parallel against the same splits:
   - local specialists: Naive Bayes, tuned linear, SVM, tiny neural net, and a committee that soft-votes calibrated versions of them
   - embedding specialist: frozen bge-small sentence embeddings (33M) with a logistic-regression head. Trains in seconds, the pickle is 20 KB, and because the encoder never moves it generalizes from meaning rather than memorized merchant strings
   - real language models, fine-tuned locally: MiniLM-L6 (22M parameters) and ModernBERT-base (150M) with a new classification head, trained end to end on the lab's split on Apple-silicon GPU or CPU (about 20 s and 3 min), evaluated and served on CPU (about 7 ms and 50 ms a call). ModernBERT first spends 45 s on a masked-LM warm-up over the task's own inputs, the laptop-sized version of task-adaptive continued pretraining, and trains at batch 16 on at most 1,200 rows so it fits a 16 GB laptop next to everything else. DeBERTa-v3 was tried and dropped: it produces NaN losses on Apple GPUs
   - a small decoder LLM, fine-tuned locally: Qwen3-0.6B with its top four blocks and a classification head trained on the Mac's GPU (about 2.5 min, 63M trainable parameters); only those weights are saved, a 125 MB delta over the base. Slower to serve (~160 ms) but it brings a language model's prior about what a merchant *is*
   - cascade: learned rules → small model → LLM → human review
   - prompted: zero-shot Haiku, an optimized few-shot prompt program, and with a Together key the open-weight frontier models (DeepSeek V4.1 Flash, Kimi K3.1 Pro, MiniMax M3.1 Flash, GLM 5.3 Flash, Qwen3.5 9B) scored one request per row with thinking off, the shape the endpoint would serve
   - fine-tuned on a GPU: LoRA (rank 8, every linear layer, NEFTune noise) with label-only targets and thinking switched off, because chain-of-thought hurts models under ~10B parameters. The default roster is Qwen3-1.7B and DeepSeek-R1-Distill-Qwen-1.5B on an L4, DeepSeek-R1-Distill-Qwen-7B on an A100-40GB and Moonshot's Moonlight-16B-A3B-Instruct on an A100-80GB (routed experts excluded from the adapters). When a base's bf16 weights would not fit half the card the base is loaded in 4-bit NF4 and the adapters train on top (QLoRA), e.g. `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B@L4`. The flagship DeepSeek V4.1 and Kimi K3.1 weights are 550B–2.8T-parameter MoEs; fine-tuning those needs a multi-H200 node, which is why the roster uses the labs' smaller open checkpoints. Serving cost is priced from the batched throughput measured in the same container at that GPU's list price, and labelled an estimate
   - distillation: when the most accurate candidate is an expensive one (a prompted frontier model, a GPU adapter, the 150M encoder) and leads the cheapest local tier by two points or more at three times the cost, it labels 1,500 unlabeled rows and a MiniLM student trains on the original data plus those labels. That is DeepSeek-R1's small-model recipe (fine-tune on the big model's outputs rather than RL) applied to a classifier; the student is scored on the same held-out set as everyone else, which the teacher never touched
   - frontier reference (Sonnet, or Kimi K3.1 Pro without an Anthropic key), used as the yardstick, never deployed
4. **Judge** keeps candidates that meet the bar, takes the cheapest whose confidence interval overlaps the best one's (costs under a dollar per million calls count as tied, so a 22M model is not passed over for a bag of words to save a hundredth of a cent), breaks ties by how well a model keeps its answers under noisy copies of the same rows, and explains the pick. Accuracy, latency and cost per 1k calls are all measured, not estimated, except the frontier cost when no key is present.
5. **Deployer** publishes `POST /v1/models/{slug}/predict`. Corrections made in the playground become real labeled data; retraining re-runs the lab with them.

### Evaluation

A held-out score that comes from the same generator as the training rows cannot separate models: on the old split every candidate scored 95-97%. The test set now has three slices, each reported on its own:

- **core**: the plain held-out draw, with no planted label noise
- **unseen**: rows built only from the eval pool of merchants, vendors, parties and phrasings, none of which appears in any training row. The headline accuracy covers core + unseen
- **perturbed**: noisy copies of core rows (casing, truncated names, typos, amount formats, processor prefixes, dropped memos). They never count toward the headline; *robustness* is the share of copies still answered right when the original was

Candidates that output a confidence also get **calibration** numbers: expected calibration error, and the **automation rate at 95%**, the share of rows the model can decide on its own while staying at least 95% accurate on them, with the confidence floor that achieves it. That is the number a review queue runs on: two models with the same accuracy can automate very different shares of traffic.

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
- Kimi K2 technical report (arXiv 2507.20534): rephrasing high-value data in several styles, with a fidelity check, beats training more epochs on the same tokens. That is the rephraser and the variant-based weighting of uploaded rows, instead of exact copies.
- DeepSeek-R1 (arXiv 2501.12948, §4.2 and §G.1): small models gain more from being fine-tuned on a strong model's outputs than from RL of their own. That is the distillation stage.
- NEFTune (Jain et al., 2023): noisy embeddings during fine-tuning, a free regularizer for the LoRA recipe. Selective prediction (Geifman & El-Yaniv, 2017) is where the automation-rate metric comes from.
- Together AI's catalog supplies the open-weight competitors and their list prices; reasoning is switched off per their API so the measured cost is the answer alone.
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
backend/app/agents      planner, data curator, teacher, distillation, orchestrator, judge, copilot
backend/app/trainers    classical + committee, embedding, encoder/decoder fine-tunes, cascade, prompted, LoRA strategies
backend/app/datasets    procedural generators (train/eval entity pools) and label-preserving perturbations
backend/app/llm.py      one client for Anthropic and Together, with per-model pricing
backend/app/firebase.py Google sign-in token verification
backend/modal_app.py    GPU fine-tune + serving
frontend/src/app        marketing site, auth, the /app product
frontend/src/components lab visualizations, copilot, 3D scene, recorded-run replay
```
