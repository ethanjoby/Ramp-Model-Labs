"use client";

import { motion, useInView } from "motion/react";
import { useMemo, useRef, useState } from "react";
import { ArrowUpRight, Check } from "lucide-react";
import { ButtonLink } from "@/components/ui/Button";
import { Logo } from "@/components/ui/Logo";
import { EASE_OUT } from "@/lib/motion";
import { usd } from "@/lib/format";

function Reveal({ children, delay = 0, className }: { children: React.ReactNode; delay?: number; className?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.55, ease: EASE_OUT, delay }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

/* ---------------------------------------------------------------- pipeline */

const AGENTS = [
  {
    name: "Planner",
    title: "Turns your sentence into a spec",
    body: "Infers the labels, what one input looks like, the accuracy bar, latency budget, and whether you care most about cost, speed, or quality.",
    detail: ["labels: 12 GL categories", "target: ≥ 85% held-out accuracy", "optimize: cost"],
  },
  {
    name: "Curator",
    title: "Builds the dataset",
    body: "Blends your examples with procedurally generated and LLM-synthesized ones, dedupes, and reserves your real data for the test set so synthetic blind spots can't inflate the score.",
    detail: ["2,049 train, 151 dev, 200 test", "sources: yours, procedural, Claude", "near-duplicates removed"],
  },
  {
    name: "Trainer",
    title: "Runs competing strategies in parallel",
    body: "Small specialist models, a LoRA fine-tune of an open model on GPUs, zero-shot LLMs, and a DSPy-style optimized prompt program. Fine-tuning is a candidate, not a default.",
    detail: ["linear · SVM · tiny MLP", "Qwen2.5 LoRA on Modal L4", "prompt search over instructions + demos"],
  },
  {
    name: "Judge",
    title: "Picks the cheapest model that clears the bar",
    body: "Every candidate is scored on the same held-out set for accuracy, macro F1, p50 latency, and cost per thousand calls. Then it explains the trade-off in plain English.",
    detail: ["Pareto frontier: accuracy × cost", "frontier model as the yardstick", "rationale you can show your CFO"],
  },
  {
    name: "Deployer",
    title: "Ships an endpoint, then keeps learning",
    body: "The winner goes live behind an API key. Corrections from your team become labeled data, and one click re-runs the whole lab.",
    detail: ["POST /v1/models/{slug}/predict", "confidence + top-3 labels", "feedback → retrain"],
  },
];

export function Pipeline() {
  return (
    <section id="how" className="bg-paper">
      <div className="mx-auto grid max-w-[1240px] gap-14 px-5 py-28 md:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] md:px-8 md:py-36">
        <div className="md:sticky md:top-24 md:self-start">
          <Reveal>
            <p className="text-[13px] text-muted">How it works</p>
            <h2 className="display mt-5 text-[clamp(36px,4.6vw,60px)]">
              General models are overqualified for most finance work.
            </h2>
            <p className="mt-6 max-w-[440px] text-[16px] leading-relaxed text-muted">
              Categorizing a card swipe doesn&apos;t need a trillion parameters. It needs a model that has seen ten
              thousand card swipes. Five agents build that model for you, end to end.
            </p>
          </Reveal>
        </div>
        <ol className="relative">
          <div className="absolute bottom-6 left-[15px] top-6 w-px bg-line" aria-hidden />
          {AGENTS.map((a, i) => (
            <Reveal key={a.name} delay={i * 0.04}>
              <li className="relative grid grid-cols-[32px_1fr] gap-5 pb-12 last:pb-0">
                <span className="relative z-10 mt-0.5 flex h-8 w-8 items-center justify-center rounded-full border border-line bg-white text-[11px] font-medium">
                  {a.name[0]}
                </span>
                <div>
                  <div className="text-[13px] text-muted">{a.name}</div>
                  <h3 className="mt-1 text-[22px] font-normal tracking-[-0.02em]">{a.title}</h3>
                  <p className="mt-3 max-w-[560px] text-[15px] leading-relaxed text-ink/70">{a.body}</p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {a.detail.map((d) => (
                      <span key={d} className="rounded-md border border-line-soft bg-white px-2.5 py-1 font-mono text-[11.5px] text-ink/70">
                        {d}
                      </span>
                    ))}
                  </div>
                </div>
              </li>
            </Reveal>
          ))}
        </ol>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- cost */

// Per-1k-call prices for a ~60-token classification request with a ~250-token instruction.
const TIERS = [
  { name: "Frontier LLM", sub: "Sonnet-class, every request", per1k: 0.74, tone: "bg-ink/80" },
  { name: "Small LLM", sub: "Haiku-class, zero-shot", per1k: 0.37, tone: "bg-ink/50" },
  { name: "Fine-tuned 0.5B", sub: "LoRA on a shared L4 GPU", per1k: 0.0056, tone: "bg-solar-deep" },
  { name: "Specialist", sub: "Trained by Model Labs, CPU", per1k: 0.000012, tone: "bg-solar" },
];

export function CostCalc() {
  const [vol, setVol] = useState(6); // log10 of monthly calls
  const calls = Math.round(10 ** vol);
  const rows = useMemo(() => TIERS.map((t) => ({ ...t, monthly: (t.per1k * calls) / 1000 })), [calls]);
  const max = rows[0].monthly;
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="cost" className="dark bg-ink text-white">
      <div className="mx-auto max-w-[1240px] px-5 py-28 md:px-8 md:py-36">
        <Reveal>
          <p className="text-[13px] text-solar">Cost at scale</p>
          <h2 className="display mt-5 max-w-[16ch] text-[clamp(36px,4.6vw,60px)]">
            The same answer, for a rounding error.
          </h2>
        </Reveal>

        <div ref={ref} className="mt-14 grid gap-12 md:grid-cols-[minmax(0,4fr)_minmax(0,8fr)]">
          <div>
            <label htmlFor="vol" className="text-[13px] text-white/55">Monthly predictions</label>
            <div className="tnum mt-2 text-[44px] font-light tracking-[-0.03em]">{calls.toLocaleString()}</div>
            <input
              id="vol" type="range" min={4} max={8} step={0.01} value={vol}
              onChange={(e) => setVol(parseFloat(e.target.value))}
              className="mt-5 w-full accent-[var(--solar)]"
            />
            <div className="mt-2 flex justify-between font-mono text-[11px] text-white/40">
              <span>10k</span><span>1M</span><span>100M</span>
            </div>
            <p className="mt-8 max-w-[340px] text-[13px] leading-relaxed text-white/50">
              Estimates from list prices: Claude Sonnet 5.5 ($2/$10 per MTok), Haiku 4.5 ($1/$5), Modal L4 at
              $0.80/hr serving ~40 req/s, and measured CPU time of a trained specialist. Every lab reports its own
              measured numbers.
            </p>
          </div>
          <div className="space-y-5">
            {rows.map((r, i) => (
              <div key={r.name}>
                <div className="flex items-baseline justify-between gap-4">
                  <div>
                    <span className="text-[15px]">{r.name}</span>
                    <span className="ml-2 text-[13px] text-white/45">{r.sub}</span>
                  </div>
                  <span className="tnum font-mono text-[15px]">{usd(r.monthly)}<span className="text-white/40">/mo</span></span>
                </div>
                <div className="mt-2 h-2.5 overflow-hidden rounded-full bg-white/[0.07]">
                  <motion.div
                    className={`h-full rounded-full ${r.tone}`}
                    initial={{ width: 0 }}
                    animate={{ width: inView ? `${Math.max(0.6, (r.monthly / max) * 100)}%` : 0 }}
                    transition={{ duration: 0.8, ease: EASE_OUT, delay: inView ? i * 0.08 : 0 }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- workflows */

const WORKFLOWS = [
  { name: "Expense categorization", input: "SQ *BLUE BOTTLE COFFEE SAN FRANCISCO CA | $14.50 | memo: coffee w/ candidate", label: "Meals & Entertainment" },
  { name: "Fraud screening", input: "card-not-present $2,499.00 at GIFTCARDMALL 03:12 local; new device fingerprint; 4 declined attempts in last 10 min", label: "fraud" },
  { name: "Support routing", input: "netsuite sync is failing with an error on GL codes", label: "accounting_integration" },
  { name: "Transaction reconciliation", input: "BANK ACH GLOBEX CORP $4,210.00 REF 58213 | GL Globex Corp $4,350.00 REF 58213 | diff $140.00; 1d apart; short pay", label: "amount_mismatch" },
  { name: "AP invoice triage", input: "Northwind Supply INV-73944 $3,867.82 PO-1464; amount + vendor match payment 3 days ago", label: "duplicate" },
  { name: "Anything you can label", input: "Upload a CSV of text and labels, or just describe the categories.", label: "your_labels" },
];

export function Workflows() {
  const [active, setActive] = useState(0);
  return (
    <section id="workflows" className="bg-paper">
      <div className="mx-auto max-w-[1240px] px-5 py-28 md:px-8 md:py-36">
        <Reveal>
          <p className="text-[13px] text-muted">Workflows</p>
          <h2 className="display mt-5 max-w-[18ch] text-[clamp(36px,4.6vw,60px)]">
            Built for the boring, high-volume decisions.
          </h2>
        </Reveal>
        <div className="mt-14 grid gap-8 md:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
          <ul className="border-t border-line">
            {WORKFLOWS.map((w, i) => (
              <li key={w.name} className="border-b border-line">
                <button
                  onMouseEnter={() => setActive(i)}
                  onFocus={() => setActive(i)}
                  onClick={() => setActive(i)}
                  className="flex w-full items-center justify-between py-5 text-left"
                >
                  <span className={`text-[20px] tracking-[-0.02em] transition-colors duration-150 ${active === i ? "text-ink" : "text-ink/40"}`}>
                    {w.name}
                  </span>
                  <ArrowUpRight className={`h-5 w-5 transition-[opacity,transform] duration-200 ${active === i ? "opacity-100" : "-translate-x-1 opacity-0"}`} />
                </button>
              </li>
            ))}
          </ul>
          <div className="rounded-3xl bg-ink p-7 text-white md:p-10">
            <div className="font-mono text-[11px] uppercase tracking-[0.08em] text-white/40">input</div>
            <motion.p
              key={`i${active}`}
              initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, ease: EASE_OUT }}
              className="mt-3 min-h-[84px] font-mono text-[15px] leading-relaxed text-white/85"
            >
              {WORKFLOWS[active].input}
            </motion.p>
            <div className="mt-10 font-mono text-[11px] uppercase tracking-[0.08em] text-white/40">specialist output</div>
            <motion.div
              key={`o${active}`}
              initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.3, ease: EASE_OUT, delay: 0.12 }}
              className="mt-3 inline-flex items-center gap-2 rounded-full bg-solar px-4 py-2 text-[15px] font-medium text-ink"
            >
              <Check className="h-4 w-4" /> {WORKFLOWS[active].label}
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- api */

export function ApiSection() {
  return (
    <section id="api" className="border-t border-line bg-white">
      <div className="mx-auto grid max-w-[1240px] grid-cols-[minmax(0,1fr)] gap-12 px-5 py-28 md:grid-cols-2 md:px-8 md:py-32">
        <Reveal>
          <p className="text-[13px] text-muted">API</p>
          <h2 className="display mt-5 text-[clamp(36px,4.2vw,54px)]">Every winner is an endpoint.</h2>
          <p className="mt-6 max-w-[440px] text-[16px] leading-relaxed text-muted">
            Scoped API keys, hashed at rest. Per-model request and latency tracking. Corrections flow back in as
            training data so the next version is better than this one.
          </p>
        </Reveal>
        <Reveal delay={0.08} className="min-w-0">
          <pre className="scroll-thin overflow-x-auto rounded-2xl bg-ink p-6 font-mono text-[12.5px] leading-[1.75] text-white/80">
{`curl $RML_HOST/v1/models/expense-categorizer-3f2a/predict \\
  -H "Authorization: Bearer $RML_KEY" \\
  -d '{"inputs": ["UBER *TRIP SAN FRANCISCO CA | $23.10"]}'

`}<span className="text-white/40">{`# 200 OK  ·  0.9ms`}</span>{`
{
  "predictions": [{
    "label": `}<span className="text-solar">{`"Ground Transportation"`}</span>{`,
    "confidence": 0.97,
    "top": [ ... ]
  }]
}`}
          </pre>
        </Reveal>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- cta + footer */

export function FinalCta() {
  return (
    <section className="bg-solar">
      <div className="mx-auto flex max-w-[1240px] flex-col items-start justify-between gap-8 px-5 py-20 md:flex-row md:items-end md:px-8 md:py-24">
        <h2 className="display max-w-[14ch] text-[clamp(40px,5.4vw,76px)] text-ink">Stop renting a genius for data entry.</h2>
        <ButtonLink href="/signup" variant="ink" size="lg">Build your first model</ButtonLink>
      </div>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="bg-ink text-white/50">
      <div className="mx-auto flex max-w-[1240px] flex-col gap-6 px-5 py-10 text-[13px] md:flex-row md:items-center md:justify-between md:px-8">
        <Logo dark />
        <p>Built at the Ramp hackathon. Specialized AI for financial workflows.</p>
      </div>
    </footer>
  );
}
