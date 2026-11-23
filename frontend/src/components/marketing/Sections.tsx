"use client";

import { motion, useInView } from "motion/react";
import { useMemo, useRef, useState } from "react";
import { ArrowUpRight, Check } from "lucide-react";
import { useEffect } from "react";
import { ButtonLink } from "@/components/ui/Button";
import { Logo } from "@/components/ui/Logo";
import { BOOKING_URL, PROMPT_EVENT } from "@/lib/contact";
import { EASE_OUT } from "@/lib/motion";
import { num, usd } from "@/lib/format";

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
    detail: ["labels: 9 support queues", "target: ≥ 85% held-out accuracy", "optimize: cost"],
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
    body: "Small specialist models, transformer encoders and a small decoder LLM fine-tuned on this machine, LoRA fine-tunes of Qwen3 and DeepSeek on cloud GPUs, zero-shot LLMs, and a DSPy-style optimized prompt program. Fine-tuning is a candidate, not a default.",
    detail: ["linear, SVM, committee", "MiniLM, ModernBERT, Qwen3-0.6B locally", "Qwen3 + DeepSeek LoRA on Modal GPUs", "prompt search over instructions + demos"],
  },
  {
    name: "Judge",
    title: "Picks the cheapest model that clears the bar",
    body: "Every candidate is scored on the same held-out set for accuracy, macro F1, p50 latency, and cost per thousand calls. Then it explains the trade-off in plain English.",
    detail: ["Pareto frontier: accuracy × cost", "frontier model as the yardstick", "a rationale in plain English"],
  },
  {
    name: "Deployer",
    title: "Ships an endpoint, and hands you the weights",
    body: "The winner goes live behind an API key, and the trained weights, spec and scorecard are yours to download. Corrections from your team become labeled data, and one click re-runs the whole lab, here or on your own machines.",
    detail: ["POST /v1/models/{slug}/predict", "download: weights + scorecard", "feedback → retrain"],
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
              The best model for your job is rarely the biggest one for rent.
            </h2>
            <p className="mt-6 max-w-[440px] text-[16px] leading-relaxed text-muted">
              Routing a support ticket, coding a transaction or triaging a claim doesn&apos;t need a trillion
              parameters billed by the token. It needs a model that has seen ten thousand of yours. Five agents build
              that model from open weights, prove it against the frontier on your own held-out rows, and hand it over.
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
  { name: "Small LLM", sub: "Haiku 5.5, zero-shot", per1k: 0.035, tone: "bg-ink/50" },
  { name: "Fine-tuned Qwen3-1.7B", sub: "LoRA on a shared L4 GPU, est.", per1k: 0.009, tone: "bg-solar-deep" },
  { name: "Fine-tuned ModernBERT", sub: "150M params, trained locally, served on CPU", per1k: 0.0013, tone: "bg-solar-deep" },
  { name: "Fine-tuned MiniLM", sub: "22M params, trained locally, served on CPU", per1k: 0.0001, tone: "bg-solar-deep" },
  { name: "Specialist", sub: "Trained by Ascent Labs, CPU", per1k: 0.000012, tone: "bg-solar" },
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
            <label htmlFor="vol" className="text-[13px] text-white/60">Monthly predictions</label>
            <div className="tnum mt-2 text-[44px] font-light tracking-[-0.03em]">{num(calls)}</div>
            <input
              id="vol" type="range" min={4} max={8} step={0.01} value={vol}
              onChange={(e) => setVol(parseFloat(e.target.value))}
              className="mt-5 w-full accent-[var(--solar)]"
            />
            <div className="mt-2 flex justify-between font-mono text-[11px] text-white/60">
              <span>10k</span><span>1M</span><span>100M</span>
            </div>
            <p className="mt-8 max-w-[340px] text-[13px] leading-relaxed text-white/60">
              Estimates from list prices: Claude Sonnet 5.5 ($2/$10 per MTok), Haiku 5.5 ($0.10/$0.50), Modal L4 at
              $0.80/hr serving ~40 req/s, and measured CPU time of a trained specialist or a fine-tuned MiniLM
              (about 7 ms a call). Every lab reports its own measured numbers.
            </p>
          </div>
          <div className="space-y-5">
            {rows.map((r, i) => (
              <div key={r.name}>
                <div className="flex items-baseline justify-between gap-4">
                  <div>
                    <span className="text-[15px]">{r.name}</span>
                    <span className="ml-2 text-[13px] text-white/60">{r.sub}</span>
                  </div>
                  <span className="tnum font-mono text-[15px]">{usd(r.monthly)}<span className="text-white/60">/mo</span></span>
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

/* ---------------------------------------------------------------- ownership */

const OWNERSHIP = [
  {
    title: "Open weights, not a rented API",
    body: "Every candidate we train starts from an open model: MiniLM, ModernBERT, Qwen3, DeepSeek's distills, Moonshot's Moonlight. The weights come out the other side as yours, with no per-token meter attached.",
  },
  {
    title: "Picked for you, not for a vendor",
    body: "Nineteen candidates, from a bag of words to a 16B-parameter fine-tune, compete on the same held-out rows. The judge takes the cheapest one that clears your bar, and shows the frontier model's score next to it so you can see what you are not paying for.",
  },
  {
    title: "Take it with you, keep training it",
    body: "Download the model with its spec and scorecard, run it on your own hardware, and keep teaching it from the corrections your team already makes. If you leave, the model leaves with you.",
  },
];

export function Ownership() {
  return (
    <section id="own" className="border-t border-line bg-white">
      <div className="mx-auto max-w-[1240px] px-5 py-28 md:px-8 md:py-36">
        <Reveal>
          <p className="text-[13px] text-muted">Why open</p>
          <h2 className="display mt-5 max-w-[16ch] text-[clamp(36px,4.6vw,60px)]">Own the intelligence you pay for.</h2>
          <p className="mt-6 max-w-[560px] text-[16px] leading-relaxed text-muted">
            The big AI vendors would like you renting the largest model they have for every row, forever. Most
            problems are solved better, and far cheaper, by a small model trained on your own data, and open
            weights mean nobody can switch it off or raise the price on you.
          </p>
        </Reveal>
        <div className="mt-14 grid gap-6 md:grid-cols-3">
          {OWNERSHIP.map((o, i) => (
            <Reveal key={o.title} delay={i * 0.06}>
              <div className="h-full rounded-3xl border border-line-soft bg-paper p-7">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-ink font-mono text-[12px] text-solar">{i + 1}</div>
                <h3 className="mt-6 text-[20px] tracking-[-0.02em]">{o.title}</h3>
                <p className="mt-3 text-[15px] leading-relaxed text-ink/70">{o.body}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- workflows */

const WORKFLOWS = [
  { name: "Support routing", input: "locked out after resetting my phone, the 2FA code never arrives and I have a demo in an hour", label: "account_access" },
  { name: "Product categorization", input: "Women's trail running shoe, Gore-Tex upper, Vibram sole, size 8, 2 pairs", label: "Footwear > Trail Running" },
  { name: "Claims triage", input: "rear-ended at a stop light; rear bumper damage, no injuries, police report #4471 attached", label: "fast_track" },
  { name: "IT ticket priority", input: "VPN down for the whole sales floor since 9am, reported by the regional director", label: "P1" },
  { name: "Expense categorization", input: "SQ *BLUE BOTTLE COFFEE SAN FRANCISCO CA | $14.50 | memo: coffee w/ candidate", label: "Meals & Entertainment" },
  { name: "Listing moderation", input: "\"authentic\" designer handbag, $40, ships from overseas, DM me to pay outside the app", label: "off_platform_payment" },
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
                  <span className={`text-[20px] tracking-[-0.02em] transition-colors duration-150 ${active === i ? "font-normal text-ink" : "font-light text-ink/50"}`}>
                    {w.name}
                  </span>
                  <ArrowUpRight className={`h-5 w-5 transition-[opacity,transform] duration-200 ${active === i ? "opacity-100" : "-translate-x-1 opacity-0"}`} />
                </button>
              </li>
            ))}
          </ul>
          <div className="rounded-3xl bg-ink p-7 text-white md:p-10">
            <div className="font-mono text-[11px] uppercase tracking-[0.08em] text-white/60">input</div>
            <motion.p
              key={`i${active}`}
              initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, ease: EASE_OUT }}
              className="mt-3 min-h-[84px] font-mono text-[15px] leading-relaxed text-white/85"
            >
              {WORKFLOWS[active].input}
            </motion.p>
            <div className="mt-10 font-mono text-[11px] uppercase tracking-[0.08em] text-white/60">specialist output</div>
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
          <h2 className="display mt-5 text-[clamp(36px,4.2vw,54px)]">Every winner is an endpoint, and a download.</h2>
          <p className="mt-6 max-w-[440px] text-[16px] leading-relaxed text-muted">
            Call it behind a per-account API key, hashed at rest, with per-model request and latency tracking. Or
            download the weights and serve it yourself. Corrections flow back in as training data either way.
          </p>
        </Reveal>
        <Reveal delay={0.08} className="min-w-0">
          <pre className="scroll-thin overflow-x-auto rounded-2xl bg-ink p-6 font-mono text-[12.5px] leading-[1.75] text-white/80">
{`curl $ASCENT_HOST/v1/models/ticket-router-3f2a/predict \\
  -H "Authorization: Bearer $ASCENT_KEY" \\
  -d '{"inputs": ["card declined at checkout, says limit reached"]}'

`}<span className="text-white/60">{`# 200 OK  ·  0.9ms`}</span>{`
{
  "predictions": [{
    "label": `}<span className="text-solar">{`"billing"`}</span>{`,
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

/* ---------------------------------------------------------------- contact */

const CALL_STEPS = [
  { title: "You bring the problem", body: "Whatever it is: an inbox, a queue, a spreadsheet nobody wants to own. Tell us what a good answer looks like." },
  { title: "We build the system around the right model", body: "Candidates from tiny to 16B parameters compete on your held-out rows; the winner is the one that fits your job, not the biggest one for rent." },
  { title: "You walk away owning it", body: "The weights, the spec, the scorecard and the serving setup are yours. Keep training it on your own corrections, on your own machines." },
];

export function Contact() {
  const [prompt, setPrompt] = useState<string | null>(null);
  useEffect(() => {
    const onPrompt = (e: Event) => setPrompt((e as CustomEvent<string>).detail || null);
    window.addEventListener(PROMPT_EVENT, onPrompt);
    return () => window.removeEventListener(PROMPT_EVENT, onPrompt);
  }, []);
  return (
    <section id="contact" className="border-t border-line bg-paper">
      <div className="mx-auto grid max-w-[1240px] gap-12 px-5 py-28 md:grid-cols-[minmax(0,6fr)_minmax(0,6fr)] md:items-center md:px-8 md:py-36">
        <Reveal>
          <p className="text-[13px] text-muted">Book a call</p>
          <h2 className="display mt-5 max-w-[14ch] text-[clamp(36px,4.6vw,60px)]">Tell us what you&apos;re stuck on.</h2>
          <p className="mt-6 max-w-[460px] text-[16px] leading-relaxed text-muted">
            Thirty minutes with the people who build the models. Bring a sample of the rows, or just the sentence you
            typed into the box at the top of this page, and we&apos;ll tell you what we&apos;d build and what it would cost.
          </p>
          {prompt && (
            <motion.div
              data-testid="contact-prompt"
              initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, ease: EASE_OUT }}
              className="mt-8 max-w-[460px] rounded-2xl border border-line-soft bg-white px-5 py-4"
            >
              <div className="font-mono text-[11px] uppercase tracking-[0.08em] text-muted">you described</div>
              <p className="mt-2 text-[15px] leading-relaxed text-ink/85">{prompt}</p>
              <p className="mt-2 text-[12.5px] text-muted">Saved for the call. We&apos;ll start from here.</p>
            </motion.div>
          )}
        </Reveal>
        <Reveal delay={0.08}>
          <div className="rounded-3xl bg-ink p-7 text-white md:p-10">
            <div className="font-mono text-[11px] uppercase tracking-[0.08em] text-white/60">what happens on the call</div>
            <ol className="mt-6 space-y-6">
              {CALL_STEPS.map((step, i) => (
                <li key={step.title} className="grid grid-cols-[32px_1fr] gap-4">
                  <span className="flex h-8 w-8 items-center justify-center rounded-full bg-white/10 font-mono text-[12px] text-solar">{i + 1}</span>
                  <div>
                    <div className="text-[17px] tracking-[-0.015em]">{step.title}</div>
                    <p className="mt-1 text-[14px] leading-relaxed text-white/60">{step.body}</p>
                  </div>
                </li>
              ))}
            </ol>
            <div className="mt-10 flex flex-col gap-3 border-t border-white/10 pt-8 sm:flex-row sm:items-center sm:justify-between">
              <ButtonLink href={BOOKING_URL} target="_blank" rel="noreferrer" variant="solar" size="lg" className="shrink-0 whitespace-nowrap">
                Book a call <ArrowUpRight className="h-4 w-4" />
              </ButtonLink>            </div>
          </div>
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
        <h2 className="display max-w-[14ch] text-[clamp(40px,5.4vw,76px)] text-ink">Stop renting intelligence by the token.</h2>
        <ButtonLink href={BOOKING_URL} target="_blank" rel="noreferrer" variant="ink" size="lg">Book a call</ButtonLink>
      </div>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="bg-ink text-white/60">
      <div className="mx-auto flex max-w-[1240px] flex-col gap-6 px-5 py-10 text-[13px] md:flex-row md:items-center md:justify-between md:px-8">
        <Logo dark />
        <p>Ascent Labs. Models you own, built from open weights. Open source is the future.</p>
      </div>
    </footer>
  );
}
