"use client";

import { animate, motion, useInView, useMotionValue, useReducedMotion, useTransform } from "motion/react";
import { useEffect, useRef, useState } from "react";
import demo from "./bench-real.json";
import { Confusion, ErrorList, ParetoChart } from "@/components/lab/parts";
import { ms, multiple, num, pct, size, usd } from "@/lib/format";
import { EASE_OUT } from "@/lib/motion";
import type { CandidateRow } from "@/lib/types";

/*
 * Every number here is a measurement from one stored lab (Card Fraud Screener, bench-real.json, exported from the
 * product's own database): all 11 candidates ran, none are illustrative. This is a different task from the replay above.
 */

const CANDIDATES = demo.candidates as unknown as CandidateRow[];
const RAN = CANDIDATES.filter((c) => c.status === "done");
const FAMILY_TONE: Record<string, string> = { classical: "bg-ink", finetuned: "bg-solar-deep", prompted: "bg-spring", reference: "bg-muted" };
const WINNER = demo.result.winner;
const SAVINGS = demo.result.savings;
const WIN = RAN.find((c) => c.key === WINNER)!.metrics;
const TARGET = demo.spec.target_accuracy;
const LIMIT = demo.spec.max_latency_ms;

/** A number that counts up from zero the first time it scrolls into view. */
function CountUp({ to, format }: { to: number; format: (v: number) => string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });
  const reduced = useReducedMotion();
  const v = useMotionValue(reduced ? to : 0);
  const text = useTransform(v, format);
  useEffect(() => {
    if (!inView || reduced) return;
    const c = animate(v, to, { duration: 1.4, ease: EASE_OUT });
    return () => c.stop();
  }, [inView, reduced, to, v]);
  return <motion.span ref={ref}>{text}</motion.span>;
}

type Bar = { key: string; label: string; value: number; shown: string; hot?: boolean; hatched?: boolean; tone?: string };

/** Horizontal bars that grow from zero, staggered, when the card scrolls into view. */
function Bars({ rows, max, min = 0, log = false }: { rows: Bar[]; max: number; min?: number; log?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  const reduced = useReducedMotion();
  const frac = (v: number) => {
    const f = log ? (Math.log10(v) - Math.log10(min)) / (Math.log10(max) - Math.log10(min)) : (v - min) / (max - min);
    return Math.min(1, Math.max(0.02, f));
  };
  return (
    <div ref={ref} className="space-y-2">
      {rows.map((r, i) => (
        <div key={r.key}>
          <div className="flex items-baseline justify-between gap-3 text-[11.5px] leading-tight">
            <span className={`truncate ${r.hot ? "font-medium text-ink" : "text-ink/75"}`}>{r.label}</span>
            <span className="tnum shrink-0 font-mono text-[11px] text-ink/70">{r.shown}</span>
          </div>
          <div className="mt-1 h-2 overflow-hidden rounded-full bg-paper-2">
            <motion.div
              className={`h-full origin-left rounded-full ${r.hatched ? "bg-[repeating-linear-gradient(135deg,var(--danger)_0_6px,rgba(194,65,12,0.55)_6px_12px)]" : r.hot ? "bg-solar" : (r.tone ?? "bg-ink")}`}
              style={{ width: `${frac(r.value) * 100}%` }}
              initial={{ scaleX: reduced ? 1 : 0 }}
              animate={{ scaleX: inView ? 1 : reduced ? 1 : 0 }}
              transition={{ duration: 0.9, delay: 0.06 * i, ease: EASE_OUT }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function Card({ title, sub, children, className = "" }: { title: string; sub: string; children: React.ReactNode; className?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.55, ease: EASE_OUT }}
      className={`rounded-xl border border-line-soft bg-white p-4 ${className}`}
    >
      <div className="text-[13px] font-medium">{title}</div>
      <div className="mt-0.5 text-[11px] leading-snug text-muted">{sub}</div>
      <div className="mt-3.5">{children}</div>
    </motion.div>
  );
}

function ParetoCard() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  return (
    <div ref={ref}>
      {/* Mounted on scroll so the frontier line draws and the dots pop in while someone is looking. */}
      {inView ? (
        <ParetoChart candidates={CANDIDATES} pareto={demo.result.pareto} target={TARGET} />
      ) : (
        <div className="h-[220px]" />
      )}
      <p className="mt-1 text-[11px] leading-snug text-muted">Up is more accurate, left is cheaper. Dashed line: the {pct(TARGET, 0)} bar from the prompt.</p>
    </div>
  );
}

const byKey = (rows: CandidateRow[], f: (c: CandidateRow) => number, dir: 1 | -1 = 1) => [...rows].sort((a, b) => dir * (f(a) - f(b)));
const tone = (c: CandidateRow) => ({ hot: c.key === WINNER, tone: FAMILY_TONE[c.family] });

const PHONE_CARDS = 3;  // the rest sit behind a button on small screens, where twelve stacked charts run to 5,000px

export function Benchmarks() {
  const [all, setAll] = useState(false);
  const m = (c: CandidateRow) => c.metrics;
  const secs = (v: number) => (v < 60 ? `${v.toFixed(1)}s` : `${(v / 60).toFixed(1)}min`);
  const labels = Object.entries(demo.dataset.label_counts as Record<string, number>).sort((a, b) => b[1] - a[1]);

  return (
    <section id="benchmarks" className="bg-paper-2/60">
      <div className="mx-auto max-w-[1240px] px-5 py-16 md:px-8 md:py-20">
        <div className="max-w-[720px]">
          <p className="text-[13px] text-muted">The scorecard</p>
          <h2 className="display mt-3 text-[clamp(32px,4vw,48px)]">Every candidate, measured the same way.</h2>
          <p className="mt-4 text-[15px] leading-relaxed text-muted">
            A second lab, Card Fraud Screener: all {RAN.length} candidates it trained or tried, scored on the same{" "}
            {num(demo.dataset.test)} held-out rows. The lab picks the cheapest model that clears your bar and shows its
            work. Every figure is measured, including the Claude and fine-tuned rows.
          </p>
        </div>

        <dl className="tnum mt-8 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-line-soft bg-line-soft lg:grid-cols-4">
          {[
            { k: "Held-out accuracy", v: <CountUp to={WIN.accuracy! * 100} format={(x) => `${x.toFixed(1)}%`} />, s: `bar was ${pct(TARGET, 0)}` },
            { k: "Cheaper than the frontier model", v: <CountUp to={SAVINGS.multiple} format={(x) => multiple(x)} />, s: SAVINGS.frontier_measured ? "frontier cost measured" : "est. from list price" },
            { k: "Latency p50", v: <CountUp to={WIN.latency_p50_ms!} format={(x) => ms(x)} />, s: "served on CPU" },
            { k: "Per million calls", v: <CountUp to={SAVINGS.monthly_at_1m.winner} format={(x) => `$${x.toFixed(2)}`} />, s: `vs $${num(SAVINGS.monthly_at_1m.frontier)} on the frontier model` },
          ].map((t) => (
            <div key={t.k} className="bg-white p-4">
              <dt className="text-[11.5px] text-muted">{t.k}</dt>
              <dd className="mt-1.5 text-[clamp(26px,2.8vw,34px)] font-light leading-none tracking-[-0.03em]">{t.v}</dd>
              <dd className="mt-1.5 text-[11px] text-muted">{t.s}</dd>
            </div>
          ))}
        </dl>

        <div className="mt-4 flex flex-wrap gap-x-5 gap-y-1 text-[12px] text-ink/70">
          {[["classical", "Classical"], ["finetuned", "Fine-tuned"], ["prompted", "Zero-shot / prompted"], ["reference", "Frontier"]].map(([k, l]) => (
            <span key={k} className="flex items-center gap-2"><span className={`h-2.5 w-2.5 rounded-full ${FAMILY_TONE[k]}`} />{l}</span>
          ))}
          <span className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-solar" />Selected</span>
        </div>

        <div className={`mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3 ${all ? "" : "[&>*:nth-child(n+4)]:hidden sm:[&>*:nth-child(n+4)]:block"}`}>
          <Card title="Accuracy against cost" sub="Hover a dot for the model">
            <ParetoCard />
          </Card>

          <Card title="Held-out accuracy" sub="95% confidence interval in brackets. Axis starts at 70%">
            <Bars
              min={0.7} max={1}
              rows={byKey(RAN, (c) => m(c).accuracy!, -1).map((c) => {
                const ci = m(c).accuracy_ci;
                return { key: c.key, label: c.name, value: m(c).accuracy!, shown: `${pct(m(c).accuracy)}${ci ? ` [${(ci[0] * 100).toFixed(0)}–${(ci[1] * 100).toFixed(0)}]` : ""}`, ...tone(c) };
              })}
            />
          </Card>

          <Card title="Winner: right and wrong" sub={`${demo.result.winner_name}. True label down, predicted across`}>
            <div className="flex flex-wrap items-start gap-x-5 gap-y-3">
              <Confusion labels={demo.spec.labels} matrix={WIN.confusion!} />
              <dl className="tnum min-w-[110px] flex-1 space-y-2.5 text-[11.5px]">
                {Object.entries(WIN.per_label_f1!).map(([l, f]) => (
                  <div key={l}>
                    <dt className="flex justify-between text-ink/75"><span>{l} F1</span><span className="font-mono">{f.toFixed(3)}</span></dt>
                    <dd className="mt-1 h-2 overflow-hidden rounded-full bg-paper-2"><div className="h-full rounded-full bg-solar-deep" style={{ width: `${f * 100}%` }} /></dd>
                  </div>
                ))}
              </dl>
            </div>
          </Card>

          <Card title="Cost per 1,000 calls" sub={`Log scale. Hatched bar: the frontier model, ${SAVINGS.frontier_measured ? "measured" : "estimated"}`}>
            <Bars
              min={1e-6} max={1} log
              rows={[
                { key: "frontier", label: "Claude Sonnet 5.5 on every call", value: SAVINGS.frontier_cost_per_1k, shown: usd(SAVINGS.frontier_cost_per_1k), hatched: true },
                ...byKey(RAN, (c) => m(c).cost_per_1k!, -1).map((c) => ({ key: c.key, label: c.name, value: m(c).cost_per_1k!, shown: usd(m(c).cost_per_1k), ...tone(c) })),
              ]}
            />
          </Card>

          <Card title="Median latency" sub={`Typical call, log scale. The prompt set a ${num(LIMIT)}ms limit`}>
            <Bars
              min={0.5} max={3000} log
              rows={byKey(RAN, (c) => m(c).latency_p50_ms!).map((c) => ({ key: c.key, label: c.name, value: m(c).latency_p50_ms!, shown: ms(m(c).latency_p50_ms), ...tone(c) }))}
            />
          </Card>

          <Card title="Tail latency (p95)" sub="Slowest 5% of calls. Log scale">
            <Bars
              min={0.5} max={3000} log
              rows={byKey(RAN, (c) => m(c).latency_p95_ms!).map((c) => ({ key: c.key, label: c.name, value: m(c).latency_p95_ms!, shown: ms(m(c).latency_p95_ms), ...tone(c) }))}
            />
          </Card>

          <Card title="Macro F1" sub="Averages both labels, so a lopsided model can't hide. Axis starts at 50%">
            <Bars
              min={0.5} max={1}
              rows={byKey(RAN, (c) => m(c).macro_f1!, -1).map((c) => ({ key: c.key, label: c.name, value: m(c).macro_f1!, shown: m(c).macro_f1!.toFixed(3), ...tone(c) }))}
            />
          </Card>

          <Card title="Cheaper than the frontier model" sub="Its cost per call divided by this model's. Log scale">
            <Bars
              min={0.5} max={100000} log
              rows={byKey(RAN.filter((c) => c.family !== "reference"), (c) => m(c).cost_per_1k!).map((c) => {
                const x = SAVINGS.frontier_cost_per_1k / m(c).cost_per_1k!;
                return { key: c.key, label: c.name, value: x, shown: multiple(x), ...tone(c) };
              })}
            />
          </Card>

          <Card title="What it still gets wrong" sub={`The ${WIN.errors!.length} held-out rows ${demo.result.winner_name} missed`}>
            <ErrorList errors={WIN.errors!} />
          </Card>

          <Card title="Model size on disk" sub="Hosted models have no files to ship. Log scale">
            <Bars
              min={100} max={1_000_000} log
              rows={byKey(RAN.filter((c) => m(c).size_kb), (c) => m(c).size_kb!).map((c) => ({ key: c.key, label: c.name, value: m(c).size_kb!, shown: size(m(c).size_kb), ...tone(c) }))}
            />
          </Card>

          <Card title="Time to train" sub="Wall clock on this lab's hardware. Log scale">
            <Bars
              min={0.1} max={1000} log
              rows={byKey(RAN.filter((c) => m(c).train_seconds), (c) => m(c).train_seconds!).map((c) => ({ key: c.key, label: c.name, value: Math.max(m(c).train_seconds!, 0.1), shown: secs(m(c).train_seconds!), ...tone(c) }))}
            />
          </Card>

          <Card title="What the lab trained on" sub={`${num(demo.dataset.train)} training rows, ${num(demo.dataset.dev)} dev, ${num(demo.dataset.test)} held out`}>
            <Bars
              max={labels[0][1]}
              rows={labels.map(([l, n]) => ({ key: l, label: l.replaceAll("_", " "), value: n, shown: num(n) }))}
            />
          </Card>
        </div>
        <button
          type="button"
          onClick={() => setAll((v) => !v)}
          className="mt-4 inline-flex h-10 w-full items-center justify-center rounded-full border border-line bg-white text-[13px] font-medium text-ink transition-colors hover:border-ink/40 sm:hidden"
        >
          {all ? "Show fewer charts" : `Show the other ${12 - PHONE_CARDS} charts`}
        </button>
      </div>
    </section>
  );
}
