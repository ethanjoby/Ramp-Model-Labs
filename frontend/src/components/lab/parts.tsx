"use client";

import clsx from "clsx";
import { AnimatePresence, motion } from "motion/react";
import { Check, Crown, Minus, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Badge, FAMILY_LABEL, FamilyDot, Spinner } from "@/components/ui/bits";
import { accuracyBasis, evalRows, humanLabel, ms, num, pct, savingsLabel, savingsTitle, size, usd } from "@/lib/format";
import { EASE, EASE_OUT } from "@/lib/motion";
import type { CandidateRow, Dataset, LabEvent, LabResult, Spec } from "@/lib/types";
import { STAGES, type LabState } from "@/lib/useLab";

/* ------------------------------------------------------------- stage rail */

export function StageRail({ stage, compact = false }: { stage: LabState["stage"]; compact?: boolean }) {
  const idx = stage === "done" ? STAGES.length : STAGES.findIndex((s) => s.key === stage);
  const failed = stage === "failed";
  // The compact rail has no visible labels, so the progress it draws needs a spoken equivalent.
  const progress = failed ? "failed" : stage === "done" ? "complete" : idx >= 0 ? `${STAGES[idx].label}, step ${idx + 1} of ${STAGES.length}` : "starting";
  return (
    <ol className={clsx("flex items-center", compact ? "gap-1.5" : "gap-2")} aria-label={compact ? `Build progress: ${progress}` : undefined}>
      {STAGES.map((s, i) => {
        const state = failed && i >= Math.max(idx, 0) ? "failed" : i < idx ? "done" : i === idx ? "active" : "todo";
        return (
          <li key={s.key} className="flex min-w-0 flex-1 flex-col gap-1.5">
            <div className="relative h-1 overflow-hidden rounded-full bg-ink/[0.08]">
              <motion.div
                className={clsx("absolute inset-y-0 left-0 rounded-full", state === "failed" ? "bg-danger/60" : "bg-ink")}
                initial={false}
                animate={{ width: state === "done" ? "100%" : state === "active" ? "55%" : "0%" }}
                transition={{ duration: 0.5, ease: EASE }}
              />
              {state === "active" && (
                <motion.div
                  className="absolute inset-y-0 w-1/3 rounded-full bg-solar"
                  animate={{ x: ["-100%", "300%"] }}
                  transition={{ duration: 1.4, repeat: Infinity, ease: "linear" }}
                />
              )}
            </div>
            {!compact && (
              <span className={clsx("text-[12px] transition-colors", state === "todo" ? "text-ink/35" : "text-ink")}>
                {s.label}
              </span>
            )}
          </li>
        );
      })}
    </ol>
  );
}

/* ------------------------------------------------------------- agent feed */

const AGENT_TONE: Record<string, string> = {
  director: "bg-ink text-white",
  planner: "bg-spring-light text-[#244a8c]",
  data: "bg-[#efe9ff] text-[#5b3fb5]",
  trainer: "bg-solar text-ink",
  evaluator: "bg-[#e3f3ea] text-smolder",
  selector: "bg-smolder text-solar",
  deployer: "bg-ink text-solar",
};

export function AgentFeed({ events, live, className }: { events: LabEvent[]; live: boolean; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const shown = events.filter((e) => e.message && e.kind !== "result" || (e.kind === "result" && e.agent === "selector"));
  useEffect(() => {
    const el = ref.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 140) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [shown.length]);
  return (
    <div ref={ref} className={clsx("scroll-thin overflow-y-auto", className)}>
      <ul className="space-y-0.5 py-1">
        <AnimatePresence initial={false}>
          {shown.map((e) => (
            <motion.li
              key={e.seq}
              initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, ease: EASE_OUT }}
              className={clsx("grid grid-cols-[76px_1fr] items-start gap-3 rounded-lg px-2 py-1.5",
                e.kind === "stage" && "mt-2", e.kind === "error" && "bg-danger/5")}
            >
              <span className={clsx("mt-px inline-flex h-[20px] items-center justify-center rounded-md px-1.5 font-mono text-[10.5px]",
                AGENT_TONE[e.agent] ?? "bg-paper-2")}>
                {e.agent === "data" ? "curator" : e.agent === "selector" ? "judge" : e.agent}
              </span>
              <span className={clsx("text-[13px] leading-[1.55]",
                e.kind === "stage" ? "font-medium text-ink" : e.kind === "error" ? "text-danger" : "text-ink/75",
                e.kind === "result" && "text-ink")}>
                {e.message}
              </span>
            </motion.li>
          ))}
        </AnimatePresence>
        {live && (
          <li className="grid grid-cols-[76px_1fr] gap-3 px-2 py-1.5">
            <span />
            <span className="caret text-[13px] text-ink/40">working</span>
          </li>
        )}
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------- leaderboard */

function statusIcon(c: CandidateRow) {
  if (c.status === "running") return <Spinner className="h-3.5 w-3.5 text-ink/60" />;
  if (c.status === "queued") return <span className="h-1.5 w-1.5 rounded-full bg-ink/25" />;
  if (c.status === "failed") return <X className="h-3.5 w-3.5 text-danger" />;
  if (c.status === "skipped") return <Minus className="h-3.5 w-3.5 text-ink/30" />;
  return <Check className="h-3.5 w-3.5 text-smolder" />;
}

export function sortCandidates(cands: CandidateRow[]) {
  const rank = (c: CandidateRow) => (c.status === "done" ? 0 : c.status === "running" ? 1 : c.status === "queued" ? 2 : c.status === "failed" ? 3 : 4);
  return [...cands].sort((a, b) =>
    (b.is_winner ? 1 : 0) - (a.is_winner ? 1 : 0) || rank(a) - rank(b) || (b.metrics.accuracy ?? 0) - (a.metrics.accuracy ?? 0));
}

export function Leaderboard({ candidates, target, onSelect, selected, compact = false, evalSource, nEval }: {
  candidates: CandidateRow[]; target?: number; onSelect?: (k: string) => void; selected?: string | null; compact?: boolean;
  evalSource?: string | null; nEval?: number | null;
}) {
  const rows = sortCandidates(candidates).filter((c) => !compact || c.status !== "skipped");
  const accs = rows.map((r) => r.metrics.accuracy ?? 0).filter(Boolean);
  const lo = Math.max(0, Math.min(...accs, target ?? 1) - 0.08);
  const basis = evalRows(evalSource, nEval);
  return (
    <div className="overflow-x-auto scroll-thin">
      <table data-testid="leaderboard" className="w-full min-w-[460px] text-left text-[13px]">
        <thead>
          <tr className="text-[11.5px] text-muted">
            <th className="py-2 pl-3 font-normal">Candidate</th>
            <th className="w-[34%] py-2 font-normal">Accuracy{basis && <span className="text-ink/45"> · {basis}</span>}</th>
            <th className="py-2 pl-5 text-right font-normal">p50</th>
            <th className="py-2 pl-5 pr-3 text-right font-normal">Cost / 1k</th>
          </tr>
        </thead>
        <tbody>
          <AnimatePresence initial={false}>
            {rows.map((c) => {
              const acc = c.metrics.accuracy;
              const w = acc != null ? Math.max(2, ((acc - lo) / (1 - lo)) * 100) : 0;
              const selectable = c.status === "done" && !!onSelect;
              const head = (
                <>
                  <span className="flex w-3.5 shrink-0 justify-center">{statusIcon(c)}</span>
                  <FamilyDot family={c.family} />
                  <span className={clsx("truncate", c.is_winner && "font-medium")}>{c.name}</span>
                  {c.is_winner && <Badge tone="solar"><Crown className="h-3 w-3" />winner</Badge>}
                  {c.family === "reference" && c.status === "done" && <Badge>yardstick</Badge>}
                </>
              );
              return (
                <motion.tr
                  layout="position"
                  key={c.key}
                  transition={{ duration: 0.35, ease: EASE }}
                  onClick={() => selectable && onSelect?.(c.key)}
                  className={clsx("border-t border-line-soft transition-colors",
                    selectable && "cursor-pointer hover:bg-paper/60",
                    selected === c.key && "bg-paper/80",
                    c.status === "skipped" && "text-ink/40")}
                >
                  <td className="py-2.5 pl-3">
                    {selectable ? (
                      // The row click is a convenience; the button is what keyboard and screen-reader users get.
                      <button type="button" aria-pressed={selected === c.key} onClick={(e) => { e.stopPropagation(); onSelect?.(c.key); }}
                        className="flex min-w-0 items-center gap-2.5 rounded-md text-left">
                        {head}
                      </button>
                    ) : (
                      <div className="flex items-center gap-2.5">{head}</div>
                    )}
                    {c.status === "skipped" && !compact && (
                      <div className="mt-0.5 pl-[46px] text-[11.5px] text-ink/40">{c.metrics.skip_reason ?? c.reason}</div>
                    )}
                    {c.status === "failed" && !compact && (
                      <div className="mt-0.5 truncate pl-[46px] text-[11.5px] text-danger/80">{c.metrics.error}</div>
                    )}
                  </td>
                  <td className="py-2.5">
                    {acc != null ? (
                      <div className="flex items-center gap-2.5">
                        <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-ink/[0.07]">
                          <motion.div
                            className={clsx("h-full rounded-full", c.is_winner ? "bg-solar-deep" : "bg-ink/70")}
                            initial={{ width: 0 }} animate={{ width: `${w}%` }} transition={{ duration: 0.6, ease: EASE_OUT }}
                          />
                          {target != null && (
                            <div className="absolute inset-y-0 w-px bg-danger/60" style={{ left: `${((target - lo) / (1 - lo)) * 100}%` }} />
                          )}
                        </div>
                        <span className="tnum w-12 text-right font-mono text-[12.5px]">{pct(acc)}</span>
                      </div>
                    ) : c.status === "running" ? (
                      <div className="skeleton h-1.5 w-full rounded-full" />
                    ) : null}
                  </td>
                  <td className="tnum py-2.5 pl-5 text-right font-mono text-[12.5px]">{ms(c.metrics.latency_p50_ms)}</td>
                  <td className="tnum py-2.5 pl-5 pr-3 text-right font-mono text-[12.5px]">{usd(c.metrics.cost_per_1k)}</td>
                </motion.tr>
              );
            })}
          </AnimatePresence>
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------- pareto chart */

export function ParetoChart({ candidates, pareto, target }: { candidates: CandidateRow[]; pareto: string[]; target?: number }) {
  const pts = candidates.filter((c) => c.status === "done" && c.metrics.accuracy != null && c.metrics.cost_per_1k != null);
  const [hover, setHover] = useState<string | null>(null);
  const W = 560, H = 280, P = { l: 48, r: 16, t: 16, b: 36 };
  const geo = useMemo(() => {
    if (!pts.length) return null;
    const costs = pts.map((p) => Math.log10(Math.max(p.metrics.cost_per_1k!, 1e-7)));
    const accs = pts.map((p) => p.metrics.accuracy!);
    let x0 = Math.floor(Math.min(...costs)), x1 = Math.ceil(Math.max(...costs));
    if (x1 - x0 < 2) { x0 -= 1; x1 += 1; }
    const y0 = Math.max(0, Math.floor((Math.min(...accs, target ?? 1) - 0.04) * 20) / 20);
    const y1 = Math.min(1, Math.ceil((Math.max(...accs) + 0.01) * 20) / 20);
    const sx = (v: number) => P.l + ((Math.log10(Math.max(v, 1e-7)) - x0) / (x1 - x0)) * (W - P.l - P.r);
    const sy = (v: number) => P.t + (1 - (v - y0) / Math.max(0.01, y1 - y0)) * (H - P.t - P.b);
    const xt = Array.from({ length: x1 - x0 + 1 }, (_, i) => x0 + i);
    const yt = Array.from({ length: 5 }, (_, i) => y0 + ((y1 - y0) * i) / 4);
    return { sx, sy, xt, yt };
  }, [pts, target, P.l, P.r, P.t, P.b]);
  if (!geo) return <div className="flex h-[280px] items-center justify-center text-[13px] text-muted">Waiting for candidates…</div>;
  const front = pts.filter((p) => pareto.includes(p.key)).sort((a, b) => a.metrics.cost_per_1k! - b.metrics.cost_per_1k!);
  const path = front.map((p, i) => `${i ? "L" : "M"}${geo.sx(p.metrics.cost_per_1k!)},${geo.sy(p.metrics.accuracy!)}`).join(" ");
  const sup = (n: number) => `10${n < 0 ? "⁻" : ""}${String(Math.abs(n)).split("").map((d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+d]).join("")}`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="group" aria-label="Accuracy versus cost for each candidate">
      {geo.yt.map((y) => (
        <g key={y}>
          <line x1={P.l} x2={W - P.r} y1={geo.sy(y)} y2={geo.sy(y)} stroke="var(--line-soft)" />
          <text x={P.l - 8} y={geo.sy(y) + 4} textAnchor="end" className="fill-muted font-mono text-[10px]">{Math.round(y * 100)}%</text>
        </g>
      ))}
      {geo.xt.map((x) => (
        <text key={x} x={geo.sx(10 ** x)} y={H - 14} textAnchor="middle" className="fill-muted font-mono text-[10px]">${sup(x)}</text>
      ))}
      <text x={W - P.r} y={H - 1} textAnchor="end" className="fill-muted text-[10px]">cost per 1k calls (log) →</text>
      {target != null && (
        <g>
          <line x1={P.l} x2={W - P.r} y1={geo.sy(target)} y2={geo.sy(target)} stroke="var(--danger)" strokeOpacity=".5" strokeDasharray="3 4" />
          <text x={W - P.r} y={geo.sy(target) - 5} textAnchor="end" className="fill-danger text-[10px]">target {Math.round(target * 100)}%</text>
        </g>
      )}
      {front.length > 1 && (
        <motion.path d={path} fill="none" stroke="var(--ink)" strokeOpacity=".35" strokeWidth={1.5}
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.8, ease: EASE_OUT }} />
      )}
      {pts.map((p) => {
        const x = geo.sx(p.metrics.cost_per_1k!), y = geo.sy(p.metrics.accuracy!);
        const fill = p.is_winner ? "var(--solar)" : p.family === "prompted" ? "var(--spring)" : p.family === "reference" ? "var(--muted)" : p.family === "finetuned" ? "var(--solar-deep)" : "var(--ink)";
        const label = `${p.name}: ${pct(p.metrics.accuracy)} accuracy at ${usd(p.metrics.cost_per_1k)} per 1k calls${p.is_winner ? ", selected" : ""}`;
        return (
          <g key={p.key} tabIndex={0} role="img" aria-label={label} className="cursor-default outline-none"
            onMouseEnter={() => setHover(p.key)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(p.key)} onBlur={() => setHover(null)}>
            {p.is_winner && <circle cx={x} cy={y} r={13} fill="var(--solar)" fillOpacity=".25" />}
            {hover === p.key && <circle cx={x} cy={y} r={p.is_winner ? 11 : 9.5} fill="none" stroke="var(--ink)" strokeOpacity=".5" strokeWidth={1.5} />}
            <motion.circle cx={x} cy={y} r={p.is_winner ? 7 : 5.5} fill={fill} stroke={p.is_winner ? "var(--ink)" : "white"} strokeWidth={1.5}
              initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", stiffness: 300, damping: 20 }} />
            {(hover === p.key || p.is_winner) && (
              <g>
                <rect x={Math.min(x + 10, W - 190)} y={y - 30} width={180} height={22} rx={6} fill="var(--ink)" />
                <text x={Math.min(x + 18, W - 182)} y={y - 15} className="fill-white text-[10.5px]">
                  {p.name.length > 24 ? p.name.slice(0, 23) + "…" : p.name} · {pct(p.metrics.accuracy)}
                </text>
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/* ------------------------------------------------------------- winner */

export function WinnerCard({ result, spec, onOpen, evalSource, nEval }: {
  result: LabResult; spec: Spec | null; onOpen?: () => void; evalSource?: string | null; nEval?: number | null;
}) {
  const m = result.winner_metrics;
  const s = result.savings;
  const published = result.meets_bar !== false;
  const rows = evalRows(evalSource, nEval ?? m.n_eval);
  return (
    <motion.div
      initial={{ opacity: 0, y: 14, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.5, ease: EASE_OUT }}
      className="relative overflow-hidden rounded-2xl bg-ink p-6 text-white md:p-7"
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-[12px] text-solar"><Crown className="h-3.5 w-3.5" /> Selected model</div>
          <h3 className="mt-2 text-[24px] font-light tracking-[-0.025em]">{result.winner_name}</h3>
          <p className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-[13px] text-white/50">
            {spec && <span>{spec.name} · {spec.labels.length} labels</span>}
            {!published && (
              <span className="rounded-full bg-danger/20 px-2 py-0.5 text-[11.5px] font-medium text-[#ffb08a]"
                title={spec ? `Missed the bar the plan set: ≥ ${Math.round(spec.target_accuracy * 100)}% accuracy and ≤ ${spec.max_latency_ms}ms p50.` : undefined}>
                Trained, not published
              </span>
            )}
          </p>
        </div>
        {onOpen && (
          <button onClick={onOpen} className="rounded-full bg-solar px-4 py-2 text-[13px] font-medium text-ink transition-colors hover:bg-solar-light">
            Open playground
          </button>
        )}
      </div>
      <div className="mt-7 grid grid-cols-2 gap-6 md:grid-cols-4">
        <Big label="Held-out accuracy" value={pct(m.accuracy)} sub={`${rows ? `on ${rows} · ` : ""}macro F1 ${m.macro_f1?.toFixed(3)}`} />
        <Big label="Cost / 1k calls" value={usd(m.cost_per_1k)} sub={`frontier ${usd(s.frontier_cost_per_1k)}${s.frontier_measured ? "" : " est."}`} title={savingsTitle(s)} />
        <Big label="Cheaper than frontier" value={savingsLabel(s)} sub={`${usd(s.monthly_at_1m.frontier)} → ${usd(s.monthly_at_1m.winner)} per 1M`} accent title={savingsTitle(s)} />
        <Big label="Latency p50" value={ms(m.latency_p50_ms)} sub={m.size_kb ? `model ${size(m.size_kb)}` : `p95 ${ms(m.latency_p95_ms)}`} />
      </div>
      <p className="mt-7 max-w-[760px] border-t border-white/10 pt-5 text-[14px] leading-relaxed text-white/75">{result.rationale}</p>
    </motion.div>
  );
}

function Big({ label, value, sub, accent, title }: { label: string; value: string; sub?: string; accent?: boolean; title?: string }) {
  return (
    <div title={title}>
      <div className="text-[11.5px] text-white/45">{label}</div>
      <div className={clsx("tnum mt-1.5 text-[30px] font-light leading-none tracking-[-0.03em]", accent && "text-solar")}>{value}</div>
      {sub && <div className="tnum mt-1.5 text-[11.5px] text-white/40">{sub}</div>}
    </div>
  );
}

/* ------------------------------------------------------------- dataset */

export function DatasetPanel({ dataset, spec }: { dataset: Dataset; spec: Spec | null }) {
  const counts = Object.entries(dataset.label_counts);
  const max = Math.max(...counts.map(([, v]) => v), 1);
  return (
    <div>
      <div className="flex flex-wrap gap-x-8 gap-y-3">
        {[["train", dataset.train], ["dev", dataset.dev], ["test", dataset.test]].map(([k, v]) => (
          <div key={k as string}>
            <div className="text-[11.5px] text-muted">{k}</div>
            <div className="tnum text-[22px] font-light tracking-[-0.02em]">{num(v as number)}</div>
          </div>
        ))}
        <div>
          <div className="text-[11.5px] text-muted">sources</div>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {Object.entries(dataset.sources).map(([k, v]) => <Badge key={k}>{k} · {num(v)}</Badge>)}
          </div>
        </div>
      </div>
      <p className="mt-4 text-[12.5px] text-muted">Evaluated on {dataset.eval_source}.</p>
      <div className="mt-5 space-y-1.5">
        {counts.map(([label, v], i) => (
          <div key={label} className="grid grid-cols-[minmax(0,180px)_1fr_44px] items-center gap-3 text-[12.5px]">
            <span className="truncate text-ink/75" title={spec?.label_descriptions?.[label]}>{humanLabel(label)}</span>
            <div className="h-1.5 overflow-hidden rounded-full bg-ink/[0.06]">
              <motion.div className="h-full rounded-full bg-ink/60" initial={{ width: 0 }} animate={{ width: `${(v / max) * 100}%` }}
                transition={{ duration: 0.5, ease: EASE_OUT, delay: Math.min(i * 0.025, 0.4) }} />
            </div>
            <span className="tnum text-right font-mono text-[11.5px] text-muted">{v}</span>
          </div>
        ))}
      </div>
      <div className="mt-6">
        <div className="text-[11.5px] text-muted">Samples</div>
        <ul className="mt-2 divide-y divide-line-soft rounded-xl border border-line-soft">
          {dataset.samples.slice(0, 6).map((s, i) => (
            <li key={i} className="flex items-start justify-between gap-4 px-3 py-2">
              <span className="min-w-0 font-mono text-[11.5px] leading-relaxed text-ink/80">{s.text}</span>
              <span className="shrink-0 text-right">
                <Badge tone="neutral">{humanLabel(s.label)}</Badge>
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------- confusion + errors */

export function Confusion({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const max = Math.max(...matrix.flat(), 1);
  const short = (l: string) => { const h = humanLabel(l); return h.length > 14 ? h.slice(0, 13) + "…" : h; };
  return (
    <div className="overflow-x-auto scroll-thin">
      <table className="border-separate border-spacing-[2px] text-[10.5px]">
        <thead>
          <tr>
            <th />
            {labels.map((l) => (
              <th key={l} className="h-24 w-7 align-bottom font-normal text-muted">
                <div className="mx-auto w-4 whitespace-nowrap [writing-mode:vertical-rl] rotate-180 text-left">{short(l)}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="whitespace-nowrap pr-2 text-right font-normal text-muted">{short(labels[i])}</th>
              {row.map((v, j) => {
                const a = v / max;
                const diag = i === j;
                return (
                  <td key={j} title={`${labels[i]} → ${labels[j]}: ${v}`}
                    className="tnum h-7 w-7 rounded-[4px] text-center font-mono"
                    style={{
                      background: v === 0 ? "var(--paper)" : diag ? `rgba(201,214,12,${0.25 + a * 0.75})` : `rgba(194,65,12,${0.15 + a * 0.85})`,
                      color: diag && a > 0.5 ? "var(--ink)" : !diag && a > 0.4 ? "white" : "var(--ink)",
                    }}>
                    {v || ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-muted">Rows: true label. Columns: predicted.</p>
    </div>
  );
}

export function ErrorList({ errors }: { errors: { text: string; expected: string; predicted: string }[] }) {
  if (!errors?.length) return <p className="text-[13px] text-muted">No errors on the held-out set.</p>;
  return (
    <ul className="divide-y divide-line-soft">
      {errors.slice(0, 8).map((e, i) => (
        <li key={i} className="py-2.5">
          <div className="font-mono text-[11.5px] leading-relaxed text-ink/80">{e.text}</div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11.5px]">
            <span className="text-muted">expected</span><Badge tone="neutral">{humanLabel(e.expected)}</Badge>
            <span className="text-muted">got</span><Badge tone="danger">{humanLabel(e.predicted)}</Badge>
          </div>
        </li>
      ))}
    </ul>
  );
}

export function SpecSummary({ spec }: { spec: Spec }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Badge tone="ink">≥ {Math.round(spec.target_accuracy * 100)}% accuracy</Badge>
      <Badge tone="neutral">≤ {spec.max_latency_ms}ms</Badge>
      <Badge tone="spring">optimize {spec.priority}</Badge>
      {spec.template && <Badge tone="neutral">generator: {spec.template.replace(/_/g, " ")}</Badge>}
      <span className="mx-1 h-3 w-px bg-line" />
      {spec.labels.slice(0, 8).map((l) => <Badge key={l}>{humanLabel(l)}</Badge>)}
      {spec.labels.length > 8 && <span className="text-[11.5px] text-muted">+{spec.labels.length - 8} more</span>}
    </div>
  );
}

export { FAMILY_LABEL };
