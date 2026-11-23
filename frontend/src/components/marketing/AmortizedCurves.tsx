"use client";

import { motion, useInView, useReducedMotion } from "motion/react";
import { useRef, useState } from "react";
import demo from "./demo-lab-full.json";
import { usd } from "@/lib/format";
import { EASE_OUT } from "@/lib/motion";
import type { CandidateRow } from "@/lib/types";

/*
 * Training is a fixed cost and serving is a per-call cost, so a fine-tuned model's all-in price per 1k calls is
 * setup / volume + serving: a 1/x curve that falls toward its floor and cuts under any API that charges the same on call one.
 * Prices come from demo-lab-full.json, whose non-classical candidates are illustrative.
 */

const ALL = demo.candidates as unknown as CandidateRow[];
const PICK: { key: string; color: string; dash?: string }[] = [
  { key: "frontier_reference", color: "var(--muted)", dash: "5 4" },
  { key: "zero_shot_kimi_k3_1_pro", color: "var(--spring)", dash: "5 4" },
  { key: "lora_moonlight_16b_a3b", color: "#e0572b" },
  { key: "lora_deepseek_r1_distill_qwen_7b", color: "#a855f7" },
  { key: "lora_qwen", color: "var(--solar-deep)" },
  { key: "svm_specialist", color: "var(--ink)" },
];
const LINES = PICK.map((p) => {
  const c = ALL.find((x) => x.key === p.key)!;
  return { ...p, name: c.name.replace(" · frontier reference", "").replace(" (LoRA)", ""), setup: c.metrics.train_cost ?? 0, per1k: c.metrics.cost_per_1k! };
});

const W = 600, H = 320, P = { l: 52, r: 16, t: 18, b: 40 };
const XMAX = 150_000, YMAX = 0.6;
const sx = (v: number) => P.l + (v / XMAX) * (W - P.l - P.r);
const sy = (v: number) => P.t + (1 - Math.min(v, YMAX) / YMAX) * (H - P.t - P.b);
const allIn = (l: (typeof LINES)[number], calls: number) => l.setup / (calls / 1000) + l.per1k;

function path(l: (typeof LINES)[number]) {
  const pts: string[] = [];
  for (let i = 0; i <= 160; i++) {
    const calls = 1500 + ((XMAX - 1500) * i) / 160;
    pts.push(`${i ? "L" : "M"}${sx(calls).toFixed(1)},${sy(allIn(l, calls)).toFixed(1)}`);
  }
  return pts.join(" ");
}

export function AmortizedCurves() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  const reduced = useReducedMotion();
  const [calls, setCalls] = useState(40_000);
  const move = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    setCalls(Math.round(Math.min(XMAX, Math.max(3000, ((x - P.l) / (W - P.l - P.r)) * XMAX))));
  };
  const rows = LINES.map((l) => ({ ...l, v: allIn(l, calls) })).sort((a, b) => a.v - b.v);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full touch-pan-y" onPointerMove={move} role="img"
        aria-label="All-in cost per 1,000 calls against call volume. Fine-tuned models start expensive and fall like 1 over x below the flat API lines.">
        <defs><clipPath id="amz"><rect x={P.l} y={P.t} width={W - P.l - P.r} height={H - P.t - P.b} /></clipPath></defs>
        {[0, 0.15, 0.3, 0.45, 0.6].map((y) => (
          <g key={y}>
            <line x1={P.l} x2={W - P.r} y1={sy(y)} y2={sy(y)} stroke="var(--line-soft)" />
            <text x={P.l - 8} y={sy(y) + 4} textAnchor="end" className="fill-muted font-mono text-[10px]">${y.toFixed(2)}</text>
          </g>
        ))}
        {[0, 50_000, 100_000, 150_000].map((x) => (
          <text key={x} x={sx(x)} y={H - 20} textAnchor="middle" className="fill-muted font-mono text-[10px]">{x === 0 ? "0" : `${x / 1000}k`}</text>
        ))}
        <text x={W - P.r} y={H - 4} textAnchor="end" className="fill-muted text-[10px]">calls served →</text>
        <text x={P.l} y={10} className="fill-muted text-[10px]">all-in $ per 1k calls, training included</text>
        <g clipPath="url(#amz)">
          {LINES.map((l, i) => (
            <motion.path key={l.key} d={path(l)} fill="none" stroke={l.color} strokeWidth={l.key === "svm_specialist" ? 2.5 : 2} strokeDasharray={l.dash}
              strokeLinecap="round"
              initial={{ pathLength: reduced ? 1 : 0 }} animate={{ pathLength: inView || reduced ? 1 : 0 }}
              transition={{ duration: 1.3, delay: 0.12 * i, ease: EASE_OUT }} />
          ))}
        </g>
        <line x1={sx(calls)} x2={sx(calls)} y1={P.t} y2={H - P.b} stroke="var(--ink)" strokeOpacity=".35" strokeDasharray="2 3" />
        {LINES.map((l) => <circle key={l.key} cx={sx(calls)} cy={sy(allIn(l, calls))} r={4.5} fill={l.color} stroke="white" strokeWidth={1.5} />)}
        <rect x={sx(calls) < W / 2 ? sx(calls) + 8 : sx(calls) - 98} y={P.t + 4} width={90} height={20} rx={6} fill="var(--ink)" />
        <text x={sx(calls) < W / 2 ? sx(calls) + 53 : sx(calls) - 53} y={P.t + 18} textAnchor="middle" className="fill-white font-mono text-[10.5px]">
          {Math.round(calls / 1000)}k calls
        </text>
      </svg>
      <ul className="mt-3 grid grid-cols-1 gap-x-6 gap-y-1.5 sm:grid-cols-2">
        {rows.map((r, i) => (
          <li key={r.key} className="flex items-center justify-between gap-3 text-[12.5px]">
            <span className="flex min-w-0 items-center gap-2">
              <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: r.color }} />
              <span className={`truncate ${i === 0 ? "font-medium" : "text-ink/75"}`}>{r.name}</span>
            </span>
            <span className="tnum font-mono text-[12px]">{usd(r.v)}</span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-[11.5px] leading-snug text-muted">
        Drag across the chart. Fine-tuning costs something up front (GPU time) and almost nothing per call, so its all-in
        price falls like 1/x with volume, then cuts under every API that charges the same on the millionth call as the first.
      </p>
    </div>
  );
}
