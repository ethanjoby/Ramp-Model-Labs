"use client";

import { AnimatePresence, motion, useInView } from "motion/react";
import { Crown, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import demo from "./demo-lab.json";
import { AgentFeed, Leaderboard, StageRail } from "@/components/lab/parts";
import { ms, multiple, pct, usd } from "@/lib/format";
import { EASE_OUT } from "@/lib/motion";
import type { LabEvent } from "@/lib/types";
import { applyLabEvent, emptyLabState, type LabState } from "@/lib/useLab";

/*
 * A real lab, replayed from its stored events with its real timing (compressed a little so the
 * quiet stretches don't drag). Nothing here is staged: same reducer as the product's lab page.
 */

const EVENTS = demo.events as LabEvent[];
const SPEED = 0.7;
const MIN_GAP_MS = 160;

const parseTs = (ts: string) => Date.parse(ts.replace(" ", "T") + (/[zZ]|[+-]\d\d:?\d\d$/.test(ts) ? "" : "Z"));

function schedule(): number[] {
  const t0 = parseTs(EVENTS[0].ts);
  let prev = 0;
  return EVENTS.map((e) => {
    const t = Math.max((parseTs(e.ts) - t0) * SPEED, prev + MIN_GAP_MS);
    prev = t;
    return t;
  });
}

export function LiveLab() {
  const frame = useRef<HTMLDivElement>(null);
  const inView = useInView(frame, { once: true, margin: "-120px" });
  const [state, setState] = useState<LabState>(emptyLabState);
  const [playing, setPlaying] = useState(false);
  const [run, setRun] = useState(0);

  useEffect(() => {
    if (!inView) return;
    const times = schedule();
    let i = 0;
    let s = emptyLabState;
    let timer = 0;
    const step = () => {
      s = applyLabEvent(s, EVENTS[i]);
      setState(s);
      i += 1;
      if (i < EVENTS.length) timer = window.setTimeout(step, times[i] - times[i - 1]);
      else setPlaying(false);
    };
    timer = window.setTimeout(() => {
      setState(emptyLabState);
      setPlaying(true);
      step();
    }, 350);
    return () => window.clearTimeout(timer);
  }, [inView, run]);

  const winner = state.result ? state.candidates[state.result.winner] : null;
  const m = winner?.metrics;
  const savings = state.result?.savings;
  const seconds = Math.round((parseTs(EVENTS[EVENTS.length - 1].ts) - parseTs(EVENTS[0].ts)) / 1000);

  return (
    <section id="demo" className="bg-paper">
      <div className="mx-auto max-w-[1240px] px-5 py-24 md:px-8 md:py-32">
        <div className="max-w-[720px]">
          <p className="text-[13px] text-muted">A recorded run</p>
          <h2 className="display mt-4 text-[clamp(36px,4.6vw,60px)]">Watch the agents build one.</h2>
          <p className="mt-6 text-[16px] leading-relaxed text-muted">
            Recorded on 30 October 2026 in offline mode, {seconds} seconds uncut. Nothing below is a mock-up: these are
            the events, candidates and held-out scores one lab produced from the prompt in the header, replayed through
            the same code that renders the product.
          </p>
        </div>

        <div ref={frame} className="mt-12 overflow-hidden rounded-3xl border border-line-soft bg-white shadow-[0_30px_80px_-40px_rgba(12,10,8,0.25)]">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line-soft px-5 py-4">
            <div className="flex min-w-0 items-center gap-3">
              <span className="shrink-0 text-[12.5px] text-muted">Prompt</span>
              <span className="truncate rounded-full bg-ink px-3.5 py-1.5 text-[13px] text-white">{demo.prompt}</span>
            </div>
            <button
              onClick={() => setRun((r) => r + 1)}
              disabled={playing}
              className="inline-flex h-8 items-center gap-1.5 rounded-full border border-line px-3 text-[12.5px] text-ink transition-colors hover:border-ink/40 disabled:opacity-40"
            >
              <RotateCcw className="h-3.5 w-3.5" /> Replay
            </button>
          </div>
          <div className="grid lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
            <div className="border-b border-line-soft p-5 lg:border-b-0 lg:border-r">
              <StageRail stage={state.stage} />
              <div className="-mx-2 mt-6">
                <Leaderboard candidates={Object.values(state.candidates)} target={state.spec?.target_accuracy} compact />
              </div>
              <AnimatePresence>
                {state.result && m && savings && (
                  <motion.div
                    initial={{ opacity: 0, y: 14, scale: 0.985 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: 0.45, ease: EASE_OUT }}
                    className="mt-6 rounded-2xl bg-ink p-5 text-white"
                  >
                    <div className="flex items-center gap-2 text-[12px] text-solar"><Crown className="h-3.5 w-3.5" /> Deployed</div>
                    <div className="mt-1.5 text-[20px] font-light tracking-[-0.02em]">{state.result.winner_name}</div>
                    <dl className="tnum mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                      {[
                        ["Held-out accuracy", pct(m.accuracy)],
                        ["Cost / 1k calls", usd(m.cost_per_1k)],
                        ["vs. frontier model", `${multiple(savings.multiple)} cheaper`],
                        ["Latency p50", ms(m.latency_p50_ms)],
                      ].map(([k, v]) => (
                        <div key={k}>
                          <dt className="text-[11.5px] text-white/45">{k}</dt>
                          <dd className="mt-1 text-[18px] font-light leading-none tracking-[-0.02em]">{v}</dd>
                        </div>
                      ))}
                    </dl>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
            <div className="p-3">
              <AgentFeed events={state.events} live={playing} className="h-[440px]" />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
