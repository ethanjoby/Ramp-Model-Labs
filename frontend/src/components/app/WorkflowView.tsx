"use client";

import clsx from "clsx";
import Link from "next/link";
import { useParams } from "next/navigation";
import { motion } from "motion/react";
import { ArrowLeft, Check, Download, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { AgentFeed } from "@/components/lab/parts";
import { Badge, Card, Stat } from "@/components/ui/bits";
import { api } from "@/lib/api";
import { humanLabel, ms, pct, usd } from "@/lib/format";
import { EASE } from "@/lib/motion";
import { WF_STAGES, useWorkflow, type WorkflowRow } from "@/lib/useWorkflow";

function Rail({ stage }: { stage: string }) {
  const idx = stage === "failed" ? -1 : WF_STAGES.findIndex((s) => s.key === stage);
  return (
    <ol className="flex items-center gap-2" aria-label="Workflow progress">
      {WF_STAGES.map((s, i) => {
        const state = stage === "failed" ? "failed" : i < idx || stage === "done" ? "done" : i === idx ? "active" : "todo";
        return (
          <li key={s.key} className="flex min-w-0 flex-1 flex-col gap-1.5">
            <div className="relative h-1 overflow-hidden rounded-full bg-ink/[0.08]">
              <motion.div className={clsx("absolute inset-y-0 left-0 rounded-full", state === "failed" ? "bg-danger/60" : "bg-ink")}
                initial={false} animate={{ width: state === "done" ? "100%" : state === "active" ? "55%" : "0%" }} transition={{ duration: 0.5, ease: EASE }} />
              {state === "active" && <motion.div className="absolute inset-y-0 w-1/3 rounded-full bg-solar" animate={{ x: ["-100%", "300%"] }} transition={{ duration: 1.4, repeat: Infinity, ease: "linear" }} />}
            </div>
            <span className={clsx("text-[12px]", state === "todo" ? "text-ink/35" : "text-ink")}>{s.label}</span>
          </li>
        );
      })}
    </ol>
  );
}

export function WorkflowView() {
  const { id } = useParams<{ id: string }>();
  const { wf, events, error, stage, live } = useWorkflow(id);
  const [rows, setRows] = useState<WorkflowRow[]>([]);
  const [onlyReview, setOnlyReview] = useState(false);
  const [rowErr, setRowErr] = useState<string | null>(null);

  const loadRows = useCallback(() => {
    if (!id) return;
    api<WorkflowRow[]>(`/api/workflows/${id}/rows?review=${onlyReview}&limit=300`).then(setRows).catch((e) => setRowErr(e.message));
  }, [id, onlyReview]);

  useEffect(() => { loadRows(); }, [loadRows, wf?.rows_done, wf?.status]);

  const review = async (row: WorkflowRow, label: string) => {
    if (!label) return;
    try {
      const updated = await api<WorkflowRow>(`/api/workflows/${id}/rows/${row.id}/review`, { method: "POST", json: { label } });
      setRows((rs) => rs.map((r) => (r.id === row.id ? updated : r)));
    } catch (e) {
      setRowErr(e instanceof Error ? e.message : "Couldn't save that decision.");
    }
  };

  const s = wf?.summary ?? {};
  const dist = useMemo(() => Object.entries(s.distribution ?? {}), [s.distribution]);
  const distMax = Math.max(1, ...dist.map(([, v]) => v));
  const labels = wf?.deployment?.labels ?? [];

  if (!wf && error) {
    return <div className="mx-auto max-w-[1180px] px-5 py-8 md:px-8"><p role="alert" className="text-[14px] text-danger">{error}</p></div>;
  }

  return (
    <div className="mx-auto max-w-[1180px] px-5 py-8 md:px-8">
      <Link href="/app/workflows" className="inline-flex items-center gap-1.5 text-[13px] text-muted transition-colors hover:text-ink"><ArrowLeft className="h-3.5 w-3.5" /> Workflows</Link>
      <div className="mt-3 flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 max-w-[780px]">
          <h1 className="truncate text-[30px] font-light tracking-[-0.03em]">{wf?.name ?? "Workflow"}</h1>
          <p className="mt-1 text-[14px] text-muted">
            {wf?.prompt || `${wf?.source_kind ?? ""} upload`}
            {wf?.deployment && <> · model <Link href={`/app/models/${wf.deployment.slug}`} className="underline underline-offset-4 hover:text-ink">{wf.deployment.name}</Link></>}
            {wf?.lab_id && <> · <Link href={`/app/labs/${wf.lab_id}`} className="underline underline-offset-4 hover:text-ink">built in a lab</Link></>}
          </p>
        </div>
        {wf?.status === "completed" && (
          <div className="flex gap-2">
            <a href={`/api/workflows/${wf.id}/export?format=xlsx`} className="inline-flex h-9 items-center gap-2 rounded-full bg-ink px-4 text-[13px] font-medium text-white hover:bg-ink-3"><Download className="h-3.5 w-3.5" /> Excel</a>
            <a href={`/api/workflows/${wf.id}/export?format=csv`} className="inline-flex h-9 items-center gap-2 rounded-full border border-line bg-white px-4 text-[13px] text-ink hover:border-ink/40">CSV</a>
          </div>
        )}
      </div>
      <div className="mt-6"><Rail stage={stage} /></div>
      {wf?.error && <p role="alert" className="mt-4 rounded-xl bg-danger/5 px-4 py-3 text-[13.5px] text-danger">{wf.error}</p>}

      {wf && (
        <Card className="mt-6 grid grid-cols-2 gap-6 p-5 md:grid-cols-5">
          <Stat label="Rows" value={wf.rows_total} hint={s.parse ? `${s.parse.kind}${s.parse.text_column ? ` · column "${s.parse.text_column}"` : ""}` : undefined} />
          <Stat label="Decided automatically" value={wf.status === "completed" ? (s.auto ?? 0) : `${wf.rows_done}/${wf.rows_total}`} hint={s.mean_confidence != null ? `mean confidence ${pct(s.mean_confidence, 0)}` : undefined} />
          <Stat label="Needs a person" value={wf.rows_review} hint={`below ${Math.round(wf.confidence_floor * 100)}% confidence`} />
          <Stat label="Model cost" value={s.cost_usd != null ? usd(s.cost_usd) : "—"} hint={s.frontier_cost_usd != null ? `frontier ${usd(s.frontier_cost_usd)}${s.frontier_measured ? "" : " est."}` : undefined} />
          <Stat label={s.labeled_rows ? "Accuracy on your labels" : "Latency"} value={s.labeled_rows ? pct(s.accuracy_on_labeled) : s.latency_ms != null ? ms(s.latency_ms) : "—"} hint={s.labeled_rows ? `${s.labeled_rows} labeled rows in the file` : undefined} />
        </Card>
      )}

      <div className="mt-6 grid items-start gap-6 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-6">
          {dist.length > 0 && (
            <Card className="p-5">
              <h2 className="text-[13px] font-medium">Decisions</h2>
              <div className="mt-3 space-y-1.5">
                {dist.map(([label, v]) => (
                  <div key={label} className="grid grid-cols-[minmax(0,180px)_1fr_44px] items-center gap-3 text-[12.5px]">
                    <span className="truncate text-ink/75">{humanLabel(label)}</span>
                    <div className="h-1.5 overflow-hidden rounded-full bg-ink/[0.06]"><motion.div className="h-full rounded-full bg-ink/60" initial={{ width: 0 }} animate={{ width: `${(v / distMax) * 100}%` }} /></div>
                    <span className="tnum text-right font-mono text-[11.5px] text-muted">{v}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}
          <Card>
            <div className="flex items-center justify-between border-b border-line-soft px-4 py-3">
              <h2 className="text-[13px] font-medium">Rows</h2>
              <div className="flex gap-1 rounded-lg bg-paper p-0.5 text-[12px]">
                {[["All", false], ["Needs review", true]].map(([t, v]) => (
                  <button key={String(t)} onClick={() => setOnlyReview(v as boolean)} aria-pressed={onlyReview === v}
                    className={clsx("rounded-md px-2.5 py-1 transition-colors", onlyReview === v ? "bg-white text-ink shadow-[0_1px_2px_rgba(12,10,8,0.06)]" : "text-muted hover:text-ink")}>{t}</button>
                ))}
              </div>
            </div>
            {rowErr && <p role="alert" className="px-4 pt-3 text-[13px] text-danger">{rowErr}</p>}
            <ul className="divide-y divide-line-soft">
              {rows.map((r) => {
                const final = r.reviewed_label ?? r.label;
                return (
                  <li key={r.id} className={clsx("grid gap-2 px-4 py-2.5 md:grid-cols-[minmax(0,1fr)_auto] md:items-center md:gap-4", r.needs_review && !r.reviewed_label && "bg-[#fffbe6]")}>
                    <span className="min-w-0 break-words font-mono text-[12px] text-ink/80">{r.text}</span>
                    <span className="flex flex-wrap items-center gap-2">
                      {final ? <Badge tone={r.reviewed_label ? "smolder" : r.needs_review ? "danger" : "solar"}>{r.reviewed_label && <Check className="h-3 w-3" />}{humanLabel(final)}</Badge> : <Badge tone="neutral">pending</Badge>}
                      {r.confidence != null && <span className="tnum font-mono text-[11.5px] text-muted">{Math.round(r.confidence * 100)}%</span>}
                      {r.truth && (r.truth === final ? <Check className="h-3.5 w-3.5 text-smolder" aria-label="matches your label" /> : <span className="flex items-center gap-1 text-[11.5px] text-danger"><X className="h-3.5 w-3.5" />{humanLabel(r.truth)}</span>)}
                      {labels.length > 0 && !r.reviewed_label && r.label && (
                        <select aria-label={`Decide the label for ${r.text}`} value="" onChange={(e) => void review(r, e.target.value)}
                          className="h-[26px] max-w-[140px] rounded-lg border border-line-soft bg-white px-1.5 text-[12px] text-muted">
                          <option value="">{r.needs_review ? "Decide…" : "Correct…"}</option>
                          {labels.map((l) => <option key={l} value={l}>{humanLabel(l)}</option>)}
                        </select>
                      )}
                    </span>
                  </li>
                );
              })}
              {rows.length === 0 && <li className="px-4 py-6 text-center text-[13px] text-muted">{onlyReview ? "Nothing needs a person." : live ? "Classifying…" : "No rows."}</li>}
            </ul>
          </Card>
        </div>
        <Card>
          <h2 className="border-b border-line-soft px-4 py-3 text-[13px] font-medium">Activity</h2>
          <div className="p-2"><AgentFeed events={events} live={live} className="max-h-[480px]" /></div>
        </Card>
      </div>
    </div>
  );
}
