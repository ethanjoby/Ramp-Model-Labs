"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { LabEvent } from "./types";

export type WorkflowSummary = {
  model?: { slug: string; name: string; candidate: string; status: string; accuracy: number | null };
  distribution?: Record<string, number>;
  rows?: number; review?: number; auto?: number; mean_confidence?: number | null; latency_ms?: number;
  cost_usd?: number; frontier_cost_usd?: number | null; frontier_measured?: boolean;
  accuracy_on_labeled?: number | null; labeled_rows?: number;
  parse?: { kind: string; count: number; labeled: number; columns: string[]; text_column: string | null; label_column: string | null; notes: string[] };
};

export type Workflow = {
  id: string; name: string; prompt: string; status: "queued" | "choosing" | "building" | "classifying" | "completed" | "failed";
  error: string | null; source_kind: string; source_name: string; rows_total: number; rows_done: number; rows_review: number;
  confidence_floor: number; summary: WorkflowSummary; lab_id: string | null; created_at: string; finished_at: string | null;
  deployment: { slug: string; name: string; status: string; model: string | null; labels: string[] } | null;
};

export type WorkflowRow = {
  id: number; idx: number; text: string; meta: Record<string, string>; label: string | null; confidence: number | null;
  route: string | null; top: { label: string; p: number }[]; needs_review: boolean; reviewed_label: string | null; truth: string | null;
};

export const WF_STAGES = [
  { key: "parse", label: "Parse" }, { key: "model", label: "Model" }, { key: "classify", label: "Classify" }, { key: "done", label: "Done" },
] as const;

const terminal = (w: Workflow | null) => !!w && (w.status === "completed" || w.status === "failed");

export function useWorkflow(id: string | null) {
  const [wf, setWf] = useState<Workflow | null>(null);
  const [events, setEvents] = useState<LabEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [trackedId, setTrackedId] = useState(id);
  // Reset while rendering when the workflow changes, so a client-side hop between two workflows never keeps the first one's feed.
  if (id !== trackedId) {
    setTrackedId(id);
    setWf(null);
    setEvents([]);
    setError(null);
  }
  // The highest seq we hold, kept in step with the list rather than zeroed by the stream effect: a replay (reconnect,
  // or a dev hot reload re-running the effect) resends history from 1, and the list is the only record of what is already shown.
  const lastSeq = useRef(0);
  useEffect(() => { lastSeq.current = events.length ? events[events.length - 1].seq : 0; }, [events]);

  const refresh = useCallback(async () => {
    if (!id) return null;
    const w = await api<Workflow>(`/api/workflows/${id}`);
    setWf(w);
    return w;
  }, [id]);

  useEffect(() => {
    if (!id) return;
    let es: EventSource | null = null;
    let closed = false;
    api<Workflow>(`/api/workflows/${id}`).then((w) => {
      if (closed) return;
      setWf(w);
      es = new EventSource(`/api/workflows/${id}/events`);
      es.addEventListener("lab", (m) => {
        const ev = JSON.parse((m as MessageEvent).data) as LabEvent;
        if (ev.seq <= lastSeq.current) return;
        lastSeq.current = ev.seq;
        setEvents((list) => (list.length && list[list.length - 1].seq >= ev.seq ? list : [...list, ev]));
        const stage = ev.data?.stage;
        if (stage === "classify" || stage === "model" || stage === "build" || stage === "done" || ev.kind === "error") void refresh();
        if (stage === "done" || stage === "failed" || ev.kind === "error") es?.close();
      });
      es.onerror = () => {
        refresh().then((latest) => { if (terminal(latest)) es?.close(); }).catch(() => {});
      };
      if (terminal(w)) {
        // the server ends the stream after replaying history; nothing else to do
      }
    }).catch((e) => setError(e instanceof Error ? e.message : "Couldn't load workflow"));
    return () => { closed = true; es?.close(); };
  }, [id, refresh]);

  const stage: (typeof WF_STAGES)[number]["key"] | "failed" =
    !wf ? "parse" : wf.status === "failed" ? "failed" : wf.status === "completed" ? "done"
      : wf.status === "classifying" ? "classify" : wf.status === "queued" ? "parse" : "model";

  return { wf, events, error, stage, live: !!wf && !terminal(wf), refresh };
}
