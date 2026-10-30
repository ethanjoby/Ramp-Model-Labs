"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { CandidateRow, Dataset, Lab, LabEvent, LabResult, Spec } from "./types";

export const STAGES = [
  { key: "plan", label: "Plan", agent: "planner" },
  { key: "data", label: "Data", agent: "data" },
  { key: "train", label: "Train", agent: "trainer" },
  { key: "select", label: "Select", agent: "selector" },
  { key: "deploy", label: "Deploy", agent: "deployer" },
] as const;

export type StageKey = (typeof STAGES)[number]["key"];

export type LabState = {
  lab: Lab | null;
  events: LabEvent[];
  stage: StageKey | "done" | "failed" | null;
  candidates: Record<string, CandidateRow>;
  spec: Spec | null;
  dataset: Dataset | null;
  result: LabResult | null;
  error: string | null;
  live: boolean;
};

export const emptyLabState: LabState = {
  lab: null, events: [], stage: null, candidates: {}, spec: null, dataset: null, result: null, error: null, live: false,
};

function hydrate(lab: Lab, prev: LabState): LabState {
  const candidates = { ...prev.candidates };
  for (const c of lab.candidates ?? []) candidates[c.key] = { ...candidates[c.key], ...c };
  const result = lab.result && "winner" in lab.result ? (lab.result as LabResult) : prev.result;
  return {
    ...prev,
    lab,
    candidates,
    spec: lab.spec && "labels" in lab.spec ? (lab.spec as Spec) : prev.spec,
    dataset: lab.dataset && "train" in lab.dataset ? (lab.dataset as Dataset) : prev.dataset,
    result,
    error: lab.error ?? prev.error,
    stage: lab.status === "completed" ? "done" : lab.status === "failed" ? "failed" : prev.stage,
  };
}

export function applyLabEvent(s: LabState, ev: LabEvent): LabState {
  const next: LabState = { ...s, events: [...s.events, ev] };
  const d = ev.data ?? {};
  if (ev.kind === "stage" && d.stage) next.stage = d.stage;
  if (d.spec) next.spec = d.spec;
  if (d.dataset) next.dataset = d.dataset;
  if (d.roster) {
    const c = { ...s.candidates };
    for (const r of d.roster) {
      c[r.key] = {
        ...c[r.key], key: r.key, name: r.name, family: r.family, blurb: r.blurb,
        status: c[r.key]?.status && c[r.key].status !== "queued" ? c[r.key].status : r.available ? "queued" : "skipped",
        metrics: c[r.key]?.metrics ?? (r.available ? {} : { skip_reason: r.reason }),
        reason: r.reason,
      };
    }
    next.candidates = c;
  }
  if (ev.kind === "candidate" && d.key) {
    const prev = s.candidates[d.key] ?? { key: d.key, name: d.key, family: "classical", metrics: {} };
    next.candidates = {
      ...s.candidates,
      [d.key]: { ...prev, status: d.status, metrics: d.metrics ?? prev.metrics },
    };
  }
  if (ev.kind === "result" && d.winner) {
    const c = { ...next.candidates };
    for (const k of Object.keys(c)) c[k] = { ...c[k], is_winner: k === d.winner };
    next.candidates = c;
  }
  if (d.result) next.result = d.result;
  if (d.stage === "done") next.stage = "done";
  if (ev.kind === "error") {
    next.stage = "failed";
    next.error = ev.message;
  }
  return next;
}

export function useLab(labId: string | null) {
  const [state, setState] = useState<LabState>(emptyLabState);
  const [trackedId, setTrackedId] = useState(labId);
  const lastSeq = useRef(0);
  // Reset while rendering when the lab changes, so a client-side hop between labs never shows stale state.
  if (labId !== trackedId) {
    setTrackedId(labId);
    setState(emptyLabState);
  }

  const refresh = useCallback(async () => {
    if (!labId) return;
    const lab = await api<Lab>(`/api/labs/${labId}`);
    setState((s) => hydrate(lab, s));
    return lab;
  }, [labId]);

  useEffect(() => {
    if (!labId) return;
    let es: EventSource | null = null;
    let closed = false;
    lastSeq.current = 0;

    api<Lab>(`/api/labs/${labId}`).then((lab) => {
      if (closed) return;
      setState((s) => hydrate(lab, s));
      es = new EventSource(`/api/labs/${labId}/events`);
      setState((s) => ({ ...s, live: true }));
      es.addEventListener("lab", (m) => {
        const ev = JSON.parse((m as MessageEvent).data) as LabEvent;
        if (ev.seq <= lastSeq.current) return;
        lastSeq.current = ev.seq;
        setState((s) => applyLabEvent(s, ev));
        if (ev.data?.stage === "done" || ev.kind === "error") {
          refresh();
          es?.close();
          setState((s) => ({ ...s, live: false }));
        }
      });
      es.onerror = () => {
        // The browser retries automatically; if the lab already finished, stop listening.
        refresh().then((lab) => {
          if (lab && (lab.status === "completed" || lab.status === "failed")) {
            es?.close();
            setState((s) => ({ ...s, live: false }));
          }
        }).catch(() => {});
      };
    }).catch((e) => setState((s) => ({ ...s, error: e.message ?? "Couldn't load lab" })));

    return () => {
      closed = true;
      es?.close();
    };
  }, [labId, refresh]);

  return { ...state, refresh };
}
