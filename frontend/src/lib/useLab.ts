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
  /** The record says the lab is completed or failed; replayed events may no longer move stage or status. */
  terminal: boolean;
};

export const emptyLabState: LabState = {
  lab: null, events: [], stage: null, candidates: {}, spec: null, dataset: null, result: null, error: null, live: false, terminal: false,
};

export const isTerminal = (status: Lab["status"] | undefined) => status === "completed" || status === "failed";

function hydrate(lab: Lab, prev: LabState): LabState {
  const candidates = { ...prev.candidates };
  for (const c of lab.candidates ?? []) candidates[c.key] = { ...candidates[c.key], ...c };
  const result = lab.result && "winner" in lab.result ? (lab.result as LabResult) : prev.result;
  const terminal = isTerminal(lab.status);
  return {
    ...prev,
    lab,
    candidates,
    spec: lab.spec && "labels" in lab.spec ? (lab.spec as Spec) : prev.spec,
    dataset: lab.dataset && "train" in lab.dataset ? (lab.dataset as Dataset) : prev.dataset,
    result,
    error: lab.error ?? prev.error,
    stage: lab.status === "completed" ? "done" : lab.status === "failed" ? "failed" : prev.stage,
    terminal,
    live: terminal ? false : prev.live,
  };
}

export function applyLabEvent(s: LabState, ev: LabEvent): LabState {
  // Events arrive in seq order; one already held is a replay (reconnect, hot reload) and must not be shown twice.
  if (s.events.length && s.events[s.events.length - 1].seq >= ev.seq) return s;
  const next: LabState = { ...s, events: [...s.events, ev] };
  const d = ev.data ?? {};
  if (s.terminal) {
    // The stored record already says how this lab ended; a replayed history only fills the feed and the roster copy.
    if (d.roster) {
      const c = { ...s.candidates };
      for (const r of d.roster) if (c[r.key]) c[r.key] = { ...c[r.key], blurb: r.blurb, reason: r.reason };
      next.candidates = c;
    }
    return next;
  }
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
    // A candidate the roster did not announce (the distillation student) arrives with its own name and family.
    const prev = s.candidates[d.key] ?? { key: d.key, name: d.name ?? d.key, family: d.family ?? "classical", metrics: {}, blurb: d.blurb };
    next.candidates = {
      ...s.candidates,
      [d.key]: { ...prev, status: d.status, metrics: d.metrics ?? prev.metrics, name: d.name ?? prev.name, family: d.family ?? prev.family },
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

/**
 * Follows one lab. The stored record is authoritative; the event stream only animates the way there.
 * `history: false` (a card with no feed) skips the stream entirely once the lab is terminal.
 */
export function useLab(labId: string | null, { history = true }: { history?: boolean } = {}) {
  const [state, setState] = useState<LabState>(emptyLabState);
  const [trackedId, setTrackedId] = useState(labId);
  const lastSeq = useRef(0);
  const current = useRef(labId);
  useEffect(() => { current.current = labId; }, [labId]);
  // Kept in step with the list (which the lab change above empties) instead of zeroed by the stream effect, so a re-run
  // effect that replays history from seq 1 skips everything already shown.
  useEffect(() => { lastSeq.current = state.events.length ? state.events[state.events.length - 1].seq : 0; }, [state.events]);
  // Reset while rendering when the lab changes, so a client-side hop between labs never shows stale state.
  if (labId !== trackedId) {
    setTrackedId(labId);
    setState(emptyLabState);
  }

  const refresh = useCallback(async () => {
    if (!labId) return;
    const lab = await api<Lab>(`/api/labs/${labId}`);
    // A slow response for a lab we have since navigated away from must not land in the new lab's state.
    if (current.current === labId) setState((s) => hydrate(lab, s));
    return lab;
  }, [labId]);

  useEffect(() => {
    if (!labId) return;
    let es: EventSource | null = null;
    let closed = false;
    let ended = false;

    const stop = () => {
      es?.close();
      es = null;
      if (!closed) setState((s) => (s.live ? { ...s, live: false } : s));
    };

    api<Lab>(`/api/labs/${labId}`).then((lab) => {
      if (closed) return;
      setState((s) => hydrate(lab, s));
      ended = isTerminal(lab.status);
      if (ended && !history) return;
      es = new EventSource(`/api/labs/${labId}/events`);
      if (!ended) setState((s) => ({ ...s, live: true }));
      es.addEventListener("lab", (m) => {
        const ev = JSON.parse((m as MessageEvent).data) as LabEvent;
        if (ev.seq <= lastSeq.current) return;
        lastSeq.current = ev.seq;
        setState((s) => applyLabEvent(s, ev));
        if (ev.data?.stage === "done" || ev.kind === "error") {
          // Re-read the record rather than trusting the event: status, result and meets_bar live there.
          if (!ended) refresh().catch(() => {});
          ended = true;
          stop();
        }
      });
      es.onerror = () => {
        // The server ends the stream once a lab is terminal; without this the browser would reconnect forever.
        if (ended) { stop(); return; }
        refresh().then((fresh) => {
          if (fresh && isTerminal(fresh.status)) { ended = true; stop(); }
        }).catch(() => {});
      };
    }).catch((e) => { if (!closed) setState((s) => ({ ...s, error: e.message ?? "Couldn't load lab" })); });

    return () => {
      closed = true;
      es?.close();
    };
  }, [labId, refresh, history]);

  return { ...state, refresh };
}

/** Re-runs `load` on focus and when the tab becomes visible, and every `intervalMs` while `active` (a lab is building). */
export function useRefetch(load: () => void, active: boolean, intervalMs = 4000) {
  useEffect(() => {
    const visible = () => document.visibilityState === "visible";
    const onVisibility = () => { if (visible()) load(); };
    window.addEventListener("focus", load);
    document.addEventListener("visibilitychange", onVisibility);
    const timer = active ? window.setInterval(() => { if (visible()) load(); }, intervalMs) : 0;
    return () => {
      window.removeEventListener("focus", load);
      document.removeEventListener("visibilitychange", onVisibility);
      if (timer) window.clearInterval(timer);
    };
  }, [load, active, intervalMs]);
}
