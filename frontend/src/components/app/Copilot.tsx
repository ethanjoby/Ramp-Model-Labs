"use client";

import clsx from "clsx";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { AnimatePresence, motion } from "motion/react";
import { ArrowUp, ArrowUpRight, Paperclip, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Badge, Spinner } from "@/components/ui/bits";
import { StageRail } from "@/components/lab/parts";
import { useSession } from "@/components/app/Shell";
import { api } from "@/lib/api";
import { humanLabel, multiple, pct, usd } from "@/lib/format";
import { EASE_OUT } from "@/lib/motion";
import type { ChatAction, ChatMsg } from "@/lib/types";
import { useLab } from "@/lib/useLab";

const DistillScene = dynamic(() => import("@/components/three/DistillScene"), { ssr: false });

const STARTERS = [
  "Categorize card transactions into GL codes, cheap enough to run on every swipe",
  "Screen transactions for fraud and flag anything suspicious",
  "Triage incoming invoices by type and urgency",
  "Route financial support tickets to the right team",
  "Reconcile bank lines against the ledger and flag anything that doesn't match",
];

type Attachment = { name: string; raw: string; count: number; labeled: number };

function LabCard({ labId }: { labId: string }) {
  const { lab, stage, spec, result, error, live } = useLab(labId);
  const m = result?.winner_metrics;
  return (
    <Link href={`/app/labs/${labId}`}
      className="group mt-3 block rounded-xl border border-line-soft bg-white p-4 transition-colors hover:border-ink/30">
      <div className="flex items-center justify-between gap-3">
        <span className="truncate text-[13.5px] font-medium">{spec?.name ?? lab?.name ?? "New lab"}</span>
        <span className="flex shrink-0 items-center gap-1.5 text-[12px] text-muted">
          {live && <Spinner className="h-3 w-3" />}
          {stage === "done" ? "Deployed" : stage === "failed" ? "Failed" : "Building"}
          <ArrowUpRight className="h-3.5 w-3.5 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
        </span>
      </div>
      <div className="mt-3"><StageRail stage={stage} compact /></div>
      {error && <p className="mt-2 text-[12.5px] text-danger">{error}</p>}
      {result && m && (
        <div className="tnum mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[12.5px] text-ink/70">
          <span className="font-medium text-ink">{result.winner_name}</span>
          <span>{pct(m.accuracy)} accuracy</span>
          <span>{usd(m.cost_per_1k)} / 1k</span>
          <span>{multiple(result.savings?.multiple)} cheaper than frontier</span>
        </div>
      )}
    </Link>
  );
}

function ActionView({ action }: { action: ChatAction }) {
  if (action.type === "lab_started" || action.type === "lab_ref") return <LabCard labId={action.lab_id} />;
  if (action.type === "prediction") {
    return (
      <div className="mt-3 overflow-hidden rounded-xl border border-line-soft bg-white">
        <ul className="divide-y divide-line-soft">
          {action.rows.map((r, i) => (
            <li key={i} className="flex items-center justify-between gap-4 px-3.5 py-2.5">
              <span className="min-w-0 truncate font-mono text-[12px] text-ink/75">{r.input}</span>
              <span className="flex shrink-0 items-center gap-2">
                <Badge tone="solar">{humanLabel(r.label)}</Badge>
                {r.route === "review" && <Badge tone="danger">needs review</Badge>}
                {r.confidence != null && <span className="tnum w-9 text-right font-mono text-[11.5px] text-muted">{Math.round(r.confidence * 100)}%</span>}
              </span>
            </li>
          ))}
        </ul>
        <Link href={`/app/models/${action.slug}`} className="block border-t border-line-soft px-3.5 py-2 text-[12px] text-muted hover:text-ink">
          {action.name} · open playground
        </Link>
      </div>
    );
  }
  if (action.type === "models") {
    return (
      <ul className="mt-3 divide-y divide-line-soft overflow-hidden rounded-xl border border-line-soft bg-white">
        {action.models.map((m) => (
          <li key={m.slug}>
            <Link href={`/app/models/${m.slug}`} className="flex items-center justify-between gap-4 px-3.5 py-2.5 text-[13px] hover:bg-paper/60">
              <span className="min-w-0 truncate">{m.name}</span>
              <span className="tnum shrink-0 font-mono text-[12px] text-muted">{pct(m.accuracy)} · {usd(m.cost_per_1k)}</span>
            </Link>
          </li>
        ))}
      </ul>
    );
  }
  return null;
}

export function Copilot() {
  const params = useSearchParams();
  const { user } = useSession();
  const [msgs, setMsgs] = useState<ChatMsg[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [attachment, setAttachment] = useState<Attachment | null>(null);
  const [attachErr, setAttachErr] = useState<string | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const handledPrompt = useRef(false);
  const queued = params.get("prompt");
  const isNew = params.get("new");

  const send = useCallback(async (message: string, att: Attachment | null) => {
    const content = message.trim();
    if (!content) return;
    setBusy(true);
    setMsgs((m) => [...m, { role: "user", content }, { role: "assistant", content: "", pending: true }]);
    try {
      const r = await api<{ reply: string; actions: ChatAction[] }>("/api/chat", {
        method: "POST", json: { message: content, attachment: att?.raw ?? null },
      });
      setMsgs((m) => [...m.slice(0, -1), { role: "assistant", content: r.reply, actions: r.actions }]);
    } catch (e) {
      setMsgs((m) => [...m.slice(0, -1), { role: "assistant", content: e instanceof Error ? e.message : "Something went wrong." }]);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    api<ChatMsg[]>("/api/chat").then(setMsgs).catch(() => {}).finally(() => setLoaded(true));
  }, []);

  // A prompt typed on the homepage carries through signup and starts a real build here.
  useEffect(() => {
    if (!loaded || !queued || handledPrompt.current) return;
    handledPrompt.current = true;
    // Drop the query in place; a router navigation here could race the build on a cold dev server.
    window.history.replaceState(null, "", "/app");
    void send(queued, null);
  }, [loaded, queued, send]);

  useEffect(() => {
    if (isNew) inputRef.current?.focus();
  }, [isNew]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [msgs.length, busy]);

  const submit = () => {
    if (busy || !text.trim()) return;
    const att = attachment;
    setText("");
    setAttachment(null);
    void send(text, att);
  };

  const onFile = async (f: File | undefined) => {
    if (!f) return;
    setAttachErr(null);
    const body = new FormData();
    body.append("file", f);
    try {
      const r = await api<Omit<Attachment, "name">>("/api/labs/parse-examples", { method: "POST", body });
      setAttachment({ name: f.name, raw: r.raw, count: r.count, labeled: r.labeled });
    } catch (e) {
      setAttachErr(e instanceof Error ? e.message : "Couldn't read that file.");
    }
    if (fileRef.current) fileRef.current.value = "";
  };

  const clear = async () => {
    await api("/api/chat", { method: "DELETE" }).catch(() => {});
    setMsgs([]);
  };

  const empty = loaded && msgs.length === 0;

  return (
    <div className="flex h-[calc(100svh-57px)] flex-col md:h-[100svh]">
      <div className="scroll-thin flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[760px] px-5 py-8">
          {msgs.length > 0 && (
            <div className="mb-4 flex justify-end">
              <button onClick={clear} className="flex items-center gap-1.5 rounded-md px-2 py-1 text-[12px] text-muted transition-colors hover:bg-ink/5 hover:text-ink">
                <Trash2 className="h-3.5 w-3.5" /> Clear chat
              </button>
            </div>
          )}
          {empty && (
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, ease: EASE_OUT }} className="pt-[6vh]">
              <div className="dark relative overflow-hidden rounded-3xl bg-ink px-7 py-10 text-white md:px-9 md:py-12">
                <DistillScene density={0.45} className="absolute inset-0" offset={[3.4, -0.1, 0]} offsetMobile={[0, -1.5, 0]} interactive={false} />
                <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(90deg,rgba(12,10,8,0.92)_0%,rgba(12,10,8,0.6)_50%,transparent_85%)]" />
                <div className="relative">
                  <h1 className="display max-w-[12ch] text-[40px] md:text-[48px]">
                    {user ? `What should we automate, ${user.name.split(" ")[0]}?` : "What should we automate?"}
                  </h1>
                  <p className="mt-4 max-w-[50ch] text-[15px] leading-relaxed text-white/65">
                    Describe a financial workflow. The agents generate data, train competing models, test them on held-out examples, and deploy the cheapest one that clears the bar.
                  </p>
                </div>
              </div>
              <div className="mt-6 grid gap-2 sm:grid-cols-2">
                {STARTERS.map((s) => (
                  <button key={s} onClick={() => void send(s, null)}
                    className="rounded-xl border border-line-soft bg-white p-3.5 text-left text-[13.5px] leading-snug text-ink/80 transition-colors hover:border-ink/30 hover:text-ink">
                    {s}
                  </button>
                ))}
              </div>
            </motion.div>
          )}
          <ul className="space-y-5">
            <AnimatePresence initial={false}>
              {msgs.map((m, i) => (
                <motion.li key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, ease: EASE_OUT }}
                  className={clsx(m.role === "user" && "flex justify-end")}>
                  {m.role === "user" ? (
                    <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl bg-ink px-4 py-2.5 text-[14px] leading-relaxed text-white">{m.content}</div>
                  ) : m.pending ? (
                    <span className="caret text-[14px] text-ink/40">Thinking</span>
                  ) : (
                    <div>
                      <p className="whitespace-pre-wrap text-[14.5px] leading-relaxed">{m.content.replace(/\*\*/g, "")}</p>
                      {m.actions?.map((a, j) => <ActionView key={j} action={a} />)}
                    </div>
                  )}
                </motion.li>
              ))}
            </AnimatePresence>
          </ul>
          <div ref={endRef} />
        </div>
      </div>

      <div className="border-t border-line-soft bg-paper px-5 py-4">
        <div className="mx-auto w-full max-w-[760px]">
          {attachErr && <p role="alert" className="mb-2 text-[12.5px] text-danger">{attachErr}</p>}
          {attachment && (
            <div className="mb-2 inline-flex items-center gap-2 rounded-lg border border-line-soft bg-white px-2.5 py-1.5 text-[12.5px]">
              <Paperclip className="h-3.5 w-3.5 text-muted" />
              {attachment.name} · {attachment.count} examples, {attachment.labeled} labeled
              <button onClick={() => setAttachment(null)} aria-label="Remove attachment" className="text-muted hover:text-ink"><X className="h-3.5 w-3.5" /></button>
            </div>
          )}
          <div className="flex items-end gap-2 rounded-2xl border border-line bg-white p-2 transition-[border-color,box-shadow] duration-150 focus-within:border-ink focus-within:shadow-[0_0_0_3px_rgba(228,242,34,0.55)]">
            <input ref={fileRef} type="file" accept=".csv,.jsonl,.json,.txt" className="hidden" onChange={(e) => void onFile(e.target.files?.[0])} />
            <button onClick={() => fileRef.current?.click()} aria-label="Attach examples"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-ink/50 transition-colors hover:bg-ink/5 hover:text-ink">
              <Paperclip className="h-4 w-4" />
            </button>
            <textarea
              ref={inputRef} value={text} rows={1} maxLength={8000}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
              placeholder="Build a model that… or classify: UBER *TRIP SF $23.10"
              className="max-h-40 min-h-9 flex-1 resize-none bg-transparent py-2 text-[14.5px] outline-none placeholder:text-ink/35 focus-visible:outline-none"
            />
            <button onClick={submit} disabled={busy || !text.trim()} aria-label="Send"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-solar text-ink transition-colors hover:bg-solar-light disabled:opacity-40">
              {busy ? <Spinner /> : <ArrowUp className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
