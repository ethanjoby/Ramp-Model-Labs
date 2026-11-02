"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { FileSpreadsheet, Mail, FileText, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { PageHeader } from "@/components/app/Shell";
import { Button } from "@/components/ui/Button";
import { Badge, Card, Spinner } from "@/components/ui/bits";
import { api } from "@/lib/api";
import { ago } from "@/lib/format";
import type { Deployment } from "@/lib/types";
import type { Workflow } from "@/lib/useWorkflow";

const KIND_ICON: Record<string, typeof FileText> = { spreadsheet: FileSpreadsheet, csv: FileSpreadsheet, json: FileSpreadsheet, email: Mail, "email+attachment": Mail };

function StatusBadge({ status }: { status: Workflow["status"] }) {
  if (status === "completed") return <Badge tone="smolder">done</Badge>;
  if (status === "failed") return <Badge tone="danger">failed</Badge>;
  return <Badge tone="solar"><Spinner className="h-3 w-3" />{status}</Badge>;
}

export function NewWorkflowForm({ defaultSlug, compact = false }: { defaultSlug?: string; compact?: boolean }) {
  const router = useRouter();
  const params = useSearchParams();
  const [deps, setDeps] = useState<Deployment[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("");
  const [slug, setSlug] = useState(defaultSlug ?? params.get("model") ?? "");
  const [floor, setFloor] = useState(0.6);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api<Deployment[]>("/api/deployments").then((d) => setDeps(d.filter((x) => x.status === "live"))).catch(() => {});
  }, []);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) { setErr("Attach a spreadsheet, CSV, email (.eml), PDF or text file."); return; }
    setBusy(true);
    setErr(null);
    const body = new FormData();
    body.append("file", file);
    body.append("prompt", prompt);
    body.append("confidence_floor", String(floor));
    if (slug) body.append("deployment", slug);
    try {
      const w = await api<Workflow>("/api/workflows", { method: "POST", body });
      router.push(`/app/workflows/${w.id}`);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Couldn't start the workflow.");
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="space-y-4">
      <label className="flex cursor-pointer items-center gap-3 rounded-xl border border-dashed border-line bg-white px-4 py-4 transition-colors hover:border-ink/40">
        <Upload className="h-5 w-5 shrink-0 text-muted" />
        <span className="min-w-0 flex-1">
          <span className="block text-[14px]">{file ? file.name : "Drop in a spreadsheet, CSV, email or PDF"}</span>
          <span className="block text-[12px] text-muted">{file ? `${(file.size / 1024).toFixed(0)} KB` : ".xlsx .csv .eml .pdf .json .txt, up to 5 MB"}</span>
        </span>
        <input ref={fileRef} type="file" className="hidden" accept=".xlsx,.xlsm,.csv,.tsv,.eml,.pdf,.json,.jsonl,.txt" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </label>
      <input value={prompt} onChange={(e) => setPrompt(e.target.value)} maxLength={2000} placeholder="What should happen to these rows? e.g. categorize each card transaction into a GL code"
        className="h-11 w-full rounded-xl border border-line bg-white px-3.5 text-[14px] outline-none transition-[border-color,box-shadow] duration-150 placeholder:text-ink/35 focus:border-ink focus:shadow-[0_0_0_3px_rgba(228,242,34,0.55)]" />
      <div className={compact ? "space-y-3" : "grid gap-3 sm:grid-cols-2"}>
        <label className="block text-[12.5px] text-muted">
          Model
          <select value={slug} onChange={(e) => setSlug(e.target.value)} className="mt-1 h-10 w-full rounded-xl border border-line bg-white px-3 text-[13.5px] text-ink">
            <option value="">Pick the best fit automatically, or build one</option>
            {deps.map((d) => <option key={d.slug} value={d.slug}>{d.name}</option>)}
          </select>
        </label>
        <label className="block text-[12.5px] text-muted">
          Send to review below {Math.round(floor * 100)}% confidence
          <input type="range" min={0.3} max={0.95} step={0.05} value={floor} onChange={(e) => setFloor(parseFloat(e.target.value))} className="mt-3 w-full accent-[var(--ink)]" />
        </label>
      </div>
      {err && <p role="alert" className="text-[13px] text-danger">{err}</p>}
      <Button type="submit" variant="ink" disabled={busy}>{busy ? <Spinner /> : "Run workflow"}</Button>
    </form>
  );
}

export function WorkflowsList() {
  const [items, setItems] = useState<Workflow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let timer = 0;
    const load = () => api<Workflow[]>("/api/workflows").then((w) => {
      setItems(w);
      if (w.some((x) => x.status !== "completed" && x.status !== "failed")) timer = window.setTimeout(load, 4000);
    }).catch((e) => setErr(e.message));
    load();
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <div className="mx-auto max-w-[1040px] px-5 py-8 md:px-8">
      <PageHeader title="Workflows" sub="A spreadsheet, inbox export or PDF in; every row decided by your best model, the uncertain ones routed to a person, results back out." />
      <Card className="mt-8 p-5">
        <h2 className="text-[13px] font-medium">New workflow</h2>
        <div className="mt-4"><NewWorkflowForm /></div>
      </Card>
      {err && <p role="alert" className="mt-6 text-[13.5px] text-danger">{err}</p>}
      {items?.length === 0 && <p className="mt-6 text-[13.5px] text-muted">No workflows yet.</p>}
      {!!items?.length && (
        <Card className="mt-6 overflow-hidden">
          <ul className="divide-y divide-line-soft">
            {items.map((w) => {
              const Icon = KIND_ICON[w.source_kind] ?? FileText;
              return (
                <li key={w.id}>
                  <Link href={`/app/workflows/${w.id}`} className="grid gap-2 px-4 py-3.5 transition-colors hover:bg-paper/60 md:grid-cols-[minmax(0,1fr)_auto] md:items-center md:gap-6">
                    <div className="flex min-w-0 items-center gap-3">
                      <Icon className="h-4 w-4 shrink-0 text-muted" strokeWidth={1.75} />
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="truncate text-[14px] font-medium">{w.name}</span>
                          <StatusBadge status={w.status} />
                        </div>
                        <p className="mt-0.5 truncate text-[12.5px] text-muted">{w.prompt || w.source_kind}{w.deployment ? ` · ${w.deployment.name}` : ""}</p>
                      </div>
                    </div>
                    <div className="tnum flex items-center gap-5 font-mono text-[12px] text-ink/70">
                      <span>{w.rows_done}/{w.rows_total} rows</span>
                      {w.rows_review > 0 && <span className="text-danger">{w.rows_review} review</span>}
                      <span className="text-muted">{ago(w.created_at)}</span>
                    </div>
                  </Link>
                </li>
              );
            })}
          </ul>
        </Card>
      )}
    </div>
  );
}
