"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, Check, Copy, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState, useSyncExternalStore } from "react";
import { Button } from "@/components/ui/Button";
import { Badge, Card, Spinner, Stat } from "@/components/ui/bits";
import { api } from "@/lib/api";
import { humanLabel, ms, multiple, pct, size, usd } from "@/lib/format";
import type { Deployment, Prediction } from "@/lib/types";

type Row = Prediction & { input: string; corrected?: string };

export function ModelView() {
  const { slug } = useParams<{ slug: string }>();
  const router = useRouter();
  const [dep, setDep] = useState<Deployment | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [rows, setRows] = useState<Row[]>([]);
  const [latency, setLatency] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [retraining, setRetraining] = useState(false);
  const [copied, setCopied] = useState(false);
  const origin = useSyncExternalStore(() => () => {}, () => window.location.origin, () => "");

  const load = useCallback(() => {
    api<Deployment>(`/api/deployments/${slug}`).then(setDep).catch((e) => setErr(e.message));
  }, [slug]);

  useEffect(load, [load]);

  const run = async () => {
    const inputs = text.split("\n").map((l) => l.trim()).filter(Boolean).slice(0, 100);
    if (!inputs.length) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await api<{ predictions: Prediction[]; latency_ms: number }>(`/api/deployments/${slug}/predict`, {
        method: "POST", json: { inputs },
      });
      setRows(r.predictions.map((p, i) => ({ ...p, input: inputs[i] })));
      setLatency(r.latency_ms);
      load();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Prediction failed.");
    } finally {
      setBusy(false);
    }
  };

  const correct = async (i: number, label: string) => {
    const row = rows[i];
    if (!label || label === row.label) return;
    try {
      await api(`/api/deployments/${slug}/feedback`, {
        method: "POST", json: { text: row.input, predicted: row.label, correct: label },
      });
      setRows((rs) => rs.map((r, j) => (j === i ? { ...r, corrected: label } : r)));
      load();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Couldn't save that correction.");
    }
  };

  const retrain = async () => {
    setRetraining(true);
    try {
      const r = await api<{ lab_id: string }>(`/api/deployments/${slug}/retrain`, { method: "POST" });
      router.push(`/app/labs/${r.lab_id}`);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Couldn't start retraining.");
      setRetraining(false);
    }
  };

  const snippet = `curl ${origin}/v1/models/${slug}/predict \\
  -H "Authorization: Bearer $RML_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"inputs": ["UBER *TRIP SF $23.10"]}'`;

  const copy = async () => {
    await navigator.clipboard.writeText(snippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  if (!dep) {
    return (
      <div className="mx-auto max-w-[1040px] px-5 py-8 md:px-8">
        {err ? <p role="alert" className="text-[14px] text-danger">{err}</p> : <div className="skeleton h-40 rounded-2xl" />}
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1040px] px-5 py-8 md:px-8">
      <Link href="/app/models" className="inline-flex items-center gap-1.5 text-[13px] text-muted transition-colors hover:text-ink">
        <ArrowLeft className="h-3.5 w-3.5" /> Models
      </Link>
      <div className="mt-3 flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="truncate text-[30px] font-light tracking-[-0.03em]">{dep.name}</h1>
            <Badge tone={dep.status === "live" ? "smolder" : "neutral"}>{dep.status}</Badge>
          </div>
          <p className="mt-1 text-[14px] text-muted">
            {dep.model} · <Link href={`/app/labs/${dep.lab_id}`} className="underline underline-offset-4 hover:text-ink">view lab</Link>
          </p>
        </div>
      </div>

      <Card className="mt-6 grid grid-cols-2 gap-6 p-5 md:grid-cols-5">
        <Stat label="Held-out accuracy" value={pct(dep.metrics.accuracy)} />
        <Stat label="Cost / 1k" value={usd(dep.metrics.cost_per_1k)} />
        <Stat label="Cheaper than frontier" value={multiple(dep.savings?.multiple)} />
        <Stat label="Latency p50" value={ms(dep.metrics.latency_p50_ms)} hint={dep.metrics.size_kb ? `model ${size(dep.metrics.size_kb)}` : undefined} />
        <Stat label="Requests served" value={dep.requests.toLocaleString()} hint={dep.avg_latency_ms != null ? `avg batch ${ms(dep.avg_latency_ms)}` : undefined} />
      </Card>

      <Card className="mt-6">
        <h2 className="border-b border-line-soft px-4 py-3 text-[13px] font-medium">Playground</h2>
        <div className="p-4">
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4}
            placeholder="One input per line"
            className="w-full resize-y rounded-xl border border-line bg-white p-3 font-mono text-[12.5px] leading-relaxed outline-none transition-[border-color,box-shadow] duration-150 placeholder:text-ink/30 focus:border-ink focus:shadow-[0_0_0_3px_rgba(228,242,34,0.55)]" />
          <div className="mt-3 flex items-center justify-between gap-3">
            <span className="tnum text-[12px] text-muted">{latency != null ? `${rows.length} predictions in ${ms(latency)}` : `${dep.labels.length} labels`}</span>
            <Button variant="ink" size="sm" onClick={run} disabled={busy || !text.trim()}>{busy ? <Spinner /> : "Run model"}</Button>
          </div>
          {err && <p role="alert" className="mt-3 text-[13px] text-danger">{err}</p>}
          {rows.length > 0 && (
            <ul className="mt-4 divide-y divide-line-soft rounded-xl border border-line-soft">
              {rows.map((r, i) => (
                <li key={i} className="grid items-center gap-2 px-3 py-2.5 md:grid-cols-[minmax(0,1fr)_auto] md:gap-4">
                  <span className="min-w-0 break-words font-mono text-[12px] text-ink/80">{r.input}</span>
                  <span className="flex flex-wrap items-center gap-2">
                    <Badge tone={r.corrected ? "neutral" : "solar"} className={r.corrected ? "line-through" : undefined}>{humanLabel(r.label)}</Badge>
                    {r.route === "review" && !r.corrected && <Badge tone="danger">needs review</Badge>}
                    {r.confidence != null && <span className="tnum font-mono text-[11.5px] text-muted">{Math.round(r.confidence * 100)}%</span>}
                    {r.corrected ? (
                      <Badge tone="smolder"><Check className="h-3 w-3" />{humanLabel(r.corrected)}</Badge>
                    ) : (
                      <select aria-label={`Correct the label for ${r.input}`} value="" onChange={(e) => void correct(i, e.target.value)}
                        className="h-[26px] max-w-[130px] rounded-lg border border-line-soft bg-white px-1.5 text-[12px] text-muted">
                        <option value="">Correct…</option>
                        {dep.labels.filter((l) => l !== r.label).map((l) => <option key={l} value={l}>{humanLabel(l)}</option>)}
                      </select>
                    )}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </Card>

      <div className="mt-6 grid items-start gap-6 md:grid-cols-2">
        <Card className="p-5">
          <h2 className="text-[13px] font-medium">Improve from corrections</h2>
          <p className="mt-1.5 text-[13.5px] leading-relaxed text-muted">
            Corrections become real labeled data. Retraining re-runs the whole lab with them included. The current model stays live.
          </p>
          <div className="mt-4 flex items-center justify-between gap-3">
            <span className="tnum text-[13px]">{dep.feedback_count} correction{dep.feedback_count === 1 ? "" : "s"} saved</span>
            <Button variant="outline" size="sm" onClick={retrain} disabled={retraining || dep.feedback_count === 0}>
              {retraining ? <Spinner /> : <RefreshCw className="h-3.5 w-3.5" />} Retrain
            </Button>
          </div>
        </Card>
        <Card className="min-w-0 p-5">
          <div className="flex items-center justify-between">
            <h2 className="text-[13px] font-medium">Call it from your code</h2>
            <button onClick={copy} aria-label="Copy code" className="rounded-md p-1 text-ink/50 transition-colors hover:bg-ink/5 hover:text-ink">
              {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
            </button>
          </div>
          <pre className="scroll-thin mt-3 overflow-x-auto rounded-xl bg-ink p-3.5 font-mono text-[11.5px] leading-relaxed text-white/85">{snippet}</pre>
          <p className="mt-2 text-[12.5px] text-muted">Create a key on the <Link href="/app/keys" className="underline underline-offset-4 hover:text-ink">API keys</Link> page.</p>
        </Card>
      </div>
    </div>
  );
}
