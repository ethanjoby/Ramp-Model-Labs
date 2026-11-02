"use client";

import Link from "next/link";
import { Plus } from "lucide-react";
import { useEffect, useState } from "react";
import { PageHeader } from "@/components/app/Shell";
import { ButtonLink } from "@/components/ui/Button";
import { Badge, Card, Spinner } from "@/components/ui/bits";
import { api } from "@/lib/api";
import { ago, multiple, pct, usd } from "@/lib/format";
import type { Lab } from "@/lib/types";

function Status({ status }: { status: Lab["status"] }) {
  if (status === "completed") return <Badge tone="smolder">deployed</Badge>;
  if (status === "failed") return <Badge tone="danger">failed</Badge>;
  return <Badge tone="solar"><Spinner className="h-3 w-3" />{status}</Badge>;
}

export default function LabsPage() {
  const [labs, setLabs] = useState<Lab[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let timer = 0;
    const load = () => api<Lab[]>("/api/labs").then((l) => {
      setLabs(l);
      window.clearTimeout(timer);
      if (l.some((x) => x.status === "queued" || x.status === "running")) timer = window.setTimeout(load, 4000);
    }).catch((e) => setErr(e.message));
    load();
    const onVisible = () => { if (document.visibilityState === "visible") load(); };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", onVisible);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("focus", onVisible);
    };
  }, []);

  return (
    <div className="mx-auto max-w-[1040px] px-5 py-8 md:px-8">
      <PageHeader title="Labs" sub="Every build: the plan, the data, the candidates, and the verdict."
        right={<ButtonLink href="/app?new=1" variant="ink" size="sm"><Plus className="h-4 w-4" /> New model</ButtonLink>} />
      {err && <p role="alert" className="mt-6 text-[13.5px] text-danger">{err}</p>}
      {!labs && !err && <div className="skeleton mt-8 h-40 rounded-2xl" />}
      {labs?.length === 0 && (
        <Card className="mt-8 p-10 text-center">
          <p className="text-[15px]">No labs yet.</p>
          <p className="mt-1 text-[13.5px] text-muted">Describe a workflow in the Copilot to start your first build.</p>
        </Card>
      )}
      {!!labs?.length && (
        <Card className="mt-8 overflow-hidden">
          <ul className="divide-y divide-line-soft">
            {labs.map((l) => (
              <li key={l.id}>
                <Link href={`/app/labs/${l.id}`} className="grid gap-2 px-4 py-3.5 transition-colors hover:bg-paper/60 md:grid-cols-[minmax(0,1fr)_auto] md:items-center md:gap-6">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-[14px] font-medium">{l.name || l.prompt}</span>
                      <Status status={l.status} />
                    </div>
                    <p className="mt-0.5 truncate text-[12.5px] text-muted">{l.prompt}</p>
                  </div>
                  <div className="tnum flex items-center gap-5 font-mono text-[12px] text-ink/70">
                    {l.status === "completed" && (
                      <>
                        <span>{pct(l.winner_metrics.accuracy)}</span>
                        <span>{usd(l.winner_metrics.cost_per_1k)} / 1k</span>
                        <span className="text-ink">{multiple(l.summary.savings?.multiple)}</span>
                      </>
                    )}
                    <span className="text-muted">{ago(l.created_at)}</span>
                  </div>
                </Link>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}
