"use client";

import Link from "next/link";
import { Plus } from "lucide-react";
import { useEffect, useState } from "react";
import { PageHeader } from "@/components/app/Shell";
import { ButtonLink } from "@/components/ui/Button";
import { Badge, Card, FAMILY_LABEL, FamilyDot } from "@/components/ui/bits";
import { api } from "@/lib/api";
import { ms, multiple, pct, usd } from "@/lib/format";
import type { Deployment } from "@/lib/types";

export default function ModelsPage() {
  const [deps, setDeps] = useState<Deployment[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api<Deployment[]>("/api/deployments").then(setDeps).catch((e) => setErr(e.message));
  }, []);

  return (
    <div className="mx-auto max-w-[1040px] px-5 py-8 md:px-8">
      <PageHeader title="Models" sub="Deployed specialists, ready to call from your code."
        right={<ButtonLink href="/app?new=1" variant="ink" size="sm"><Plus className="h-4 w-4" /> New model</ButtonLink>} />
      {err && <p role="alert" className="mt-6 text-[13.5px] text-danger">{err}</p>}
      {!deps && !err && <div className="skeleton mt-8 h-40 rounded-2xl" />}
      {deps?.length === 0 && (
        <Card className="mt-8 p-10 text-center">
          <p className="text-[15px]">No models deployed yet.</p>
          <p className="mt-1 text-[13.5px] text-muted">Finished labs deploy their winning model here automatically.</p>
        </Card>
      )}
      <div className="mt-8 grid gap-4 md:grid-cols-2">
        {deps?.map((d) => (
          <Link key={d.id} href={`/app/models/${d.slug}`} className="rounded-2xl border border-line-soft bg-white p-5 transition-colors hover:border-ink/30">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <h2 className="truncate text-[16px] font-medium tracking-[-0.01em]">{d.name}</h2>
                <p className="mt-1 flex items-center gap-1.5 truncate text-[12.5px] text-muted">
                  <FamilyDot family={d.family} /> {FAMILY_LABEL[d.family] ?? d.family} · {d.model}
                </p>
              </div>
              <Badge tone={d.status === "live" ? "smolder" : "neutral"}>{d.status}</Badge>
            </div>
            <dl className="tnum mt-5 grid grid-cols-4 gap-3">
              {[
                ["Accuracy", pct(d.metrics.accuracy)],
                ["Cost / 1k", usd(d.metrics.cost_per_1k)],
                ["p50", ms(d.metrics.latency_p50_ms)],
                ["Savings", multiple(d.savings?.multiple)],
              ].map(([k, v]) => (
                <div key={k} className="min-w-0">
                  <dt className="text-[11.5px] text-muted">{k}</dt>
                  <dd className="mt-0.5 truncate font-mono text-[13px]">{v}</dd>
                </div>
              ))}
            </dl>
          </Link>
        ))}
      </div>
    </div>
  );
}
