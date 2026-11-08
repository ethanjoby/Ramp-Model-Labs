"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { AgentFeed, Confusion, DatasetPanel, ErrorList, Leaderboard, ParetoChart, Scorecard, SpecSummary, StageRail, WinnerCard } from "@/components/lab/parts";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/bits";
import { api } from "@/lib/api";
import { useLab } from "@/lib/useLab";

function Section({ title, className, children }: { title: string; className?: string; children: React.ReactNode }) {
  return (
    <Card className={className}>
      <h2 className="border-b border-line-soft px-4 py-3 text-[13px] font-medium">{title}</h2>
      <div className="p-4">{children}</div>
    </Card>
  );
}

export function LabView() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { lab, events, stage, candidates, spec, dataset, result, error, live } = useLab(id);
  const [picked, setPicked] = useState<string | null>(null);
  const [deleteErr, setDeleteErr] = useState<string | null>(null);

  const rows = useMemo(() => Object.values(candidates), [candidates]);
  const selectedKey = picked ?? result?.winner ?? null;
  const selected = selectedKey ? candidates[selectedKey] : null;
  const slug = result?.deployment?.slug;
  const finished = stage === "done" || stage === "failed";

  const remove = async () => {
    if (!window.confirm("Delete this lab and its deployed model? API calls to the model will stop working.")) return;
    try {
      await api(`/api/labs/${id}`, { method: "DELETE" });
      router.replace("/app/labs");
    } catch (e) {
      setDeleteErr(e instanceof Error ? e.message : "Couldn't delete this lab.");
    }
  };

  if (!lab && error) {
    return (
      <div className="mx-auto max-w-[1180px] px-5 py-8 md:px-8">
        <p role="alert" className="text-[14px] text-danger">{error}</p>
        <Link href="/app/labs" className="mt-3 inline-block text-[13.5px] underline underline-offset-4">Back to labs</Link>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1180px] px-5 py-8 md:px-8">
      <Link href="/app/labs" className="inline-flex items-center gap-1.5 text-[13px] text-muted transition-colors hover:text-ink">
        <ArrowLeft className="h-3.5 w-3.5" /> Labs
      </Link>
      <div className="mt-3 flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 max-w-[780px]">
          <h1 className="text-[30px] font-light tracking-[-0.03em]">{spec?.name ?? lab?.name ?? "Starting lab"}</h1>
          <p className="mt-1 text-[14px] text-muted">{lab?.prompt}</p>
        </div>
        {finished && <Button variant="danger" size="sm" onClick={remove}><Trash2 className="h-3.5 w-3.5" /> Delete</Button>}
      </div>
      {deleteErr && <p role="alert" className="mt-2 text-[13px] text-danger">{deleteErr}</p>}
      {spec && <div className="mt-4"><SpecSummary spec={spec} /></div>}
      <div className="mt-6"><StageRail stage={stage} /></div>
      {error && <p role="alert" className="mt-4 rounded-xl bg-danger/5 px-4 py-3 text-[13.5px] text-danger">{error}</p>}

      {result && (
        <div className="mt-6">
          <WinnerCard result={result} spec={spec} evalSource={lab?.eval_source ?? dataset?.eval_source} nEval={lab?.n_eval ?? dataset?.test}
            onOpen={slug ? () => router.push(`/app/models/${slug}`) : undefined} />
        </div>
      )}

      <div className="mt-6 grid items-start gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-6">
          <Section title="Candidates">
            <div className="-mx-4 -my-2">
              <Leaderboard candidates={rows} target={spec?.target_accuracy} onSelect={setPicked} selected={selectedKey}
                evalSource={lab?.eval_source ?? dataset?.eval_source} nEval={lab?.n_eval ?? dataset?.test} />
            </div>
          </Section>
          {rows.some((c) => c.status === "done") && (
            <Section title="Accuracy vs cost">
              <ParetoChart candidates={rows} pareto={result?.pareto ?? []} target={spec?.target_accuracy} />
            </Section>
          )}
          {selected?.metrics.confusion && spec && (
            <Section title={`${selected.name}: where it goes wrong`}>
              {selected.metrics.notes && <p className="mb-4 text-[12.5px] leading-relaxed text-muted">{selected.metrics.notes}</p>}
              <Scorecard metrics={selected.metrics} className="mb-4" />
              <Confusion labels={spec.labels} matrix={selected.metrics.confusion} />
              <div className="mt-5 border-t border-line-soft pt-3">
                <ErrorList errors={selected.metrics.errors ?? []} />
              </div>
            </Section>
          )}
        </div>
        <div className="min-w-0 space-y-6">
          <Section title="Agent activity">
            <AgentFeed events={events} live={live && !finished} className="-mx-2 -my-2 max-h-[520px]" />
          </Section>
          {dataset && (
            <Section title="Dataset">
              <DatasetPanel dataset={dataset} spec={spec} />
            </Section>
          )}
        </div>
      </div>
    </div>
  );
}
