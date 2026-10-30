"use client";

import { Check, Copy } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { PageHeader } from "@/components/app/Shell";
import { Button } from "@/components/ui/Button";
import { Card, Spinner } from "@/components/ui/bits";
import { api } from "@/lib/api";
import { ago } from "@/lib/format";

type Key = { id: string; name: string; prefix: string; created_at: string; last_used_at: string | null };

export default function KeysPage() {
  const [keys, setKeys] = useState<Key[] | null>(null);
  const [name, setName] = useState("");
  const [fresh, setFresh] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(() => {
    api<Key[]>("/api/keys").then(setKeys).catch((e) => setErr(e.message));
  }, []);

  useEffect(load, [load]);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const r = await api<{ key: string }>("/api/keys", { method: "POST", json: { name: name.trim() } });
      setFresh(r.key);
      setCopied(false);
      setName("");
      load();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Couldn't create the key.");
    } finally {
      setBusy(false);
    }
  };

  const revoke = async (k: Key) => {
    if (!window.confirm(`Revoke "${k.name}"? Anything using this key stops working immediately.`)) return;
    try {
      await api(`/api/keys/${k.id}`, { method: "DELETE" });
      load();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Couldn't revoke the key.");
    }
  };

  const copy = async () => {
    if (!fresh) return;
    await navigator.clipboard.writeText(fresh);
    setCopied(true);
  };

  return (
    <div className="mx-auto max-w-[760px] px-5 py-8 md:px-8">
      <PageHeader title="API keys" sub="Call your deployed models from your own code." />

      <form onSubmit={create} className="mt-8 flex gap-2">
        <input value={name} onChange={(e) => setName(e.target.value)} required maxLength={60} placeholder="Key name, like production"
          aria-label="Key name"
          className="h-10 min-w-0 flex-1 rounded-full border border-line bg-white px-4 text-[14px] outline-none transition-[border-color,box-shadow] duration-150 placeholder:text-ink/30 focus:border-ink focus:shadow-[0_0_0_3px_rgba(228,242,34,0.55)]" />
        <Button type="submit" variant="ink" disabled={busy || !name.trim()}>{busy ? <Spinner /> : "Create key"}</Button>
      </form>
      {err && <p role="alert" className="mt-3 text-[13.5px] text-danger">{err}</p>}

      {fresh && (
        <div className="mt-4 rounded-2xl bg-ink p-4 text-white">
          <p className="text-[13px] text-solar">Copy this key now. You won&apos;t be able to see it again.</p>
          <div className="mt-2 flex items-center gap-2">
            <code className="min-w-0 flex-1 break-all font-mono text-[12.5px]">{fresh}</code>
            <button onClick={copy} aria-label="Copy key" className="shrink-0 rounded-md p-1.5 text-white/70 transition-colors hover:bg-white/10 hover:text-white">
              {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
            </button>
          </div>
        </div>
      )}

      {!keys && !err && <div className="skeleton mt-6 h-24 rounded-2xl" />}
      {keys?.length === 0 && <p className="mt-6 text-[13.5px] text-muted">No active keys.</p>}
      {!!keys?.length && (
        <Card className="mt-6 overflow-hidden">
          <ul className="divide-y divide-line-soft">
            {keys.map((k) => (
              <li key={k.id} className="flex items-center justify-between gap-4 px-4 py-3">
                <div className="min-w-0">
                  <div className="truncate text-[14px]">{k.name}</div>
                  <div className="tnum truncate font-mono text-[11.5px] text-muted">
                    {k.prefix}… · created {ago(k.created_at)} · {k.last_used_at ? `used ${ago(k.last_used_at)}` : "never used"}
                  </div>
                </div>
                <Button variant="danger" size="sm" onClick={() => void revoke(k)}>Revoke</Button>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}
