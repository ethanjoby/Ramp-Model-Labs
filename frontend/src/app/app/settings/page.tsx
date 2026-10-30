"use client";

import { useEffect, useState } from "react";
import { PageHeader, useSession } from "@/components/app/Shell";
import { Button } from "@/components/ui/Button";
import { Badge, Card, FAMILY_LABEL, FamilyDot } from "@/components/ui/bits";
import { api } from "@/lib/api";

type Strategy = { key: string; name: string; family: string; blurb: string; available: boolean; reason: string };

export default function SettingsPage() {
  const { user } = useSession();
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api<{ strategies: Strategy[] }>("/api/capabilities").then((c) => setStrategies(c.strategies)).catch((e) => setErr(e.message));
  }, []);

  const logoutAll = async () => {
    if (!window.confirm("Sign out of every device, including this one?")) return;
    try {
      await api("/api/auth/logout-all", { method: "POST" });
      window.location.href = "/login";
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Couldn't sign out.");
    }
  };

  return (
    <div className="mx-auto max-w-[760px] px-5 py-8 md:px-8">
      <PageHeader title="Settings" />
      {err && <p role="alert" className="mt-4 text-[13.5px] text-danger">{err}</p>}

      <Card className="mt-8 p-5">
        <div className="flex items-center justify-between gap-4">
          <h2 className="text-[13px] font-medium">Account</h2>
          <Badge tone="neutral">{user?.provider === "google" ? "Signed in with Google" : "Signed in"}</Badge>
        </div>
        <dl className="mt-3 space-y-2 text-[14px]">
          <div className="flex justify-between gap-4"><dt className="text-muted">Name</dt><dd className="truncate">{user?.name}</dd></div>
          <div className="flex justify-between gap-4"><dt className="text-muted">Email</dt><dd className="truncate">{user?.email}</dd></div>
        </dl>
      </Card>

      <Card className="mt-4 overflow-hidden">
        <h2 className="px-5 pt-5 text-[13px] font-medium">Training strategies</h2>
        <p className="mt-1 px-5 text-[13.5px] text-muted">What each lab can try on this server.</p>
        <ul className="mt-3 divide-y divide-line-soft border-t border-line-soft">
          {strategies.map((st) => (
            <li key={st.key} className="flex items-start justify-between gap-4 px-5 py-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2 text-[13.5px]">
                  <FamilyDot family={st.family} /> {st.name}
                  <span className="text-[11.5px] text-muted">{FAMILY_LABEL[st.family] ?? st.family}</span>
                </div>
                <p className="mt-0.5 text-[12.5px] leading-relaxed text-muted">{st.available ? st.blurb : st.reason}</p>
              </div>
              <Badge tone={st.available ? "smolder" : "neutral"}>{st.available ? "on" : "off"}</Badge>
            </li>
          ))}
        </ul>
      </Card>

      <Card className="mt-4 flex items-center justify-between gap-4 p-5">
        <div>
          <h2 className="text-[13px] font-medium">Sessions</h2>
          <p className="mt-1 text-[13.5px] text-muted">Revoke every active session on all devices.</p>
        </div>
        <Button variant="danger" size="sm" onClick={logoutAll}>Sign out everywhere</Button>
      </Card>
    </div>
  );
}
