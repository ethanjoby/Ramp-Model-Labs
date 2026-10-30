"use client";

import dynamic from "next/dynamic";
import { useRouter, useSearchParams } from "next/navigation";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { Logo } from "@/components/ui/Logo";
import { Spinner } from "@/components/ui/bits";
import { api } from "@/lib/api";
import { firebaseConfigured, pendingRedirectToken, signInWithGoogle } from "@/lib/firebase";
import { EASE_OUT } from "@/lib/motion";

const DistillScene = dynamic(() => import("@/components/three/DistillScene"), { ssr: false });

function safeNext(n: string | null) {
  return n && n.startsWith("/app") && !n.startsWith("//") ? n : "/app";
}

function GoogleMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 48 48" className={className} aria-hidden>
      <path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9.1 3.6l6.8-6.8C35.8 2.4 30.3 0 24 0 14.6 0 6.5 5.4 2.6 13.3l7.9 6.1C12.4 13.6 17.7 9.5 24 9.5z" />
      <path fill="#4285F4" d="M46.5 24.5c0-1.6-.1-3.1-.4-4.5H24v8.5h12.7c-.6 3-2.3 5.5-4.8 7.2l7.6 5.9c4.5-4.1 7-10.2 7-17.1z" />
      <path fill="#FBBC05" d="M10.5 28.6A14.5 14.5 0 0 1 9.7 24c0-1.6.3-3.1.8-4.6l-7.9-6.1A24 24 0 0 0 0 24c0 3.9.9 7.5 2.6 10.7l7.9-6.1z" />
      <path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.6-5.9c-2.1 1.4-4.9 2.3-8.3 2.3-6.3 0-11.6-4.1-13.5-9.9l-7.9 6.1C6.5 42.6 14.6 48 24 48z" />
    </svg>
  );
}

export function AuthForm({ mode }: { mode: "login" | "signup" }) {
  const router = useRouter();
  const params = useSearchParams();
  const prompt = params.get("prompt");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [shake, setShake] = useState(0);

  const dest = prompt ? `/app?prompt=${encodeURIComponent(prompt)}` : safeNext(params.get("next"));

  const finish = async (idToken: string) => {
    await api("/api/auth/firebase", { method: "POST", json: { id_token: idToken } });
    router.replace(dest);
    router.refresh();
  };

  const fail = (e: unknown) => {
    setErr(e instanceof Error ? e.message : "Something went wrong.");
    setShake((s) => s + 1);
    setBusy(false);
  };

  // A popup-blocked browser came back here via redirect; pick up where it left off.
  useEffect(() => {
    pendingRedirectToken().then((t) => {
      if (!t) return;
      setBusy(true);
      finish(t).catch(fail);
    }).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const signIn = async () => {
    setErr(null);
    setBusy(true);
    try {
      await finish(await signInWithGoogle());
    } catch (e) {
      fail(e);
    }
  };

  return (
    <div className="grid min-h-[100svh] md:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
      <div className="flex flex-col px-6 py-6 md:px-12">
        <Logo />
        <div className="mx-auto flex w-full max-w-[400px] flex-1 flex-col justify-center py-12">
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, ease: EASE_OUT }}>
            <h1 className="display text-[46px]">{mode === "login" ? "Welcome back." : "Create your lab."}</h1>
            <p className="mt-3 text-[15px] text-muted">
              {prompt ? "Sign in with Google and your model starts training right away." :
                mode === "login" ? "Sign in to your models and labs." : "Your first specialized model is one prompt away."}
            </p>
            {prompt && (
              <div className="mt-5 rounded-xl border border-line-soft bg-white p-3.5 text-[13.5px] leading-relaxed text-ink/75">
                <span className="mb-1 block text-[12px] text-muted">Queued prompt</span>
                {prompt}
              </div>
            )}
          </motion.div>

          <motion.div key={shake} animate={shake ? { x: [0, -10, 9, -6, 0] } : {}} transition={{ duration: 0.34 }} className="mt-8">
            <button
              onClick={signIn}
              disabled={busy || !firebaseConfigured}
              className="flex h-12 w-full items-center justify-center gap-3 rounded-full bg-ink px-5 text-[15px] font-medium text-white transition-[background-color,transform] duration-150 hover:bg-ink-3 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50"
            >
              {busy ? <Spinner /> : <><span className="flex h-6 w-6 items-center justify-center rounded-full bg-white"><GoogleMark className="h-3.5 w-3.5" /></span> Continue with Google</>}
            </button>
            {err && <p role="alert" className="mt-3 text-[13.5px] text-danger">{err}</p>}
            {!firebaseConfigured && (
              <p className="mt-3 text-[13px] text-muted">
                Google sign-in isn&apos;t configured for this build. Add the <code className="font-mono text-[12px]">NEXT_PUBLIC_FIREBASE_*</code> variables and restart.
              </p>
            )}
          </motion.div>
          <p className="mt-6 text-[13px] leading-relaxed text-muted">
            We read your name, email and photo from Google and nothing else. New accounts are created on first sign-in.
          </p>
        </div>
        <p className="text-[12px] text-muted">Sessions are httpOnly, signed, and revocable from Settings.</p>
      </div>
      <div className="dark relative hidden overflow-hidden bg-ink md:block">
        <DistillScene density={0.7} className="absolute inset-0" interactive={false} />
        <div className="absolute bottom-10 left-10 right-10 text-white">
          <p className="display max-w-[16ch] text-[34px] text-white/90">The big model teaches. The small model works.</p>
        </div>
      </div>
    </div>
  );
}
