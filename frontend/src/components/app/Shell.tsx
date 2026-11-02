"use client";

import clsx from "clsx";
import Image from "next/image";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { AnimatePresence, motion } from "motion/react";
import { Boxes, FlaskConical, KeyRound, LogOut, Menu, MessageSquare, Plus, Settings, Workflow, X } from "lucide-react";
import { createContext, Suspense, useContext, useEffect, useRef, useState } from "react";
import { Logo } from "@/components/ui/Logo";
import { api } from "@/lib/api";
import type { User } from "@/lib/types";
import { EASE_OUT } from "@/lib/motion";

type Caps = { llm: boolean; modal: boolean; local_lm?: boolean };
const Ctx = createContext<{ user: User | null; caps: Caps | null }>({ user: null, caps: null });
export const useSession = () => useContext(Ctx);

const NAV = [
  { href: "/app", label: "Copilot", icon: MessageSquare, exact: true },
  { href: "/app/labs", label: "Labs", icon: FlaskConical },
  { href: "/app/models", label: "Models", icon: Boxes },
  { href: "/app/workflows", label: "Workflows", icon: Workflow },
  { href: "/app/keys", label: "API keys", icon: KeyRound },
  { href: "/app/settings", label: "Settings", icon: Settings },
];

function Nav({ onNavigate }: { onNavigate?: () => void }) {
  const path = usePathname();
  return (
    <nav className="space-y-0.5">
      {NAV.map((n) => {
        const active = n.exact ? path === n.href : path.startsWith(n.href);
        return (
          <Link key={n.href} href={n.href} onClick={onNavigate}
            className={clsx("relative flex h-9 items-center gap-2.5 rounded-lg px-2.5 text-[13.5px] transition-colors duration-150",
              active ? "text-ink" : "text-ink/55 hover:bg-ink/[0.04] hover:text-ink")}>
            {active && (
              <motion.span layoutId="nav-pill" className="absolute inset-0 rounded-lg bg-white shadow-[0_1px_2px_rgba(12,10,8,0.06),0_0_0_1px_var(--line-soft)]"
                transition={{ type: "spring", stiffness: 380, damping: 30 }} />
            )}
            <n.icon className="relative h-4 w-4" strokeWidth={1.75} />
            <span className="relative">{n.label}</span>
          </Link>
        );
      })}
    </nav>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [caps, setCaps] = useState<Caps | null>(null);
  const [open, setOpen] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);

  // The drawer is a dialog: it takes focus, closes on Escape and the page behind it does not scroll.
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => { document.body.style.overflow = prev; window.removeEventListener("keydown", onKey); };
  }, [open]);

  useEffect(() => {
    api<User>("/api/auth/me").then(setUser).catch(() => {});
    api<Caps>("/api/capabilities").then(setCaps).catch(() => {});
  }, []);

  const logout = async () => {
    await api("/api/auth/logout", { method: "POST" }).catch(() => {});
    // Loaded on demand so the Firebase SDK stays out of every app page's bundle.
    await import("@/lib/firebase").then((m) => m.signOutOfFirebase()).catch(() => {});
    router.replace("/login");
    router.refresh();
  };

  const side = (
    <div className="flex h-full flex-col px-3 py-4">
      <div className="px-2"><Logo /></div>
      <Link href="/app?new=1" onClick={() => setOpen(false)}
        className="mt-6 flex h-9 items-center gap-2 rounded-lg bg-solar px-3 text-[13.5px] font-medium text-ink transition-[background-color,transform] duration-150 hover:bg-solar-light active:scale-[0.98]">
        <Plus className="h-4 w-4" /> New model
      </Link>
      <div className="mt-5"><Suspense><Nav onNavigate={() => setOpen(false)} /></Suspense></div>
      <div className="mt-auto space-y-3">
        {caps && (
          <div className="rounded-xl border border-line-soft bg-white/60 p-3 text-[11.5px] leading-relaxed text-muted">
            <div className="mb-1.5 font-medium text-ink/80">Engines</div>
            <div className="flex items-center justify-between"><span>Local specialists</span><span className="text-smolder">on</span></div>
            <div className="flex items-center justify-between"><span>Local LM fine-tune</span><span className={caps.local_lm ? "text-smolder" : "text-ink/35"}>{caps.local_lm ? "on" : "off"}</span></div>
            <div className="flex items-center justify-between"><span>Claude agents</span><span className={caps.llm ? "text-smolder" : "text-ink/35"}>{caps.llm ? "on" : "off"}</span></div>
            <div className="flex items-center justify-between"><span>Modal GPU fine-tune</span><span className={caps.modal ? "text-smolder" : "text-ink/35"}>{caps.modal ? "on" : "off"}</span></div>
          </div>
        )}
        <div className="flex items-center gap-2.5 rounded-lg px-2 py-1.5">
          {user?.avatar_url ? (
            <Image src={user.avatar_url} alt="" width={28} height={28} referrerPolicy="no-referrer" className="h-7 w-7 rounded-full object-cover" />
          ) : (
            <div className="flex h-7 w-7 items-center justify-center rounded-full bg-ink text-[11px] font-medium text-solar">
              {user?.name?.[0]?.toUpperCase() ?? "·"}
            </div>
          )}
          <div className="min-w-0 flex-1">
            <div className="truncate text-[13px]">{user?.name ?? " "}</div>
            <div className="truncate text-[11.5px] text-muted">{user?.email ?? " "}</div>
          </div>
          <button onClick={logout} aria-label="Sign out" className="rounded-md p-1.5 text-ink/50 transition-colors hover:bg-ink/5 hover:text-ink">
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <Ctx.Provider value={{ user, caps }}>
      <div className="min-h-[100svh] bg-paper">
        <aside className="fixed inset-y-0 left-0 hidden w-[236px] border-r border-line-soft bg-paper md:block">{side}</aside>
        <div className="sticky top-0 z-30 flex items-center justify-between border-b border-line-soft bg-paper/90 px-4 py-3 backdrop-blur md:hidden">
          <Logo />
          <button onClick={() => setOpen(true)} aria-label="Open menu" aria-expanded={open} aria-controls="mobile-nav" className="rounded-md p-1.5"><Menu className="h-5 w-5" /></button>
        </div>
        <AnimatePresence>
          {open && (
            <>
              <motion.div className="fixed inset-0 z-40 bg-ink/30 md:hidden" onClick={() => setOpen(false)}
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />
              <motion.aside id="mobile-nav" role="dialog" aria-modal="true" aria-label="Navigation" className="fixed inset-y-0 left-0 z-50 w-[260px] bg-paper md:hidden"
                initial={{ x: -280 }} animate={{ x: 0, transition: { duration: 0.3, ease: EASE_OUT } }} exit={{ x: -280, transition: { duration: 0.2 } }}>
                <button ref={closeRef} onClick={() => setOpen(false)} aria-label="Close menu" className="absolute right-3 top-4 rounded-md p-1.5"><X className="h-4 w-4" /></button>
                {side}
              </motion.aside>
            </>
          )}
        </AnimatePresence>
        <main className="md:pl-[236px]">{children}</main>
      </div>
    </Ctx.Provider>
  );
}

export function PageHeader({ title, sub, right }: { title: string; sub?: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-[30px] font-light tracking-[-0.03em]">{title}</h1>
        {sub && <p className="mt-1 text-[14px] text-muted">{sub}</p>}
      </div>
      {right}
    </div>
  );
}
