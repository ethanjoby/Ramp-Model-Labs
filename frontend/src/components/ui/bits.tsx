import clsx from "clsx";
import type { ReactNode } from "react";

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={clsx("rounded-2xl border border-line-soft bg-white", className)}>{children}</div>;
}

export function Badge({ tone = "neutral", children, className }: {
  tone?: "neutral" | "solar" | "ink" | "spring" | "danger" | "smolder"; children: ReactNode; className?: string;
}) {
  const tones = {
    neutral: "bg-paper-2 text-ink/70",
    solar: "bg-solar text-ink",
    ink: "bg-ink text-white",
    spring: "bg-spring-light text-[#244a8c]",
    danger: "bg-danger/10 text-danger",
    smolder: "bg-smolder text-solar",
  };
  return (
    <span className={clsx("inline-flex h-[22px] items-center gap-1 rounded-full px-2 text-[11.5px] font-medium", tones[tone], className)}>
      {children}
    </span>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={clsx("h-4 w-4 animate-spin", className)} aria-hidden>
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeOpacity=".2" strokeWidth="2.5" />
      <path d="M21 12a9 9 0 0 0-9-9" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  );
}

export function Stat({ label, value, hint, className }: { label: string; value: ReactNode; hint?: ReactNode; className?: string }) {
  return (
    <div className={className}>
      <div className="text-[12px] text-muted">{label}</div>
      <div className="tnum mt-1 text-[26px] font-light tracking-[-0.03em] leading-none">{value}</div>
      {hint && <div className="mt-1.5 text-[12px] text-muted">{hint}</div>}
    </div>
  );
}

export const FAMILY_LABEL: Record<string, string> = {
  classical: "Specialist",
  prompted: "Prompted",
  finetuned: "Fine-tuned",
  reference: "Frontier",
};

export function FamilyDot({ family }: { family: string }) {
  const c = { classical: "bg-ink", prompted: "bg-spring", finetuned: "bg-solar-deep", reference: "bg-muted" }[family] ?? "bg-muted";
  return <span className={clsx("inline-block h-2 w-2 rounded-full", c)} aria-hidden />;
}
