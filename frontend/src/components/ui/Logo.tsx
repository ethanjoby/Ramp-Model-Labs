import Link from "next/link";
import clsx from "clsx";

/*
 * The mark is the hero scene flattened: a wide structure (the general model) holding a small,
 * solid, bright core (the specialist). Pointy-top hexagons share one center; three struts tie them.
 */
const OUTER = "16 3 27.26 9.5 27.26 22.5 16 29 4.74 22.5 4.74 9.5";
const INNER = "16 10.8 20.5 13.4 20.5 18.6 16 21.2 11.5 18.6 11.5 13.4";

export function Mark({ className, strokeWidth = 2.4 }: { className?: string; strokeWidth?: number }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden fill="none">
      <polygon points={OUTER} stroke="currentColor" strokeWidth={strokeWidth} strokeLinejoin="round" />
      <path d="M16 3v7.8M27.26 22.5l-6.76-3.9M4.74 22.5l6.76-3.9" stroke="currentColor" strokeWidth={strokeWidth * 0.62} strokeLinecap="round" opacity=".6" />
      <polygon points={INNER} fill="var(--solar)" />
    </svg>
  );
}

export function Logo({ dark = false, className, compact = false }: { dark?: boolean; className?: string; compact?: boolean }) {
  return (
    <Link href="/" className={clsx("group inline-flex items-center gap-2.5", className)} aria-label="Ramp Model Labs home">
      <Mark className={clsx("h-[26px] w-[26px] transition-transform duration-200 ease-[var(--ease-std)] group-hover:rotate-[30deg]", dark ? "text-white" : "text-ink")} />
      {!compact && (
        <span className={clsx("text-[15.5px] tracking-[-0.025em]", dark ? "text-white" : "text-ink")}>
          <span className="font-semibold">ramp</span>
          <span className={clsx("ml-1.5 font-normal", dark ? "text-white/60" : "text-ink/55")}>model labs</span>
        </span>
      )}
    </Link>
  );
}
