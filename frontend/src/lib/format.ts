// Formatting is pinned to en-US: locale-dependent output in prerendered HTML would mismatch on hydration.
const money = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const whole = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
export const num = (v: number) => whole.format(v);

export const pct = (v?: number | null, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);

export function usd(v?: number | null): string {
  if (v == null) return "—";
  if (v === 0) return "$0";
  if (v < 0.0001) return `$${v.toExponential(1).replace("e-", "e−")}`;
  if (v < 0.01) return `$${v.toFixed(5)}`;
  if (v < 1) return `$${v.toFixed(3)}`;
  return `$${money.format(v)}`;
}

export function ms(v?: number | null): string {
  if (v == null) return "—";
  if (v < 1) return `${(v * 1000).toFixed(0)}µs`;
  if (v < 1000) return `${v.toFixed(v < 10 ? 1 : 0)}ms`;
  return `${(v / 1000).toFixed(2)}s`;
}

export function multiple(v?: number | null): string {
  if (v == null) return "—";
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M×`;
  if (v >= 10_000) return `${Math.round(v / 1000)}k×`;
  if (v >= 100) return `${whole.format(Math.round(v))}×`;
  return `${v.toFixed(1)}×`;
}

export function size(kb?: number | null): string {
  if (kb == null) return "—";
  if (kb > 1024 * 1024) return `${(kb / 1024 / 1024).toFixed(1)} GB`;
  if (kb > 1024) return `${(kb / 1024).toFixed(1)} MB`;
  return `${kb.toFixed(0)} KB`;
}

export function ago(iso: string): string {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export const humanLabel = (l: string) => (l.includes("_") ? l.replace(/_/g, " ") : l);

/* ---- qualifiers: every score carries what it was scored on, every estimate says so */

/** "200 synthetic rows", "18 of your rows", or "" when the basis is unknown. */
export function evalRows(source?: string | null, n?: number | null): string {
  const kind = !source ? null : /synthetic/i.test(source) ? "synthetic" : /your|real|user|upload/i.test(source) ? "yours" : null;
  const count = n != null && n > 0 ? num(n) : null;
  if (kind === "yours") return count ? `${count} of your rows` : "your rows";
  if (kind === "synthetic") return count ? `${count} synthetic rows` : "synthetic rows";
  return count ? `${count} rows` : "";
}

/** "96.0% on 200 synthetic rows"; falls back to the bare percentage when nothing is known about the eval set. */
export function accuracyBasis(acc?: number | null, source?: string | null, n?: number | null): string {
  const rows = evalRows(source, n);
  return acc == null || !rows ? pct(acc) : `${pct(acc)} on ${rows}`;
}

type SavingsLike = { multiple?: number | null; frontier_measured?: boolean | null } | null | undefined;

export const EST_TITLE =
  "Estimated from list prices: no Claude key was set, so the frontier model did not run on this test set. Add ANTHROPIC_API_KEY and the comparison is measured.";
export const MEASURED_TITLE = "Measured: the frontier model ran on the same held-out rows.";

/** The savings multiple with an "est." marker whenever the frontier side was priced, not run. */
export function savingsLabel(s: SavingsLike): string {
  if (!s || s.multiple == null) return "—";
  return s.frontier_measured ? multiple(s.multiple) : `${multiple(s.multiple)} est.`;
}

export function savingsTitle(s: SavingsLike): string | undefined {
  if (!s || s.multiple == null) return undefined;
  return s.frontier_measured ? MEASURED_TITLE : EST_TITLE;
}
