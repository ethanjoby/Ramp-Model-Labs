export const pct = (v?: number | null, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);

export function usd(v?: number | null): string {
  if (v == null) return "—";
  if (v === 0) return "$0";
  if (v < 0.0001) return `$${v.toExponential(1).replace("e-", "e−")}`;
  if (v < 0.01) return `$${v.toFixed(5)}`;
  if (v < 1) return `$${v.toFixed(3)}`;
  return `$${v.toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 })}`;
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
  if (v >= 100) return `${Math.round(v).toLocaleString()}×`;
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
