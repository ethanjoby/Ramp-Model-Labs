export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

function detail(body: unknown): string {
  if (!body || typeof body !== "object") return "Something went wrong.";
  const d = (body as { detail?: unknown }).detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d) && d[0]?.msg) return String(d[0].msg).replace(/^Value error, /, "");
  return "Something went wrong.";
}

export async function api<T = unknown>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const { json, headers, ...rest } = init;
  const res = await fetch(path, {
    credentials: "same-origin",
    ...rest,
    // The custom header is the CSRF token: no cross-site page can send it without a preflight we never grant.
    headers: { "X-Requested-With": "rml", ...(json !== undefined ? { "Content-Type": "application/json" } : {}), ...headers },
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  });
  const text = await res.text();
  const body = text ? (() => { try { return JSON.parse(text); } catch { return text; } })() : null;
  if (res.status === 401 && typeof window !== "undefined" && window.location.pathname.startsWith("/app")) {
    const next = window.location.pathname + window.location.search;
    window.location.assign(new URL(`/login?next=${encodeURIComponent(next)}`, window.location.origin).href);
    // The page is on its way to the login screen; settling this promise would only flash an error first.
    return new Promise<T>(() => {});
  }
  if (!res.ok) throw new ApiError(res.status, detail(body));
  return body as T;
}
