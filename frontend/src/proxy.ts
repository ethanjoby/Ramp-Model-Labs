import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// Optimistic gate only: the API re-verifies the signed session on every request.
export function proxy(request: NextRequest) {
  const hasSession = request.cookies.has("rml_session");
  const { pathname } = request.nextUrl;
  if (pathname.startsWith("/app") && !hasSession) {
    const url = new URL("/login", request.url);
    url.searchParams.set("next", pathname);
    return NextResponse.redirect(url);
  }
  if ((pathname === "/login" || pathname === "/signup") && hasSession) {
    const url = new URL("/app", request.url);
    url.search = request.nextUrl.search;
    return NextResponse.redirect(url);
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/app/:path*", "/login", "/signup"],
};
