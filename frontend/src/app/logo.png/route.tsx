import { logoLockup } from "@/lib/brand-image";

// A stable link for the logo (cached by Next). Use it from emails, booking pages and profiles.
export function GET() {
  return logoLockup("dark");
}
