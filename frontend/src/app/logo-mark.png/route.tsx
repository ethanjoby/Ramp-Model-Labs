import { logoMark } from "@/lib/brand-image";

// Same as /logo.png; kept so older links keep working.
export function GET() {
  return logoMark("dark");
}
