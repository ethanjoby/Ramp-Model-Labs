import { logoMark } from "@/lib/brand-image";

// The mark alone with no background, ink stroke, for light surfaces.
export function GET() {
  return logoMark("transparent");
}
