import { OG_SIZE, socialCard } from "@/lib/brand-image";

export const alt = "Ascent Labs: bring the problem, keep the model";
export const size = OG_SIZE;
export const contentType = "image/png";

export default function Image() {
  return socialCard();
}
