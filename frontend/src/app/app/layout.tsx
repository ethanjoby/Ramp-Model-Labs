import { Shell } from "@/components/app/Shell";

export const metadata = { title: "Ascent Labs" };

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return <Shell>{children}</Shell>;
}
