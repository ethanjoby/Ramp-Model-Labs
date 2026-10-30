import { Shell } from "@/components/app/Shell";

export const metadata = { title: "Ramp Model Labs" };

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return <Shell>{children}</Shell>;
}
