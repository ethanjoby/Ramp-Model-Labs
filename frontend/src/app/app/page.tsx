import { Suspense } from "react";
import { Copilot } from "@/components/app/Copilot";

export default function Page() {
  return (
    <Suspense>
      <Copilot />
    </Suspense>
  );
}
