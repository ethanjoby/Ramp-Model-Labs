import { Suspense } from "react";
import { LabView } from "@/components/app/LabView";

export default function Page() {
  return (
    <Suspense>
      <LabView />
    </Suspense>
  );
}
