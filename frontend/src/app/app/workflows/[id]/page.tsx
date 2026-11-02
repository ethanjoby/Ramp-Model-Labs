import { Suspense } from "react";
import { WorkflowView } from "@/components/app/WorkflowView";

export default function Page() {
  return (
    <Suspense>
      <WorkflowView />
    </Suspense>
  );
}
