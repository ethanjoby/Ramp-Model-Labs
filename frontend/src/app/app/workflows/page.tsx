import { Suspense } from "react";
import { WorkflowsList } from "@/components/app/WorkflowsList";

export default function Page() {
  return (
    <Suspense>
      <WorkflowsList />
    </Suspense>
  );
}
