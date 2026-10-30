import { Suspense } from "react";
import { ModelView } from "@/components/app/ModelView";

export default function Page() {
  return (
    <Suspense>
      <ModelView />
    </Suspense>
  );
}
