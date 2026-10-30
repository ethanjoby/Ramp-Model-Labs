import { Suspense } from "react";
import { AuthForm } from "@/components/app/AuthForm";

export const metadata = { title: "Sign in · Ramp Model Labs" };

export default function Page() {
  return (
    <Suspense>
      <AuthForm mode="login" />
    </Suspense>
  );
}
