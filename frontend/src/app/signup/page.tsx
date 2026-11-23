import { Suspense } from "react";
import { AuthForm } from "@/components/app/AuthForm";

export const metadata = { title: "Sign up · Ascent Labs" };

export default function Page() {
  return (
    <Suspense>
      <AuthForm mode="signup" />
    </Suspense>
  );
}
