import clsx from "clsx";
import Link from "next/link";
import type { ComponentProps } from "react";

type Variant = "solar" | "ink" | "ghost" | "outline" | "ghost-dark" | "danger";
const base =
  "inline-flex items-center justify-center gap-2 rounded-full font-medium tracking-[-0.01em] transition-[transform,background-color,color,box-shadow] duration-150 ease-[var(--ease-std)] active:scale-[0.97] disabled:pointer-events-none disabled:opacity-45 select-none";
const variants: Record<Variant, string> = {
  solar: "bg-solar text-ink hover:bg-solar-light shadow-[inset_0_-1px_0_rgba(0,0,0,0.12)]",
  ink: "bg-ink text-white hover:bg-ink-3",
  ghost: "text-ink hover:bg-ink/5",
  outline: "border border-line bg-white text-ink hover:border-ink/40",
  "ghost-dark": "text-white/80 hover:text-white hover:bg-white/10",
  danger: "text-danger hover:bg-danger/10",
};
const sizes = { sm: "h-8 px-3.5 text-[13px]", md: "h-10 px-5 text-sm", lg: "h-12 px-6 text-[15px]" };

type Common = { variant?: Variant; size?: keyof typeof sizes };

export function Button({ variant = "ink", size = "md", className, ...p }: Common & ComponentProps<"button">) {
  return <button className={clsx(base, variants[variant], sizes[size], className)} {...p} />;
}

export function ButtonLink({ variant = "ink", size = "md", className, ...p }: Common & ComponentProps<typeof Link>) {
  return <Link className={clsx(base, variants[variant], sizes[size], className)} {...p} />;
}
