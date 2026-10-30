"use client";

import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { AnimatePresence, motion, useScroll, useTransform } from "motion/react";
import { ArrowRight, CornerDownLeft } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Logo } from "@/components/ui/Logo";
import { ButtonLink } from "@/components/ui/Button";
import { EASE_OUT } from "@/lib/motion";

const DistillScene = dynamic(() => import("@/components/three/DistillScene"), { ssr: false });

const EXAMPLES = [
  "Categorize corporate card transactions into our 12 GL expense codes. Keep it cheap.",
  "Flag card transactions that look like fraud from the risk signals. Needs 90% accuracy.",
  "Route inbound support tickets to card, reimbursement, bill pay, or accounting queues.",
  "Triage AP invoices: approve, needs review, likely duplicate, or reject.",
  "Reconcile bank lines against the ledger and flag amount mismatches, timing gaps, and duplicates.",
];

export function Hero() {
  const router = useRouter();
  const section = useRef<HTMLElement>(null);
  const [value, setValue] = useState("");
  const [ex, setEx] = useState(0);
  const [focused, setFocused] = useState(false);

  // The copy hands over to the next section as the cloud collapses behind it.
  const { scrollYProgress } = useScroll({ target: section, offset: ["start start", "end start"] });
  const opacity = useTransform(scrollYProgress, [0, 0.55], [1, 0]);
  const y = useTransform(scrollYProgress, [0, 0.55], [0, -56]);

  useEffect(() => {
    if (focused || value) return;
    const id = setInterval(() => setEx((i) => (i + 1) % EXAMPLES.length), 3800);
    return () => clearInterval(id);
  }, [focused, value]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const prompt = (value || EXAMPLES[ex]).trim();
    router.push(`/signup?prompt=${encodeURIComponent(prompt)}`);
  };

  return (
    <section ref={section} className="dark relative isolate min-h-[100svh] overflow-hidden bg-ink text-white">
      <DistillScene className="absolute inset-0 -z-10" offset={[1.7, 0.05, 0]} offsetMobile={[0, -2.4, 0]} collapseOnScroll />
      <div className="pointer-events-none absolute inset-0 -z-10 bg-[linear-gradient(180deg,rgba(12,10,8,0.85)_0%,rgba(12,10,8,0.55)_55%,rgba(12,10,8,0.1)_100%)] md:bg-[linear-gradient(90deg,rgba(12,10,8,0.95)_0%,rgba(12,10,8,0.75)_34%,rgba(12,10,8,0.12)_62%,transparent_80%)]" />

      <header className="mx-auto flex max-w-[1240px] items-center justify-between px-5 py-5 md:px-8">
        <Logo dark />
        <nav className="hidden items-center gap-7 text-[13.5px] text-white/65 md:flex">
          <a href="#demo" className="transition-colors hover:text-white">See a run</a>
          <a href="#how" className="transition-colors hover:text-white">How it works</a>
          <a href="#cost" className="transition-colors hover:text-white">Cost</a>
          <a href="#workflows" className="transition-colors hover:text-white">Workflows</a>
          <a href="#api" className="transition-colors hover:text-white">API</a>
        </nav>
        <div className="flex items-center gap-1.5">
          <ButtonLink href="/login" variant="ghost-dark" size="sm">Sign in</ButtonLink>
          <ButtonLink href="/signup" variant="solar" size="sm">Start building</ButtonLink>
        </div>
      </header>

      <motion.div style={{ opacity, y }} className="mx-auto flex min-h-[calc(100svh-84px)] max-w-[1240px] flex-col justify-start px-5 pb-[38vh] pt-6 md:justify-center md:px-8 md:pb-24 md:pt-0">
        <motion.h1
          initial={{ opacity: 0, y: 28 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, ease: EASE_OUT }}
          className="display max-w-[11ch] text-[clamp(50px,8.6vw,124px)]"
        >
          One prompt in. A specialist model out.
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: EASE_OUT, delay: 0.12 }}
          className="mt-7 max-w-[560px] text-[17px] leading-relaxed text-white/70"
        >
          Describe a financial workflow in plain English. A team of agents writes the spec, builds the data,
          trains and fine-tunes competing models, scores every one on held-out examples, and ships the cheapest
          that clears your bar.
        </motion.p>

        <motion.form
          onSubmit={submit}
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: EASE_OUT, delay: 0.22 }}
          className="mt-10 flex w-full max-w-[660px] items-center gap-2 rounded-2xl bg-white p-2 pl-5 text-ink shadow-[0_24px_80px_-24px_rgba(0,0,0,0.6)] transition-shadow focus-within:shadow-[0_0_0_3px_rgba(228,242,34,0.9),0_24px_80px_-24px_rgba(0,0,0,0.6)]"
        >
          <div className="relative min-w-0 flex-1">
            <input
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              aria-label="Describe the model you need"
              className="h-11 w-full bg-transparent text-[15px] text-ink outline-none placeholder:text-transparent"
              placeholder="Describe the model you need"
            />
            {!value && (
              <div className="pointer-events-none absolute inset-0 flex items-center overflow-hidden">
                <AnimatePresence mode="wait">
                  <motion.span
                    key={ex}
                    initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.3, ease: EASE_OUT }}
                    className="truncate text-[15px] text-ink/45"
                  >
                    {EXAMPLES[ex]}
                  </motion.span>
                </AnimatePresence>
              </div>
            )}
          </div>
          <button
            type="submit"
            className="inline-flex h-11 shrink-0 items-center gap-2 rounded-xl bg-ink px-4 text-sm font-medium text-white transition-[background-color,transform] duration-150 hover:bg-ink-3 active:scale-[0.97]"
          >
            Build it <CornerDownLeft className="h-4 w-4" />
          </button>
        </motion.form>
        <motion.p
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-4 hidden items-center gap-2 text-[13px] text-white/50 md:flex"
        >
          Press enter to try the example, or
          <a href="#demo" className="inline-flex items-center gap-1 text-white/80 underline-offset-4 hover:text-white hover:underline">
            watch a recorded run <ArrowRight className="h-3.5 w-3.5" />
          </a>
        </motion.p>
      </motion.div>
    </section>
  );
}
