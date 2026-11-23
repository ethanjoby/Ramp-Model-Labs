"use client";

import dynamic from "next/dynamic";
import { AnimatePresence, motion, useReducedMotion, useScroll, useTransform } from "motion/react";
import { ArrowRight, CornerDownLeft } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Logo } from "@/components/ui/Logo";
import { ButtonLink } from "@/components/ui/Button";
import { BOOKING_URL, PROMPT_EVENT } from "@/lib/contact";
import { EASE_OUT } from "@/lib/motion";

const DistillScene = dynamic(() => import("@/components/three/DistillScene"), { ssr: false });

const EXAMPLES = [
  "Our support inbox gets 3,000 tickets a week and two people route them by hand.",
  "We pay a vendor per call to categorize our product catalog and it still gets half of them wrong.",
  "Claims sit for days before anyone decides if they're fast-track or need an adjuster.",
  "IT tickets all look urgent; we need P1 to P4 from the description alone.",
  "Our accountant codes 4,000 card transactions a month into GL categories by hand.",
  "We moderate marketplace listings for counterfeits and off-platform payments one at a time.",
];

export function Hero() {
  const section = useRef<HTMLElement>(null);
  const [value, setValue] = useState("");
  const [ex, setEx] = useState(0);
  const [focused, setFocused] = useState(false);
  const [hovered, setHovered] = useState(false);
  const reduced = useReducedMotion();

  // The copy hands over to the next section as the cloud collapses behind it.
  const { scrollYProgress } = useScroll({ target: section, offset: ["start start", "end start"] });
  const opacity = useTransform(scrollYProgress, [0, 0.55], [1, 0]);
  const y = useTransform(scrollYProgress, [0, 0.55], [0, -56]);

  // The example is a suggestion, not a slideshow: it holds still while someone reads it, types, or asked for less motion.
  useEffect(() => {
    if (focused || hovered || value || reduced) return;
    const id = setInterval(() => setEx((i) => (i + 1) % EXAMPLES.length), 3800);
    return () => clearInterval(id);
  }, [focused, hovered, value, reduced]);

  // The public site has no sign-in: the prompt is handed to the contact section, where it waits next to the booking calendar.
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const prompt = (value || EXAMPLES[ex]).trim();
    window.dispatchEvent(new CustomEvent(PROMPT_EVENT, { detail: prompt }));
    document.getElementById("contact")?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
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
          <a href="#own" className="transition-colors hover:text-white">Why open</a>
          <a href="#workflows" className="transition-colors hover:text-white">Workflows</a>
          <a href="#api" className="transition-colors hover:text-white">API</a>
          <a href="#contact" className="transition-colors hover:text-white">Contact</a>
        </nav>
        <ButtonLink href={BOOKING_URL} target="_blank" rel="noreferrer" variant="solar" size="sm">Book a call</ButtonLink>
      </header>

      {/* The entrance is CSS (.rise), so the server HTML is readable before hydration and without JS. */}
      <motion.div style={{ opacity, y }} className="mx-auto flex min-h-[calc(100svh-84px)] max-w-[1240px] flex-col justify-start px-5 pb-[38vh] pt-6 md:justify-center md:px-8 md:pb-24 md:pt-0">
        <h1 className="rise display max-w-[11ch] text-[clamp(50px,8.6vw,124px)] [--rise-y:28px] [animation-duration:0.8s]">
          Bring the problem. Keep the model.
        </h1>
        <p className="rise mt-7 max-w-[580px] text-[17px] leading-relaxed text-white/70 [animation-delay:120ms]">
          Tell us what you need help with, whatever the problem is. We build an AI system that solves it, on the
          best model for your job rather than the one a big vendor wants you renting by the token. It is trained from
          open weights, so you own it, keep training it, and run it yourself. Open source is the future.
        </p>

        <form
          onSubmit={submit}
          onPointerEnter={() => setHovered(true)}
          onPointerLeave={() => setHovered(false)}
          className="rise mt-10 flex w-full max-w-[660px] items-center gap-2 rounded-2xl bg-white p-2 pl-5 text-ink shadow-[0_24px_80px_-24px_rgba(0,0,0,0.6)] transition-shadow focus-within:shadow-[0_0_0_3px_rgba(228,242,34,0.9),0_24px_80px_-24px_rgba(0,0,0,0.6)] [animation-delay:220ms]"
        >
          <div className="relative min-w-0 flex-1">
            <input
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              aria-label="Describe the problem you need solved"
              className="h-11 w-full bg-transparent text-[15px] text-ink outline-none placeholder:text-transparent"
              placeholder="Describe the problem you need solved"
            />
            {!value && (
              <div data-testid="hero-example" className="pointer-events-none absolute inset-0 flex items-center overflow-hidden">
                <AnimatePresence mode="wait" initial={false}>
                  <motion.span
                    key={ex}
                    initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.3, ease: EASE_OUT }}
                    className="truncate text-[15px] text-ink/55"
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
            Tell us <CornerDownLeft className="h-4 w-4" />
          </button>
        </form>
        <p className="rise mt-4 hidden items-center gap-2 text-[13px] text-white/60 md:flex [animation-delay:400ms]">
          Press enter to bring this to a call, or
          <a href="#demo" className="inline-flex items-center gap-1 text-white/80 underline-offset-4 hover:text-white hover:underline">
            watch a recorded run <ArrowRight className="h-3.5 w-3.5" />
          </a>
        </p>
      </motion.div>
    </section>
  );
}
