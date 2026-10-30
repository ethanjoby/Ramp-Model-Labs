// One personality, used everywhere: "corporate" from the motion-design skill.
export const EASE = [0.2, 0, 0, 1] as const; // moving on screen
export const EASE_OUT = [0.05, 0.7, 0.1, 1] as const; // entering
export const EASE_IN = [0.3, 0, 1, 1] as const; // exiting
export const DUR = { fast: 0.2, base: 0.3, slow: 0.45, page: 0.55 } as const;

export const enter = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0, transition: { duration: DUR.slow, ease: EASE_OUT } },
  exit: { opacity: 0, y: -6, transition: { duration: DUR.fast, ease: EASE_IN } },
};

export const spring = { type: "spring", stiffness: 300, damping: 22 } as const;
