"use client";

import { MotionConfig } from "motion/react";

// Reduced motion follows the OS setting: motion drops transform animations tree-wide and keeps opacity.
export function Providers({ children }: { children: React.ReactNode }) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}
