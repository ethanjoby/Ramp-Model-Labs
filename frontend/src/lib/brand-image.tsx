import { ImageResponse } from "next/og";

/*
 * The brand rendered as a bitmap, for places that cannot run the site: link previews in Messages, Slack and
 * social cards (1200x630) and the Apple touch icon (180x180). Same mark as components/ui/Logo.tsx: a wide
 * hexagon (the general model) holding a small solid solar core (the specialist).
 */
const INK = "#0c0a08";
const SOLAR = "#e4f222";
const OUTER = "16 3 27.26 9.5 27.26 22.5 16 29 4.74 22.5 4.74 9.5";
const INNER = "16 10.8 20.5 13.4 20.5 18.6 16 21.2 11.5 18.6 11.5 13.4";

function Mark({ px, stroke = 2.4, color = "white" }: { px: number; stroke?: number; color?: string }) {
  return (
    <svg width={px} height={px} viewBox="0 0 32 32" fill="none">
      <polygon points={OUTER} stroke={color} strokeWidth={stroke} strokeLinejoin="round" />
      <path d="M16 3v7.8M27.26 22.5l-6.76-3.9M4.74 22.5l6.76-3.9" stroke={color} strokeWidth={stroke * 0.62} strokeLinecap="round" opacity="0.6" />
      <polygon points={INNER} fill={SOLAR} />
    </svg>
  );
}

export const OG_SIZE = { width: 1200, height: 630 };
export const ICON_SIZE = { width: 180, height: 180 };

/** The social card: wordmark, the headline, one line of what we do. */
export function socialCard() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%", height: "100%", display: "flex", flexDirection: "column", justifyContent: "space-between",
          padding: "64px 72px", background: INK, color: "white", fontFamily: "sans-serif",
          backgroundImage: `radial-gradient(circle at 82% 50%, rgba(228,242,34,0.22) 0%, rgba(228,242,34,0.06) 22%, rgba(12,10,8,0) 48%)`,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
          <Mark px={56} />
          <div style={{ display: "flex", fontSize: 34, letterSpacing: -1 }}>
            <span style={{ fontWeight: 700 }}>ascent</span>
            <span style={{ marginLeft: 12, color: "rgba(255,255,255,0.6)" }}>labs</span>
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 26 }}>
          <div style={{ fontSize: 86, lineHeight: 1.0, letterSpacing: -4, fontWeight: 300, maxWidth: 900 }}>
            Bring the problem. Keep the model.
          </div>
          <div style={{ fontSize: 30, lineHeight: 1.3, color: "rgba(255,255,255,0.7)", maxWidth: 860 }}>
            We build the AI system that solves it, on the best open model for the job, and you keep the weights.
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 14, fontSize: 22, color: "rgba(255,255,255,0.55)" }}>
          <div style={{ width: 12, height: 12, borderRadius: 999, background: SOLAR }} />
          Specialist models you own, built from open weights
        </div>
      </div>
    ),
    OG_SIZE,
  );
}

/** The home-screen / touch icon: the mark alone on ink. iOS rounds the corners itself. */
export function touchIcon() {
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", background: INK }}>
        <Mark px={124} stroke={2.2} />
      </div>
    ),
    ICON_SIZE,
  );
}
