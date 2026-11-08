// Ramp Model Labs — product demo shot list for AIOProductOS Studio (https://aioproductos.com/studio).
// Run through demo/.work/studio-run/driver.mjs. Two takes, concatenated afterwards:
//   TAKE A  BEATS=hero,copilot      starts a real lab on camera, ends before heavy training begins
//   TAKE B  BEATS=lab,model,workflow,outro   the finished lab, playground, workflow, outro (idle machine)
// env: LAB_ID (finished GL lab), WF_ID (finished workflow), BEATS, DRY=1 (no real lab), NAME, VOICE=1, MUSIC
const BASE = process.env.BASE || "http://127.0.0.1:3100";
const LAB_ID = process.env.LAB_ID;
const WF_ID = process.env.WF_ID;
const DRY = process.env.DRY === "1";
const PROMPT = "Categorize card transactions into GL codes, cheap enough to run on every swipe";

const THEME = {
  bg: "#1f1c19",            // a hair above pure black so Studio's dark-frame trimmer never eats a title card
  accent: "#e4f222",        // solar
  text: "#f4f2f0",          // paper
  font: '-apple-system, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif',
  brand: "ramp model labs",
};

export default async function shoot(call) {
  const beats = [];
  const beat = (name, fn) => beats.push({ name, fn });

  // ------------------------------------------------------------------ 1. hook
  beat("hero", async () => {
    await call("studio_goto", {
      url: `${BASE}/`, eyebrow: "RAMP MODEL LABS", title: "One prompt in. A specialist model out.",
      wait_for: 'input[aria-label="Describe the model you need"]', dwell_ms: 3200,
    });
    await call("studio_hold", { ms: 2600 });
    await call("studio_caption", { eyebrow: "THE IDEA", caption_text: "Describe a financial workflow in plain English." });
    await call("studio_hold", { ms: 2400 });
    await call("studio_type", { selector: 'input[aria-label="Describe the model you need"]', input_text: PROMPT, delay_ms: 34 });
    await call("studio_hold", { target: "form button[type=submit]", ms: 1400 });
    await call("studio_caption", { clear: true });
  });

  // ------------------------------------------------------------------ 2. the lab starts
  beat("copilot", async () => {
    const url = DRY ? `${BASE}/app` : `${BASE}/app?prompt=${encodeURIComponent(PROMPT)}`;
    await call("studio_goto", { url, eyebrow: "THE LAB", title: "Agents take it from here", wait_for: "textarea", dwell_ms: 2200 });
    await call("studio_caption", { eyebrow: "AGENTS", caption_text: "A planner writes the spec. A curator builds the dataset." });
    await call("studio_hold", { ms: DRY ? 5200 : 6800 });
    if (!DRY) {
      await call("studio_caption", { eyebrow: "LIVE", caption_text: "Every step is on the record while it happens." });
      await call("studio_click", { target: 'a[href^="/app/labs/lab_"]', settle_ms: 500 });
      await call("studio_hold", { ms: 1600 });
      await call("studio_zoom", { target: "Agent activity", zoom_scale: 1.3, duration_ms: 1300 });
      await call("studio_hold", { ms: 4200 });
      await call("studio_zoom", { reset: true, duration_ms: 900 });
    } else {
      await call("studio_type", { selector: "textarea", input_text: PROMPT, delay_ms: 30 });
      await call("studio_hold", { ms: 1500 });
    }
    await call("studio_caption", { clear: true });
  });

  // ------------------------------------------------------------------ 3. the finished lab
  beat("lab", async () => {
    await call("studio_goto", {
      url: `${BASE}/app/labs/${LAB_ID}`, eyebrow: "A FEW MINUTES LATER",
      title: process.env.N_CANDIDATES ? `${process.env.N_CANDIDATES} candidates. One held-out test.` : "Every candidate. One held-out test.",
      wait_for: "[data-testid=leaderboard] tbody tr", dwell_ms: 2600,
    });
    await call("studio_caption", { eyebrow: "CANDIDATES", caption_text: "Classical specialists, fine-tuned MiniLM and ModernBERT, a LoRA on a GPU, and prompted Claude." });
    await call("studio_zoom", { target: "[data-testid=leaderboard]", zoom_scale: 1.25, duration_ms: 1400 });
    await call("studio_hold", { ms: 4600 });
    await call("studio_caption", { eyebrow: "MEASURED", caption_text: "Every one is scored on the same held-out rows. Accuracy, latency and cost are measured, not estimated." });
    await call("studio_highlight", { target: "[data-testid=leaderboard] tbody tr:first-child", eyebrow: "WINNER", label: "Cheapest model that clears the bar" });
    await call("studio_hold", { ms: 4800 });
    await call("studio_highlight", { clear: true });
    await call("studio_zoom", { reset: true, duration_ms: 900 });
    await call("studio_caption", { eyebrow: "THE VERDICT", caption_text: "The judge ships the cheapest model that clears the bar, and explains the pick." });
    await call("studio_scroll", { to_target: "text=Cheaper than frontier", duration_ms: 1500 });
    await call("studio_zoom", { target: "Cheaper than frontier", zoom_scale: 1.5, duration_ms: 1300 });
    await call("studio_hold", { ms: 3800 });
    await call("studio_zoom", { reset: true, duration_ms: 900 });
    await call("studio_caption", { clear: true });
  });

  // ------------------------------------------------------------------ 4. playground + API
  beat("model", async () => {
    await call("studio_click", { target: "Open playground", settle_ms: 500 });
    await call("studio_hold", { ms: 1800 });
    await call("studio_caption", { eyebrow: "DEPLOYED", caption_text: "The winner is live behind an API key the moment it is picked." });
    await call("studio_type", {
      selector: 'textarea[placeholder="One input per line"]',
      input_text: "UBER *TRIP SAN FRANCISCO CA $23.10\nFIGMA MONTHLY $45.00\nSWEETGREEN NYC $14.75",
      delay_ms: 22,
    });
    await call("studio_click", { target: "Run model", settle_ms: 500 });
    await call("studio_hold", { ms: 1600 });
    await call("studio_caption", { eyebrow: "FAST AND CHEAP", caption_text: "Milliseconds a call. Fractions of a cent per thousand." });
    await call("studio_zoom", { target: "ul.divide-y", zoom_scale: 1.45, duration_ms: 1200 });
    await call("studio_hold", { ms: 4200 });
    await call("studio_zoom", { reset: true, duration_ms: 900 });
    await call("studio_caption", { eyebrow: "IT LEARNS", caption_text: "Corrections made here become labeled data. Retrain re-runs the whole lab with them." });
    await call("studio_scroll", { to_target: "text=Retrain", duration_ms: 1300 });
    await call("studio_highlight", { target: "Retrain", eyebrow: "FEEDBACK LOOP", label: "Corrections → retrain → a better successor" });
    await call("studio_hold", { ms: 3800 });
    await call("studio_highlight", { clear: true });
    await call("studio_caption", { clear: true });
  });

  // ------------------------------------------------------------------ 5. workflows
  beat("workflow", async () => {
    await call("studio_goto", {
      url: `${BASE}/app/workflows/${WF_ID}`, eyebrow: "WORKFLOWS", title: "A spreadsheet in. Decisions out.",
      wait_for: "text=Decided automatically", dwell_ms: 2600,
    });
    await call("studio_caption", { eyebrow: "FILES IN", caption_text: "Drop in a spreadsheet, an inbox export or a PDF. Every row gets decided by your best model." });
    await call("studio_hold", { ms: 4600 });
    await call("studio_caption", { eyebrow: "HUMANS WHERE IT COUNTS", caption_text: "The uncertain rows go to a person. Their decisions train the next model." });
    await call("studio_click", { target: "Needs review", settle_ms: 500 });
    await call("studio_hold", { ms: 3600 });
    await call("studio_caption", { eyebrow: "RESULTS OUT", caption_text: "Results export back to Excel with prediction, confidence and runner-up on every row." });
    await call("studio_highlight", { target: "Excel", eyebrow: "EXPORT", label: "Your columns + prediction, confidence, needs_review" });
    await call("studio_hold", { ms: 3800 });
    await call("studio_highlight", { clear: true });
    await call("studio_caption", { clear: true });
  });

  // ------------------------------------------------------------------ 6. outro
  beat("outro", async () => {
    await call("studio_goto", {
      url: `${BASE}/`, eyebrow: "RAMP MODEL LABS", title: "Stop renting a genius for data entry.",
      wait_for: 'input[aria-label="Describe the model you need"]', dwell_ms: 2400,
    });
    await call("studio_hold", { ms: 1200 });
    await call("studio_end_card", { title: "Ramp Model Labs", subtitle: "One prompt in. A specialist model out.", url: "Built at the Ramp hackathon" });
    await call("studio_hold", { ms: 3600 });
  });

  // ------------------------------------------------------------------ run
  await call("studio_start", {
    width: 1920, height: 1080, theme: THEME, color_scheme: "light", show_cursor: true,
    storage_state_path: process.env.AUTH || new URL("./.work/studio-run/auth.json", import.meta.url).pathname,
  });
  const only = process.env.BEATS ? new Set(process.env.BEATS.split(",")) : null;
  for (const b of beats) {
    if (only && !only.has(b.name)) continue;
    console.log(`\n=== beat: ${b.name} ===`);
    await b.fn();
  }
  if (process.env.CANCEL === "1") { await call("studio_cancel", {}); return; }
  await call("studio_finish", {
    name: process.env.NAME || "take",
    voiceover: process.env.VOICE === "1",
    voice: process.env.VOICE_NAME || "Samantha",
    subtitles: "srt",
    ...(process.env.MUSIC ? { music: process.env.MUSIC } : {}),
  });
}
