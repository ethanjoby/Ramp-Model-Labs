import { defineConfig, devices } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

/*
 * End-to-end tests run against a production build of the frontend, a fresh backend on its own
 * SQLite file, and the Firebase Auth emulator, so Google sign-in is exercised without real accounts.
 */

const ROOT = process.cwd();

function dotenvLocal(): Record<string, string> {
  const p = join(ROOT, ".env.local");
  if (!existsSync(p)) return {};
  return Object.fromEntries(
    readFileSync(p, "utf8").split("\n").filter((l) => l.includes("=") && !l.trim().startsWith("#"))
      .map((l) => { const i = l.indexOf("="); return [l.slice(0, i).trim(), l.slice(i + 1).trim()]; }),
  );
}

const local = dotenvLocal();
const PROJECT = process.env.E2E_FIREBASE_PROJECT ?? local.NEXT_PUBLIC_FIREBASE_PROJECT_ID ?? "rml-e2e";
const WEB = 3100;
const API = 8100;
const EMULATOR = 9099;

const firebaseEnv = {
  NEXT_PUBLIC_FIREBASE_PROJECT_ID: PROJECT,
  NEXT_PUBLIC_FIREBASE_API_KEY: local.NEXT_PUBLIC_FIREBASE_API_KEY ?? "fake-api-key",
  NEXT_PUBLIC_FIREBASE_APP_ID: local.NEXT_PUBLIC_FIREBASE_APP_ID ?? "1:1:web:e2e",
  NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN: local.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN ?? `${PROJECT}.firebaseapp.com`,
  NEXT_PUBLIC_FIREBASE_AUTH_EMULATOR: `http://127.0.0.1:${EMULATOR}`,
};

export default defineConfig({
  testDir: "./e2e",
  timeout: 150_000,
  expect: { timeout: 15_000 },
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? "github" : "list",
  use: { baseURL: `http://127.0.0.1:${WEB}`, trace: "retain-on-failure", screenshot: "only-on-failure" },
  projects: [
    { name: "setup", testMatch: /auth\.setup\.ts/ },
    { name: "chromium", use: { ...devices["Desktop Chrome"], storageState: "e2e/.auth/user.json" }, dependencies: ["setup"] },
  ],
  webServer: [
    {
      command: `firebase emulators:start --only auth --project ${PROJECT} --config e2e/firebase.json`,
      url: `http://127.0.0.1:${EMULATOR}/`, reuseExistingServer: true, timeout: 120_000,
    },
    {
      command: `mkdir -p .e2e && rm -f .e2e/labs.db && cd ../backend && uv run uvicorn app.main:app --port ${API}`,
      url: `http://127.0.0.1:${API}/health`, reuseExistingServer: !process.env.CI, timeout: 120_000,
      env: {
        FIREBASE_PROJECT_ID: PROJECT, FIREBASE_AUTH_EMULATOR_HOST: `127.0.0.1:${EMULATOR}`, FRONTEND_ORIGIN: `http://127.0.0.1:${WEB}`,
        DATABASE_URL: `sqlite:///${join(ROOT, ".e2e", "labs.db")}`, SECRET_KEY: "e2e-secret-not-for-production",
        ANTHROPIC_API_KEY: "", MODAL_ENABLED: "false",
      },
    },
    {
      command: `npx next build && npx next start -p ${WEB}`,
      url: `http://127.0.0.1:${WEB}/login`, reuseExistingServer: !process.env.CI, timeout: 400_000,
      env: { ...firebaseEnv, API_URL: `http://127.0.0.1:${API}`, NEXT_DIST_DIR: ".next-e2e" },
    },
  ],
});
