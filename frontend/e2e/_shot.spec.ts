import { test, expect } from "@playwright/test";
import fs from "node:fs";
test.use({ storageState: { cookies: [], origins: [] } });
test("logo links serve PNGs", async ({ page }) => {
  const out: string[] = [];
  for (const path of ["/logo.png", "/logo-light.png", "/logo-mark.png", "/opengraph-image", "/apple-icon"]) {
    const r = await page.request.get(path);
    out.push(path + " " + r.status() + " " + r.headers()["content-type"] + " " + (await r.body()).length + " bytes");
    expect(r.status()).toBe(200);
    fs.writeFileSync("/private/tmp/claude-501/-Users-ethanawesome07gmail-com-Downloads-ramp-model-labs/3c14c69e-1b7e-400a-9ee1-92e9b80ae421/scratchpad/logo/" + path.replace(/[\/]/g, "").replace(/\.png$/, "") + ".png", await r.body());
  }
  fs.writeFileSync("/private/tmp/claude-501/-Users-ethanawesome07gmail-com-Downloads-ramp-model-labs/3c14c69e-1b7e-400a-9ee1-92e9b80ae421/scratchpad/logo/status.txt", out.join("\n"));
});
