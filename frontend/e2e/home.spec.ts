import { expect, test } from "@playwright/test";

test.use({ storageState: { cookies: [], origins: [] } });

test("hero renders the scene and the recorded run replays to a deployed model", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /One prompt in/ })).toBeVisible();
  await expect(page.locator("canvas")).toHaveCount(1);
  await page.getByRole("link", { name: "See a run" }).click();
  await expect(page.getByText("Deployed", { exact: true })).toBeVisible({ timeout: 30_000 });
  await expect(page.locator("#demo").getByText("winner", { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test("without WebGL the hero keeps its headline and shows the still fallback", async ({ page }) => {
  // Only WebGL is refused; 2D contexts stay available so nothing else on the page changes.
  await page.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, kind: string, ...rest: unknown[]) {
      return /webgl/i.test(kind) ? null : original.call(this, kind, ...rest);
    } as typeof HTMLCanvasElement.prototype.getContext;
  });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /One prompt in/ })).toBeVisible();
  await expect(page.getByTestId("scene-fallback")).toBeVisible();
  await expect(page.locator("canvas")).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("the example prompt holds still under reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  const example = page.getByTestId("hero-example");
  await expect(example).toBeVisible();
  const before = await example.textContent();
  expect(before).toBeTruthy();
  // The rotation period is 3.8 s, so a 5 s hold would show a change if the interval were running.
  await page.waitForTimeout(5000);
  expect(await example.textContent()).toBe(before);
});

test("the hero prompt carries into sign-up", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Describe the model you need").fill("Route support tickets to the right queue");
  await page.keyboard.press("Enter");
  await page.waitForURL(/\/signup\?prompt=/);
  await expect(page.getByText("Queued prompt")).toBeVisible();
  await expect(page.getByRole("button", { name: /Continue with Google/ })).toBeEnabled();
});

test("no horizontal overflow at phone width", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  for (const path of ["/", "/login"]) {
    await page.goto(path);
    await page.waitForTimeout(800);
    const [scrollW, innerW] = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
    expect(scrollW, path).toBeLessThanOrEqual(innerW);
  }
});

test("the app is gated", async ({ page }) => {
  await page.goto("/app/labs");
  await page.waitForURL(/\/login\?next=/);
  expect((await page.request.get("/api/auth/me")).status()).toBe(401);
});
