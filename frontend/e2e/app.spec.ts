import { expect, test } from "@playwright/test";
import { vendorsCsv, waitForLab } from "./helpers";

test.describe.configure({ mode: "serial" });

test("a starter prompt builds, scores and deploys a model, then the playground learns from a correction", async ({ page }) => {
  await page.goto("/app");
  await expect(page.getByRole("heading", { name: /What should we automate/ })).toBeVisible();
  await page.getByRole("button", { name: /Screen transactions for fraud/ }).click();
  const card = page.locator('a[href^="/app/labs/lab_"]').first();
  await expect(card).toBeVisible({ timeout: 30_000 });
  const labId = (await card.getAttribute("href"))!.split("/").pop()!;

  await page.goto(`/app/labs/${labId}`);
  await expect(page.getByText("Agent activity")).toBeVisible();
  await expect(page.getByText("Selected model")).toBeVisible({ timeout: 150_000 });
  const caps = await (await page.request.get("/api/capabilities")).json();
  await expect(page.getByTestId("leaderboard").locator("tbody tr")).toHaveCount(caps.strategies.length);
  await expect(page.getByText("winner", { exact: true })).toBeVisible();

  const lab = await waitForLab(page, labId);
  expect(lab.status).toBe("completed");
  const available = caps.strategies.filter((s: { available: boolean }) => s.available).length;
  expect(lab.candidates.filter((c: { status: string }) => c.status === "done")).toHaveLength(available);
  expect(lab.candidates.filter((c: { status: string }) => c.status === "failed")).toHaveLength(0);

  await page.getByRole("button", { name: "Open playground" }).click();
  await page.waitForURL(/\/app\/models\//);
  await page.getByPlaceholder("One input per line").fill("card-not-present $2,499.00 at GIFTCARDMALL 03:12; new device; 4 declines\ncontactless $14.20 at BLUE BOTTLE 08:10; recurring vendor");
  await page.getByRole("button", { name: "Run model" }).click();
  await expect(page.getByText(/2 predictions in/)).toBeVisible();
  await expect(page.getByText("0 corrections saved")).toBeVisible();
  await page.locator("select").first().selectOption({ index: 1 });
  await expect(page.getByText("1 correction saved")).toBeVisible();
  await expect(page.getByRole("button", { name: "Retrain" })).toBeEnabled();
});

test("the GL-code starter builds a lab too (the offline copilot used to read it as a prediction request)", async ({ page }) => {
  await page.goto("/app");
  await page.getByRole("button", { name: "Clear chat" }).click();
  await page.getByRole("button", { name: /Categorize card transactions into GL codes/ }).click();
  await expect(page.getByText(/^On it\./)).toBeVisible({ timeout: 20_000 });
  await expect(page.locator('a[href^="/app/labs/lab_"]').first()).toBeVisible({ timeout: 30_000 });
});

test("uploaded labeled examples drive a custom task evaluated on real rows", async ({ page }) => {
  await page.goto("/app");
  await expect(page.getByRole("button", { name: "Clear chat" })).toBeVisible();  // history has loaded; count after, not before
  const before = await page.locator('a[href^="/app/labs/lab_"]').count();
  await page.setInputFiles('input[type="file"]', { name: "vendors.csv", mimeType: "text/csv", buffer: vendorsCsv() });
  await expect(page.getByText(/vendors\.csv · 60 examples, 60 labeled/)).toBeVisible();
  await page.locator("textarea").fill("Build a model that classifies card merchants into software, travel, or meals");
  await page.keyboard.press("Enter");
  await expect(page.locator('a[href^="/app/labs/lab_"]')).toHaveCount(before + 1, { timeout: 30_000 });
  const labId = (await page.locator('a[href^="/app/labs/lab_"]').last().getAttribute("href"))!.split("/").pop()!;
  const lab = await waitForLab(page, labId);
  expect(lab.status).toBe("completed");
  expect(lab.spec.labels).toEqual(["meals", "software", "travel"]);
  expect(lab.dataset.eval_source).toBe("your real examples");
  expect(lab.dataset.sources.user).toBe(60);
  expect(lab.result.winner_metrics.accuracy).toBeGreaterThan(0.5);
});

test("API keys are shown once, work on /v1, and die when revoked", async ({ page }) => {
  await page.goto("/app/keys");
  await page.getByLabel("Key name").fill("e2e");
  await page.getByRole("button", { name: "Create key" }).click();
  const key = await page.locator("code").first().innerText();
  expect(key).toMatch(/^rml_live_/);
  const slug = (await (await page.request.get("/api/deployments")).json())[0].slug;
  const ok = await page.request.post(`/v1/models/${slug}/predict`, { data: { inputs: ["UBER *TRIP"] }, headers: { Authorization: `Bearer ${key}` } });
  expect(ok.status()).toBe(200);
  page.once("dialog", (d) => d.accept());
  await page.getByRole("button", { name: "Revoke" }).first().click();
  await expect(page.getByText("No active keys.")).toBeVisible();
  const dead = await page.request.post(`/v1/models/${slug}/predict`, { data: { inputs: ["UBER *TRIP"] }, headers: { Authorization: `Bearer ${key}` } });
  expect(dead.status()).toBe(401);
});

test("settings shows the Google account and sign-out revokes the session", async ({ page }) => {
  await page.goto("/app/settings");
  await expect(page.getByText("Signed in with Google")).toBeVisible();
  await page.getByLabel("Sign out").click();
  await page.waitForURL(/\/login/);
  expect((await page.request.get("/api/auth/me")).status()).toBe(401);
});
