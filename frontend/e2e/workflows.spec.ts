import { expect, test } from "@playwright/test";
import { vendorsCsv } from "./helpers";

test.describe.configure({ mode: "serial" });

test("a spreadsheet runs through a model, uncertain rows go to review, and results export", async ({ page }) => {
  const deployments = await (await page.request.get("/api/deployments")).json();
  test.skip(deployments.length === 0, "needs a deployed model from app.spec.ts");
  await page.goto("/app/workflows");
  await expect(page.getByRole("heading", { name: "Workflows" })).toBeVisible();
  const rows = ["text", ...Array.from({ length: 15 }, (_, i) => `"card-not-present $${(i * 37) % 900 + 20}.00 at ${["GIFTCARDMALL", "STAPLES", "LUXE WATCHES", "WHOLE FOODS"][i % 4]} 0${i % 9}:1${i % 6} local; ${i % 3 ? "recurring vendor" : "new device fingerprint"}"`)];
  await page.setInputFiles('input[type="file"]', { name: "october.csv", mimeType: "text/csv", buffer: Buffer.from(rows.join("\n")) });
  await page.getByPlaceholder(/What should happen to these rows/).fill("flag the risky transactions");
  await page.getByRole("button", { name: "Run workflow" }).click();
  await page.waitForURL(/\/app\/workflows\/wf_/);
  await expect(page.getByRole("heading", { name: "october.csv" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Excel" })).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText("Decided automatically")).toBeVisible();
  const list = page.locator("li").filter({ hasText: /card-not-present/ });
  await expect(list.first()).toBeVisible();
  expect(await list.count()).toBe(15);
  const xlsx = await page.request.get(page.url().replace(/\/app\/workflows\//, "/api/workflows/") + "/export?format=xlsx");
  expect(xlsx.status()).toBe(200);
  expect(xlsx.headers()["content-type"]).toContain("spreadsheetml");
  const csv = await page.request.get(page.url().replace(/\/app\/workflows\//, "/api/workflows/") + "/export?format=csv");
  expect((await csv.text()).split("\n")[0]).toContain("prediction,confidence,needs_review,runner_up");
});

test("a labeled upload with a matching prompt reuses the model and reports accuracy on the file", async ({ page }) => {
  await page.goto("/app/workflows");
  await page.setInputFiles('input[type="file"]', { name: "vendors.csv", mimeType: "text/csv", buffer: vendorsCsv() });
  await page.getByPlaceholder(/What should happen to these rows/).fill("classify card merchants into software, travel, or meals");
  await page.getByRole("button", { name: "Run workflow" }).click();
  await page.waitForURL(/\/app\/workflows\/wf_/);
  await expect(page.getByRole("link", { name: "Excel" })).toBeVisible({ timeout: 300_000 });
  await expect(page.getByText("Accuracy on your labels")).toBeVisible();
  await page.getByRole("button", { name: "Needs review" }).click();
  await expect(page.getByText("Nothing needs a person.").or(page.locator("select").first())).toBeVisible();
});
