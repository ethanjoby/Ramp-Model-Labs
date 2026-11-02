import { expect, type Page } from "@playwright/test";

/** Signs in through the real button; the Auth emulator stands in for Google's account chooser. */
export async function signInWithGoogle(page: Page) {
  await page.goto("/login");
  const popupPromise = page.waitForEvent("popup");
  await page.getByRole("button", { name: /Continue with Google/ }).click();
  const popup = await popupPromise;
  await popup.waitForLoadState();
  const existing = popup.getByText(/@example\.com/).first();
  if (await existing.count()) {
    await existing.click();
  } else {
    await popup.getByText(/Add new account/i).first().click();
    await popup.getByRole("button", { name: /Auto-generate/i }).click();
    await popup.getByRole("button", { name: /Sign in with Google\.com/i }).click();
  }
  await page.waitForURL(/\/app/);
  await expect(page.getByRole("link", { name: "Labs", exact: true })).toBeVisible();
}

export async function waitForLab(page: Page, labId: string, timeoutMs = 150_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const lab = await (await page.request.get(`/api/labs/${labId}`)).json();
    if (lab.status === "completed" || lab.status === "failed") return lab;
    await page.waitForTimeout(1000);
  }
  throw new Error(`lab ${labId} did not finish`);
}

export function vendorsCsv() {
  const groups: Record<string, string[]> = {
    software: ["GITHUB.COM", "FIGMA MONTHLY", "ATLASSIAN", "NOTION LABS", "ZOOM.US", "SLACK TECHNOLOGIES", "AWS EMEA", "DATADOG", "VERCEL INC", "LINEAR APP"],
    travel: ["UNITED AIRLINES", "DELTA AIR LINES", "MARRIOTT SFO", "HILTON NYC", "UBER TRIP", "LYFT RIDE", "AMTRAK", "HERTZ RENT A CAR", "AIRBNB", "SOUTHWEST AIR"],
    meals: ["SWEETGREEN", "CHIPOTLE", "BLUE BOTTLE COFFEE", "DOORDASH", "STARBUCKS", "SHAKE SHACK", "PANERA BREAD", "UBER EATS", "WHOLE FOODS", "TARTINE BAKERY"],
  };
  const rows: string[] = [];
  for (const [label, merchants] of Object.entries(groups)) {
    for (let i = 0; i < 20; i++) rows.push(`"${merchants[i % 10]} #${1000 + i * 37} $${(((i * 7919) % 400) + 5).toFixed(2)}",${label}`);
  }
  return Buffer.from("text,label\n" + rows.join("\n"));
}
