import { test as setup } from "@playwright/test";
import { signInWithGoogle } from "./helpers";

setup("sign in once and keep the session", async ({ page }) => {
  await signInWithGoogle(page);
  await page.context().storageState({ path: "e2e/.auth/user.json" });
});
