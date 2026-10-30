import { getApps, initializeApp } from "firebase/app";
import {
  GoogleAuthProvider, connectAuthEmulator, getAuth, getRedirectResult, signInWithPopup, signInWithRedirect, type Auth,
} from "firebase/auth";

const cfg = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};

export const firebaseConfigured = Boolean(cfg.apiKey && cfg.projectId && cfg.appId);

let auth: Auth | null = null;

function firebaseAuth(): Auth {
  if (auth) return auth;
  auth = getAuth(getApps()[0] ?? initializeApp(cfg));
  const emulator = process.env.NEXT_PUBLIC_FIREBASE_AUTH_EMULATOR;
  if (emulator) connectAuthEmulator(auth, emulator, { disableWarnings: true });
  return auth;
}

function provider() {
  const p = new GoogleAuthProvider();
  p.setCustomParameters({ prompt: "select_account" });
  return p;
}

/** Opens the Google account chooser and resolves with a Firebase ID token for the backend to verify. */
export async function signInWithGoogle(): Promise<string> {
  const a = firebaseAuth();
  try {
    const cred = await signInWithPopup(a, provider());
    return cred.user.getIdToken();
  } catch (e) {
    const code = (e as { code?: string }).code ?? "";
    if (code === "auth/popup-blocked" || code === "auth/operation-not-supported-in-this-environment") {
      await signInWithRedirect(a, provider());
      return new Promise<string>(() => {}); // the page is navigating away
    }
    if (code === "auth/popup-closed-by-user" || code === "auth/cancelled-popup-request") throw new Error("Sign-in was closed before finishing.");
    if (code === "auth/unauthorized-domain") throw new Error("This domain isn't authorized for Google sign-in yet.");
    if (code === "auth/operation-not-allowed") throw new Error("Google sign-in isn't enabled on the Firebase project yet.");
    if (code === "auth/configuration-not-found") throw new Error("Firebase Authentication isn't set up on this project yet.");
    throw new Error("Google sign-in didn't go through. Try again.");
  }
}

/** After a redirect-based sign-in, returns the ID token once; null when there is nothing pending. */
export async function pendingRedirectToken(): Promise<string | null> {
  if (!firebaseConfigured) return null;
  const r = await getRedirectResult(firebaseAuth()).catch(() => null);
  return r ? r.user.getIdToken() : null;
}

export async function signOutOfFirebase() {
  if (auth) await auth.signOut().catch(() => {});
}
