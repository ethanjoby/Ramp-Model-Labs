import type { Metadata, Viewport } from "next";
import { Geist, IBM_Plex_Mono } from "next/font/google";
import { Providers } from "@/components/ui/Providers";
import "./globals.css";

const sans = Geist({ variable: "--font-sans-face", subsets: ["latin"] });
const mono = IBM_Plex_Mono({ variable: "--font-mono-face", subsets: ["latin"], weight: ["400", "500"] });

const TITLE = "Ascent Labs";
const DESCRIPTION = "Bring us the problem. We build the AI system that solves it on the best open model for the job, and you keep the weights.";

// Social cards need absolute URLs; the deployment sets the host, local dev falls back to the dev server.
function siteUrl(): URL {
  try {
    return new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000");
  } catch {
    return new URL("http://localhost:3000");
  }
}

// opengraph-image.tsx, twitter-image.tsx and apple-icon.tsx next to this file render the card and icon that
// Messages, Slack and social sites show for a link; Next wires them in here. They need metadataBase to be the
// public URL (NEXT_PUBLIC_SITE_URL), because previews only load absolute image URLs.
export const metadata: Metadata = {
  metadataBase: siteUrl(),
  title: { default: TITLE, template: `%s · ${TITLE}` },
  description: DESCRIPTION,
  applicationName: TITLE,
  openGraph: { type: "website", siteName: TITLE, title: TITLE, description: DESCRIPTION, url: "/", locale: "en_US" },
  twitter: { card: "summary_large_image", title: TITLE, description: DESCRIPTION },
  appleWebApp: { title: TITLE },
};

export const viewport: Viewport = { themeColor: "#0c0a08" };

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} h-full`}>
      <body className="min-h-full">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
