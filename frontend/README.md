This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

## Brand assets

The logo is rendered from code (`src/lib/brand-image.tsx`), so these links always match the site. Prefix them with the
deployed host, e.g. `https://your-domain/logo.png`:

- `/logo.png` — the mark alone on dark, 1024×1024 (the logo; avatars, profile pictures, booking pages)
- `/logo-light.png` — the mark alone on light; `/logo-transparent.png` — the mark with no background, for light surfaces
- `/logo-wordmark.png` and `/logo-wordmark-light.png` — the horizontal lockup with the name, 1600×480
- `/opengraph-image` — the 1200×630 link-preview card; `/apple-icon` — the 180×180 touch icon

Link previews in Messages, Slack and social sites need `NEXT_PUBLIC_SITE_URL` set to the public URL.
