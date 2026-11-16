import type { NextConfig } from "next";

// Provlepsi is a fully client-side app: `next build` emits a static site in `out/`
// that can be hosted anywhere (or opened from any static file server).
// When the site lives under a sub-path (GitHub Pages serves it at /<repo>/),
// set NEXT_PUBLIC_BASE_PATH, e.g. "/Provlepsi". The Pages workflow does this for you.
const basePath = (process.env.NEXT_PUBLIC_BASE_PATH ?? "").replace(/\/+$/, "");

const nextConfig: NextConfig = {
  output: "export",
  basePath: basePath || undefined,
  // Emit folder/index.html so static hosts like GitHub Pages resolve clean URLs.
  trailingSlash: true,
  images: { unoptimized: true },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
