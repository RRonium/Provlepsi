import type { NextConfig } from "next";

// Provlepsi is a fully client-side app: `next build` emits a static site in `out/`
// that can be hosted anywhere (or opened from any static file server).
const nextConfig: NextConfig = {
  output: "export",
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
