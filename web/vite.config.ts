import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies /api to `recoup-ops serve` (default :8080), so the
// console and the API share an origin in development exactly as in production.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": process.env.RECOUP_API ?? "http://127.0.0.1:8080" },
  },
  // Observable Plot + d3 are most of the ~190 KB gzipped bundle. For a LAN
  // console read by one operator that is fine; split only if it grows.
  build: { outDir: "dist", sourcemap: true, chunkSizeWarningLimit: 700 },
});
