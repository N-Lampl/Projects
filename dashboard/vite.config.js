import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" -> relative asset URLs, so the built site works locally AND under any
// GitHub Pages subpath (e.g. /Projects/) without further config.
export default defineConfig({
  base: "./",
  plugins: [react()],
});
