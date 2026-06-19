# Results dashboard (React + Vite)

An interactive showcase of the whole portfolio: headline before/after results, per-track coverage,
every project with its metrics + figures, and a "how to build on this" roadmap. It reads each project's
`results/metrics.json` + figures — so as you extend the repo, re-run the generator and the dashboard
updates itself.

## Run it locally

```bash
cd dashboard
npm install
npm run data      # aggregate ../**/results into src/data/projects.json + public/figures (Python 3)
npm run dev       # http://localhost:5173
```

`npm run build` produces a static site in `dist/`; `npm run preview` serves it.

## How it works

- [`generate_data.py`](generate_data.py) walks the six tracks, reads each project's
  `results/metrics.json` and README title/tagline, copies its figures into `public/figures/`, and
  writes [`src/data/projects.json`](src/data/projects.json) (committed, so the app works on a fresh
  clone without Python). Headline numbers are pulled from real metrics, not hardcoded.
- The React app ([`src/App.jsx`](src/App.jsx)) renders highlights, a track-coverage chart, a
  filterable/searchable project grid with a figure lightbox, and the roadmap. No chart library —
  lightweight CSS/SVG to keep the build robust.
- `vite.config.js` uses `base: "./"` so the build works locally **and** under any GitHub Pages subpath.

## Deploy (GitHub Pages)

A workflow at [`.github/workflows/deploy-dashboard.yml`](../.github/workflows/deploy-dashboard.yml)
builds and publishes on every push to `dashboard/`. **One-time setup:** repo **Settings → Pages →
Build and deployment → Source: "GitHub Actions"**. It then deploys to
`https://n-lampl.github.io/Cyber-Projects/`.

## Updating after you add/change a project

```bash
npm run data && git add public/figures src/data/projects.json && git commit -m "refresh dashboard data"
```

> Note: `npm audit` flags the dev-server-only esbuild/vite advisory. It does not affect the built
> static site (the fix is a breaking vite major); safe to ignore for this static dashboard.
