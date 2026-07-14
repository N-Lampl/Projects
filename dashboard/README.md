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

## Live Playground (interactive demos)

The top "Playground" section runs two **real trained models entirely client-side** — no backend,
no API. Model weights are exported to JSON and the scoring math is reproduced in plain JavaScript
(see [`src/playground/`](src/playground/)), matching the Python originals to ~1e-6:

- **Prompt-injection detector** — a TF-IDF + LogisticRegression guard trained in the LLM-security
  track; weights committed at `src/data/injection_model.json`.
- **Fraud-evasion sandbox** — the logistic baseline + adversarially-trained gradient boosting from
  `06-financial-ml/CAPSTONE-adversarial-fraud`, with the greedy evasion ported to JS.

Re-export the model JSON only when those underlying models change:

```bash
python exporters/export_injection_detector.py
python exporters/export_fraud_models.py
node exporters/parity_check.mjs        # asserts JS↔Python parity (Δ ≤ 1e-6)
```

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
`https://n-lampl.github.io/Projects/`.

## Updating after you add/change a project

```bash
npm run data && git add public/figures src/data/projects.json && git commit -m "refresh dashboard data"
```

> Note: `npm audit` flags the dev-server-only esbuild/vite advisory. It does not affect the built
> static site (the fix is a breaking vite major); safe to ignore for this static dashboard.
