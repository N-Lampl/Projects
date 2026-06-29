// Parity check: confirm the exported JSON reproduces the Python model scores.
// Independent re-implementation of the scoring math (mirrors the logic in
// src/playground/{injectionModel,fraudModel}.js) compared against the `sanity`
// / `seeds` values the Python exporters embedded.
//
// Run:  node dashboard/exporters/parity_check.mjs
// Exits non-zero if any score differs from Python by more than TOL.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const dataDir = resolve(here, "../src/data");
const load = (f) => JSON.parse(readFileSync(resolve(dataDir, f), "utf8"));

const TOL = 1e-6;
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
let failures = 0;

function check(label, got, want) {
  const d = Math.abs(got - want);
  const ok = d <= TOL;
  if (!ok) failures++;
  console.log(`  ${ok ? "OK " : "FAIL"}  got=${got.toFixed(8)} want=${want.toFixed(8)} Δ=${d.toExponential(2)}  ${label}`);
}

// ----------------------------- injection ----------------------------- //
function injectionScore(m, text) {
  const tokens = text.toLowerCase().match(/[a-z0-9_]{2,}/g) || [];
  const counts = new Map();
  for (let i = 0; i < tokens.length; i++) {
    const add = (t) => counts.set(t, (counts.get(t) || 0) + 1);
    add(tokens[i]);
    if (i + 1 < tokens.length) add(`${tokens[i]} ${tokens[i + 1]}`);
  }
  let sumSq = 0;
  const ws = [];
  for (const [term, count] of counts) {
    const idx = m.vocabulary[term];
    if (idx === undefined) continue;
    const w = (1 + Math.log(count)) * m.idf[idx];
    ws.push([idx, w]);
    sumSq += w * w;
  }
  const norm = Math.sqrt(sumSq) || 1;
  let logit = m.intercept;
  for (const [idx, w] of ws) logit += (w / norm) * m.coef[idx];
  return sigmoid(logit);
}

console.log("[injection] TF-IDF + LogReg parity:");
const inj = load("injection_model.json");
for (const s of inj.sanity) check(s.text.slice(0, 56), injectionScore(inj, s.text), s.proba);

// ------------------------------- fraud ------------------------------- //
const fraud = load("fraud_model.json");
const std = (x, sc) => x.map((v, i) => (v - sc.mean[i]) / sc.scale[i]);

function baseline(m, x) {
  const z = std(x, m.baseline.scaler);
  let logit = m.baseline.intercept;
  for (let i = 0; i < z.length; i++) logit += m.baseline.coef[i] * z[i];
  return sigmoid(logit);
}
function evalTree(t, z) {
  let n = 0;
  while (t.left[n] !== -1) n = z[t.feature[n]] <= t.threshold[n] ? t.left[n] : t.right[n];
  return t.value[n];
}
function hardened(m, x) {
  const z = std(x, m.hardened.scaler);
  let raw = m.hardened.init;
  for (const t of m.hardened.trees) raw += m.hardened.learningRate * evalTree(t, z);
  return sigmoid(raw);
}

console.log("[fraud] baseline (logreg) parity:");
for (const s of fraud.seeds) check("seed", baseline(fraud, s.x), s.baselineScore);
console.log("[fraud] hardened (gboost) parity:");
for (const s of fraud.seeds) check("seed", hardened(fraud, s.x), s.hardenedScore);

console.log(failures ? `\nFAILED: ${failures} mismatch(es)` : "\nAll parity checks passed.");
process.exit(failures ? 1 : 0);
