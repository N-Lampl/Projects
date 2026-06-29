// Client-side port of the adversarial-fraud capstone
// (06-financial-ml/CAPSTONE-adversarial-fraud).
//
// Reproduces, in plain JS:
//   * the logistic baseline      score = sigmoid(w·standardize(x) + b)
//   * the hardened gradient-boost score = sigmoid(init + lr·Σ tree(x))
//   * the greedy finite-difference evasion (attack.py) against the baseline,
//     respecting mutability / bounds / integer / consistency constraints.
//
// All weights/trees come from src/data/fraud_model.json, produced by
// dashboard/exporters/export_fraud_models.py.

import model from "../data/fraud_model.json";

export const FEATURES = model.features;
export const THRESHOLD = model.threshold;
export const SEEDS = model.seeds;
export const HEADLINE = model.headline;
const ATK = model.attack;

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

const NAMES = FEATURES.map((f) => f.name);
const MUTABLE_IDX = FEATURES.map((f, i) => (f.mutable ? i : -1)).filter((i) => i >= 0);
const INT_IDX = new Set(FEATURES.map((f, i) => (f.integer ? i : -1)).filter((i) => i >= 0));
const AMOUNT_IDX = NAMES.indexOf("amount");
const AVG_AMOUNT_IDX = NAMES.indexOf("avg_amount_30d");

function standardize(x, scaler) {
  return x.map((v, i) => (v - scaler.mean[i]) / scaler.scale[i]);
}

/** Baseline logistic P(fraud). */
export function scoreBaseline(x) {
  const { scaler, coef, intercept } = model.baseline;
  const z = standardize(x, scaler);
  let logit = intercept;
  for (let i = 0; i < z.length; i++) logit += coef[i] * z[i];
  return sigmoid(logit);
}

function evalTree(tree, z) {
  let node = 0;
  while (tree.left[node] !== -1) {
    node = z[tree.feature[node]] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
  }
  return tree.value[node];
}

/** Hardened gradient-boosting P(fraud). */
export function scoreHardened(x) {
  const { scaler, init, learningRate, trees } = model.hardened;
  const z = standardize(x, scaler);
  let raw = init;
  for (const tree of trees) raw += learningRate * evalTree(tree, z);
  return sigmoid(raw);
}

// --- greedy finite-difference evasion against the baseline (port of evade) --- //

function featureRange(i) {
  return FEATURES[i].max - FEATURES[i].min;
}

function project(x, x0) {
  const out = x.slice();
  for (let i = 0; i < out.length; i++) {
    if (!FEATURES[i].mutable) {
      out[i] = x0[i]; // immutable: revert to original
      continue;
    }
    out[i] = Math.min(Math.max(out[i], FEATURES[i].min), FEATURES[i].max);
    if (INT_IDX.has(i)) out[i] = Math.round(out[i]);
  }
  // consistency: amount stays a plausible fraction of historical spend
  const floor = Math.min(
    ATK.consistencyAmountFloor * x0[AVG_AMOUNT_IDX],
    FEATURES[AMOUNT_IDX].max,
  );
  out[AMOUNT_IDX] = Math.max(out[AMOUNT_IDX], floor);
  return out;
}

/**
 * Run the greedy evasion against the baseline, yielding one snapshot per step so
 * the UI can animate the score sliding under the threshold. Returns the full
 * trajectory: [{ x, baseline, hardened }] including the starting point.
 */
export function evadeTrajectory(x0) {
  const ranges = {};
  for (const i of MUTABLE_IDX) ranges[i] = featureRange(i);
  const stepAbs = {};
  const fdAbs = {};
  for (const i of MUTABLE_IDX) {
    stepAbs[i] = ATK.stepFrac * ranges[i];
    fdAbs[i] = ATK.fdEpsFrac * ranges[i];
  }

  let x = x0.slice();
  const traj = [{ x: x.slice(), baseline: scoreBaseline(x), hardened: scoreHardened(x) }];

  for (let step = 0; step < ATK.steps; step++) {
    if (scoreBaseline(x) < THRESHOLD) break; // evaded

    // central finite-difference gradient over mutable features
    const grad = {};
    let gnorm = 0;
    for (const i of MUTABLE_IDX) {
      const h = fdAbs[i];
      const xp = x.slice();
      const xm = x.slice();
      xp[i] += h;
      xm[i] -= h;
      const g = (scoreBaseline(xp) - scoreBaseline(xm)) / (2 * h);
      grad[i] = g;
      gnorm += g * g;
    }
    gnorm = Math.sqrt(gnorm) || 1;

    // normalized descent step (lower the score)
    for (const i of MUTABLE_IDX) {
      x[i] = x[i] - (grad[i] / gnorm) * stepAbs[i];
    }
    x = project(x, x0);
    traj.push({ x: x.slice(), baseline: scoreBaseline(x), hardened: scoreHardened(x) });
  }
  return traj;
}
