// Client-side port of the real prompt-injection detector
// (04-llm-security/p7-defend-rag: TF-IDF + LogisticRegression).
//
// Reproduces sklearn's scoring path exactly so the browser verdict matches the
// Python model bit-for-bit:
//   lowercase -> tokenize (\b\w\w+\b) -> unigrams+bigrams -> raw counts
//   -> sublinear tf (1+log(count)) -> * idf -> L2-normalize -> dot(coef)+intercept
//   -> sigmoid -> compare to threshold.
//
// Weights are loaded from src/data/injection_model.json, produced by
// dashboard/exporters/export_injection_detector.py.

import model from "../data/injection_model.json";

const { vocabulary, idf, coef, intercept, threshold } = model;

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// sklearn default token_pattern is \b\w\w+\b; on lowercased ASCII text this is
// equivalent to runs of 2+ word characters.
function tokenize(text) {
  return text.toLowerCase().match(/[a-z0-9_]{2,}/g) || [];
}

// Unigrams + bigrams, matching TfidfVectorizer(ngram_range=(1, 2)).
function buildTerms(tokens) {
  const terms = [];
  for (let i = 0; i < tokens.length; i++) {
    terms.push({ term: tokens[i], pos: [i] });
    if (i + 1 < tokens.length) {
      terms.push({ term: `${tokens[i]} ${tokens[i + 1]}`, pos: [i, i + 1] });
    }
  }
  return terms;
}

/**
 * Score a string. Returns the injection probability, the BLOCK/ALLOW verdict,
 * and per-token contributions (positive = pushes toward "injection") so the UI
 * can highlight the trigger words.
 */
export function scoreText(text) {
  const tokens = tokenize(text);
  const terms = buildTerms(tokens);

  // raw term counts + which positions each term covers
  const counts = new Map();
  const positions = new Map();
  for (const { term, pos } of terms) {
    counts.set(term, (counts.get(term) || 0) + 1);
    if (!positions.has(term)) positions.set(term, pos);
  }

  // tf-idf weights for in-vocabulary terms, then L2 norm
  const weights = []; // { idx, w, term, count }
  let sumSq = 0;
  for (const [term, count] of counts) {
    const idx = vocabulary[term];
    if (idx === undefined) continue;
    const tf = 1 + Math.log(count); // sublinear_tf
    const w = tf * idf[idx];
    weights.push({ idx, w, term, count });
    sumSq += w * w;
  }
  const norm = Math.sqrt(sumSq) || 1;

  // logit + per-term contribution to the score
  let logit = intercept;
  const tokenContrib = new Array(tokens.length).fill(0);
  for (const { idx, w, term, count } of weights) {
    const nw = w / norm;
    const contribution = nw * coef[idx];
    logit += contribution;
    // attribute the term's contribution across the token(s) it covers
    const pos = positions.get(term);
    const share = contribution / count / pos.length;
    for (const p of pos) tokenContrib[p] += share;
  }

  return {
    proba: sigmoid(logit),
    blocked: sigmoid(logit) >= threshold,
    threshold,
    tokens: tokens.map((t, i) => ({ text: t, contribution: tokenContrib[i] })),
  };
}

export const injectionThreshold = threshold;
export const injectionSanity = model.sanity || [];
