import { useEffect, useMemo, useRef, useState } from "react";
import {
  FEATURES,
  HEADLINE,
  SEEDS,
  THRESHOLD,
  evadeTrajectory,
  scoreBaseline,
  scoreHardened,
} from "./fraudModel.js";

const REPO_PATH =
  "https://github.com/N-Lampl/Projects/tree/main/ml-security-portfolio/06-financial-ml/CAPSTONE-adversarial-fraud";

const prefersReduced = () =>
  typeof window !== "undefined" &&
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

function sliderStep(f) {
  if (f.integer) return 1;
  if (f.max <= 1) return 0.01;
  if (f.max >= 1000) return 10;
  return 1;
}

function fmt(f, v) {
  const dp = f.name.includes("risk") ? 2 : 0;
  const n = Number(v).toFixed(dp);
  if (f.unit === "$") return `$${n}`;
  if (!f.unit || f.unit === "0/1") return n;
  return `${n} ${f.unit}`;
}

function Gauge({ name, score }) {
  const flagged = score >= THRESHOLD;
  const pct = Math.round(score * 100);
  const thrPct = Math.round(THRESHOLD * 100);
  return (
    <div>
      <div className="pg-gauge-head">
        <span className="pg-gauge-name">{name}</span>
        <span className={`pg-gauge-verdict ${flagged ? "flag" : "slip"}`}>
          {flagged ? "FRAUD · flagged" : "slips through ✓"}
        </span>
      </div>
      <div className="pg-meter" role="img" aria-label={`${name} fraud probability ${pct}%`}>
        <div className={`pg-meter-fill ${flagged ? "cool" : "hot"}`} style={{ width: `${pct}%` }} />
        <div className="pg-meter-thr" style={{ left: `${thrPct}%` }} />
      </div>
      <div className="pg-gauge-score">P(fraud) = {score.toFixed(3)}</div>
    </div>
  );
}

export default function FraudDemo() {
  const [seedIdx, setSeedIdx] = useState(0);
  const [x, setX] = useState(() => SEEDS[0].x.slice());
  const [running, setRunning] = useState(false);
  const [outcome, setOutcome] = useState(null); // null | 'held' | 'broke' | 'failed'
  const timer = useRef(null);

  useEffect(() => () => clearTimeout(timer.current), []);

  const baseline = useMemo(() => scoreBaseline(x), [x]);
  const hardened = useMemo(() => scoreHardened(x), [x]);

  const stop = () => {
    clearTimeout(timer.current);
    timer.current = null;
  };

  const loadSeed = (i) => {
    stop();
    setRunning(false);
    setOutcome(null);
    setSeedIdx(i);
    setX(SEEDS[i].x.slice());
  };

  const setFeature = (i, value) => {
    stop();
    setRunning(false);
    setOutcome(null);
    setX((prev) => {
      const next = prev.slice();
      next[i] = value;
      return next;
    });
  };

  const finish = (frame) => {
    setRunning(false);
    if (frame.baseline >= THRESHOLD) setOutcome("failed");
    else setOutcome(frame.hardened >= THRESHOLD ? "held" : "broke");
  };

  const autoEvade = () => {
    if (running) return;
    const traj = evadeTrajectory(x);
    setRunning(true);
    setOutcome(null);
    if (prefersReduced() || traj.length <= 1) {
      const last = traj[traj.length - 1];
      setX(last.x.slice());
      finish(last);
      return;
    }
    let i = 1;
    const tick = () => {
      const frame = traj[i];
      setX(frame.x.slice());
      i += 1;
      if (i >= traj.length) {
        finish(frame);
        return;
      }
      timer.current = setTimeout(tick, 55);
    };
    tick();
  };

  return (
    <div className="pg-demo">
      <div className="pg-demo-head">
        <div>
          <h3 className="pg-demo-title">Fraud-evasion sandbox</h3>
          <div className="pg-demo-tag">
            real logistic baseline + adversarially-trained gradient boosting ·{" "}
            <a href={REPO_PATH} target="_blank" rel="noreferrer">
              06-financial-ml/CAPSTONE-adversarial-fraud
            </a>
          </div>
        </div>
        <span className="pg-badge">runs in your browser · no server</span>
      </div>

      <div className="pg-demo-body">
        <div className="pg-seeds">
          <span className="pg-cap">load a flagged fraud:</span>
          {SEEDS.map((_, i) => (
            <button
              key={i}
              className={`pg-chip ${seedIdx === i ? "danger" : ""}`}
              onClick={() => loadSeed(i)}
            >
              fraud #{i + 1}
            </button>
          ))}
        </div>

        <div className="pg-fraud-grid">
          <div>
            {FEATURES.filter((f) => f.mutable).map((f) => {
              const i = FEATURES.indexOf(f);
              return (
                <div className="pg-feat" key={f.name}>
                  <div className="pg-feat-row">
                    <span className="pg-feat-label">{f.label}</span>
                    <span className="pg-feat-val">{fmt(f, x[i])}</span>
                  </div>
                  <input
                    type="range"
                    min={f.min}
                    max={f.max}
                    step={sliderStep(f)}
                    value={x[i]}
                    disabled={running}
                    onChange={(e) => setFeature(i, Number(e.target.value))}
                    aria-label={f.label}
                  />
                </div>
              );
            })}

            <p className="pg-immutable-note">
              Locked below = server-side account history the attacker can’t touch. The evasion
              may only move the transaction fields above.
            </p>
            {FEATURES.filter((f) => !f.mutable).map((f) => {
              const i = FEATURES.indexOf(f);
              return (
                <div className="pg-feat locked" key={f.name}>
                  <div className="pg-feat-row">
                    <span className="pg-feat-label">
                      {f.label} <span className="pg-lock">locked</span>
                    </span>
                    <span className="pg-feat-val locked">{fmt(f, x[i])}</span>
                  </div>
                </div>
              );
            })}
          </div>

          <div>
            <div className="pg-gauges">
              <Gauge name="Baseline model (logistic)" score={baseline} />
              <Gauge name="Hardened model (adv-trained)" score={hardened} />
            </div>

            <div className="pg-actions">
              <button className="btn btn-primary" onClick={autoEvade} disabled={running}>
                {running ? "crafting evasion…" : "▶ Auto-evade the baseline"}
              </button>
              <button className="btn" onClick={() => loadSeed(seedIdx)} disabled={running}>
                Reset
              </button>
            </div>

            <div className="pg-status">
              {outcome === "held" && (
                <>
                  <b className="broke">Baseline fooled</b>: the fraud now scores below the alert
                  line and slips through. The <b className="win">hardened model still flags it.</b>
                </>
              )}
              {outcome === "broke" && (
                <>Both models evaded on this one. Try another seed (transfer is rare: ~13%).</>
              )}
              {outcome === "failed" && (
                <>The attack couldn’t push this transaction under the line within the budget.</>
              )}
              {!outcome && !running && (
                <>Drag the transaction fields, or hit <b>Auto-evade</b> to watch the greedy attack work.</>
              )}
            </div>
          </div>
        </div>

        <details className="pg-details">
          <summary>How this works</summary>
          <div className="pg-details-body">
            <p>
              The attacker runs a feasibility-constrained greedy search: it numerically estimates{" "}
              <code>d(score)/d(feature)</code> for each mutable field and steps downhill in fraud
              probability, staying inside plausible bounds, rounding integer fields, and keeping the
              amount a realistic fraction of the account’s history.
            </p>
            <ul>
              <li><b>Mutable:</b> amount, hour, merchant risk, distance, basket size.</li>
              <li><b>Immutable:</b> account age, 30-day averages, country risk, card-present.</li>
            </ul>
            <p>
              Against the linear baseline this hits a <b>100% attack success rate</b>. Three rounds
              of adversarial training with a non-linear gradient-boosting head (which can carve out
              the bounded fraud region a single hyperplane can’t) cut it to <b>0%</b>, while clean
              PR-AUC actually <em>improved</em> from {HEADLINE.prAucBefore} to 0.65. Both models run
              here exactly as exported from Python.
            </p>
          </div>
        </details>
      </div>
    </div>
  );
}
