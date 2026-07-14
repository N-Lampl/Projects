import { useCountUp } from "../hooks.js";
import { IconArrow, IconGithub, IconTerminal } from "./icons.jsx";

function Stat({ value, label, accent }) {
  const n = useCountUp(value, true);
  return (
    <div className="stat">
      <div className={`stat-num ${accent ? "accent" : ""}`}>{n}</div>
      <div className="stat-lab">{label}</div>
    </div>
  );
}

export default function Hero({ data }) {
  const t = data.totals;
  return (
    <header className="hero wrap" id="overview">
      <div>
        <span className="eyebrow"><IconTerminal width={15} height={15} /> attack &amp; defend ML</span>
        <h1>
          I break and harden <span className="grad">machine-learning systems</span>.
        </h1>
        <p className="hero-lead">
          I'm a data scientist moving into ML security. {t.projects} focused projects covering
          detection, adversarial robustness, model privacy, LLM security, and the ML supply
          chain. Everything runs offline and maps to MITRE ATLAS / OWASP.
        </p>
        <div className="hero-cta">
          <a className="btn btn-primary" href="#playground">Try the live demos <IconArrow /></a>
          <a className="btn" href="#results">See the results</a>
          <a className="btn" href={data.repoUrl} target="_blank" rel="noreferrer">
            <IconGithub /> Source code
          </a>
        </div>
      </div>

      <div className="stats">
        <Stat value={t.projects} label="projects" />
        <Stat value={t.projects} label="passing tests" accent />
        <Stat value={t.tracks} label="tracks" />
        <Stat value={t.figures} label="result figures" />
      </div>
    </header>
  );
}
