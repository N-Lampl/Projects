import { IconArrow, IconGithub } from "./icons.jsx";

export default function Hero({ data }) {
  const t = data.totals;
  return (
    <header className="hero wrap" id="overview">
      <span className="eyebrow">learning ML security</span>
      <h1>Learning to break and harden machine-learning systems.</h1>
      <p className="hero-lead">
        I'm a data scientist moving into ML security, learning it one project at a time.
        {t.projects} focused projects across detection, adversarial robustness, model privacy,
        LLM security, and the ML supply chain. Each one runs offline.
      </p>
      <div className="hero-cta">
        <a className="btn btn-primary" href="#playground">Try the live demos <IconArrow /></a>
        <a className="btn" href="#results">See the results</a>
        <a className="btn" href={data.repoUrl} target="_blank" rel="noreferrer">
          <IconGithub /> Source code
        </a>
      </div>
      <div className="hero-meta">
        <span><strong>{t.projects}</strong> projects</span>
        <span className="dot" aria-hidden="true">·</span>
        <span><strong>{t.tracks}</strong> tracks</span>
        <span className="dot" aria-hidden="true">·</span>
        <span><strong>{t.figures}</strong> result figures</span>
      </div>
    </header>
  );
}
