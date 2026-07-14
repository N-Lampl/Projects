import { IconArrow, IconGithub } from "./icons.jsx";

export default function Hero({ data }) {
  const t = data.totals;
  return (
    <header className="hero wrap" id="overview">
      <span className="eyebrow">attack &amp; defend ML</span>
      <h1>I break and harden machine-learning systems.</h1>
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
