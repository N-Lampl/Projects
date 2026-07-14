import { IconGithub, IconMoon, IconSun } from "./icons.jsx";

const LINKS = [
  ["#playground", "Playground"],
  ["#results", "Results"],
  ["#projects", "Projects"],
];

export default function Nav({ theme, onToggleTheme, repoUrl }) {
  return (
    <nav className="nav">
      <div className="wrap nav-inner">
        <a className="brand" href="#overview" aria-label="Home">
          <span className="brand-mark">NL</span>
          <span>
            <span className="brand-name">Nick&nbsp;Lampl</span>
            <br />
            <span className="brand-role">ML / AI Security</span>
          </span>
        </a>
        <div className="nav-links">
          {LINKS.map(([href, label]) => (
            <a key={href} className="nav-link" href={href}>{label}</a>
          ))}
        </div>
        <div className="nav-right">
          <button
            className="icon-btn"
            onClick={onToggleTheme}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? <IconSun /> : <IconMoon />}
          </button>
          <a className="btn" href={repoUrl} target="_blank" rel="noreferrer">
            <IconGithub /> GitHub
          </a>
        </div>
      </div>
    </nav>
  );
}
