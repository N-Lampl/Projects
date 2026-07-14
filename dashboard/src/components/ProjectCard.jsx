import { IconArrow, IconExternal, IconStar } from "./icons.jsx";

const asset = (u) => `${import.meta.env.BASE_URL}${u}`;

export default function ProjectCard({ project, repoUrl, onOpen }) {
  const { name, trackName, kind, summary, figures, repoPath } = project;
  const open = () => onOpen(project);
  return (
    <div
      className="card"
      role="button"
      tabIndex={0}
      onClick={open}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), open())}
      aria-label={`Open details for ${name}`}
    >
      <div className="card-head">
        <div>
          <h3 className="card-title">{name}</h3>
          <div className="card-track">{trackName}</div>
        </div>
        {kind === "flagship" && <span className="tag flag"><IconStar width={12} height={12} /> flagship</span>}
        {kind === "seed" && <span className="tag seed">seed</span>}
      </div>

      {summary && <p className="card-sum">{summary}</p>}

      {figures?.length > 0 && (
        <div className="card-figs">
          {figures.slice(0, 2).map((f) => (
            <img key={f.url} className="fig-thumb" src={asset(f.url)} alt={f.name} loading="lazy" />
          ))}
        </div>
      )}

      <div className="card-foot">
        <span className="card-cta">View details <IconArrow width={14} height={14} /></span>
        <a
          className="card-link"
          href={`${repoUrl}/tree/main/${repoPath}`}
          target="_blank"
          rel="noreferrer"
          onClick={(e) => e.stopPropagation()}
        >
          code <IconExternal width={13} height={13} />
        </a>
      </div>
    </div>
  );
}
