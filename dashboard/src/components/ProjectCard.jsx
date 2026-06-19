const asset = (url) => `${import.meta.env.BASE_URL}${url}`;

export default function ProjectCard({ project, repoUrl, onOpenFigure }) {
  const { name, trackName, kind, summary, metrics, figures, repoPath } = project;
  return (
    <article className="card">
      <div className="top">
        <h3>{name}</h3>
        {kind === "flagship" && <span className="badge flagship">★ flagship</span>}
        {kind === "seed" && <span className="badge seed">seed</span>}
      </div>
      <span className="badge track">{trackName}</span>

      {summary && <p className="summary">{summary}</p>}

      {metrics?.length > 0 && (
        <div className="chips">
          {metrics.map((m) => (
            <span className="chip" key={m.k}>
              {m.k.replace(/_/g, " ")} <b>{m.v}</b>
            </span>
          ))}
        </div>
      )}

      {figures?.length > 0 && (
        <div className="thumbs">
          {figures.map((f, i) => (
            <img
              key={f.url}
              className="thumb"
              src={asset(f.url)}
              alt={f.name}
              loading="lazy"
              onClick={() => onOpenFigure(figures, i)}
            />
          ))}
        </div>
      )}

      <div className="links">
        <a href={`${repoUrl}/tree/main/${repoPath}`} target="_blank" rel="noreferrer">
          code ↗
        </a>
        <a href={`${repoUrl}/blob/main/${repoPath}/README.md`} target="_blank" rel="noreferrer">
          readme ↗
        </a>
      </div>
    </article>
  );
}
