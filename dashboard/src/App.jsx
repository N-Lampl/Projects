import { useMemo, useState } from "react";
import data from "./data/projects.json";
import ProjectCard from "./components/ProjectCard.jsx";
import Lightbox from "./components/Lightbox.jsx";

export default function App() {
  const [track, setTrack] = useState("all");
  const [query, setQuery] = useState("");
  const [lb, setLb] = useState({ figures: null, index: null });

  const maxCount = Math.max(...data.tracks.map((t) => t.projectIds.length));

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return data.projects.filter((p) => {
      if (track !== "all" && p.track !== track) return false;
      if (!q) return true;
      return (
        p.name.toLowerCase().includes(q) ||
        p.summary.toLowerCase().includes(q) ||
        p.trackName.toLowerCase().includes(q)
      );
    });
  }, [track, query]);

  const openFigure = (figures, index) => setLb({ figures, index });
  const move = (d) =>
    setLb((s) => ({
      ...s,
      index: (s.index + d + s.figures.length) % s.figures.length,
    }));

  return (
    <>
      {/* hero */}
      <header className="hero">
        <div className="wrap">
          <h1>
            ML Security Portfolio — <span className="accent">attack &amp; defend</span>
          </h1>
          <p>
            A data scientist's pivot into ML security: {data.totals.projects} self-contained projects
            across {data.totals.tracks} tracks — detection engineering, adversarial robustness, model
            privacy, LLM red-teaming, and ML supply-chain — each reproducible and grounded in MITRE
            ATLAS / OWASP. Every project runs offline; results below are generated, not mocked-up.
          </p>
          <div className="meta">
            <a href={data.repoUrl} target="_blank" rel="noreferrer">{data.repoUrl.replace("https://", "")} ↗</a>
            <span>· data generated {data.generatedAt}</span>
          </div>
          <div className="stats">
            <div className="stat"><div className="num">{data.totals.projects}</div><div className="lab">projects</div></div>
            <div className="stat"><div className="num good">{data.totals.projects}</div><div className="lab">passing tests</div></div>
            <div className="stat"><div className="num">{data.totals.tracks}</div><div className="lab">tracks</div></div>
            <div className="stat"><div className="num">{data.totals.figures}</div><div className="lab">result figures</div></div>
          </div>
        </div>
      </header>

      <main className="wrap">
        {/* highlights */}
        <section>
          <h2 className="section-title">Headline results</h2>
          <p className="section-sub">Real before/after numbers pulled from each project's metrics.json.</p>
          <div className="highlights">
            {data.highlights.map((h) => (
              <div className="hl" key={h.title}>
                <h3>{h.title}</h3>
                <div className="ba">
                  <span className="pill before">{h.before}</span>
                  <span className="arrow">→</span>
                  <span className="pill after">{h.after}</span>
                  <span className="lab">{h.label}</span>
                </div>
                <p>{h.blurb}</p>
              </div>
            ))}
          </div>
        </section>

        {/* per-track counts */}
        <section>
          <h2 className="section-title">Coverage by track</h2>
          <p className="section-sub">Balanced across the field — not just the flashy attacks.</p>
          <div className="trackbars">
            {data.tracks.map((t) => (
              <div className="trackbar" key={t.id}>
                <span className="name">{t.name}</span>
                <span className="bar" style={{ width: `${(t.projectIds.length / maxCount) * 100}%` }} />
                <span>{t.projectIds.length}</span>
              </div>
            ))}
          </div>
        </section>

        {/* projects */}
        <section>
          <h2 className="section-title">All projects</h2>
          <p className="section-sub">Filter by track or search. Click a chart to enlarge.</p>
          <div className="controls">
            <button className={`pill-btn ${track === "all" ? "active" : ""}`} onClick={() => setTrack("all")}>
              All ({data.projects.length})
            </button>
            {data.tracks.map((t) => (
              <button
                key={t.id}
                className={`pill-btn ${track === t.id ? "active" : ""}`}
                onClick={() => setTrack(t.id)}
              >
                {t.name} ({t.projectIds.length})
              </button>
            ))}
            <input
              className="search"
              placeholder="search projects…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>

          {visible.length === 0 ? (
            <p className="empty">No projects match “{query}”.</p>
          ) : (
            <div className="grid">
              {visible.map((p) => (
                <ProjectCard key={p.id} project={p} repoUrl={data.repoUrl} onOpenFigure={openFigure} />
              ))}
            </div>
          )}
        </section>

        {/* roadmap */}
        <section>
          <h2 className="section-title">How to build on this</h2>
          <p className="section-sub">Each project ships an offline path; here's how to take it further.</p>
          <div className="roadmap">
            {data.roadmap.map((r) => (
              <div className="rm" key={r.title}>
                <h4>{r.title}</h4>
                <p>{r.detail}</p>
              </div>
            ))}
          </div>
        </section>
      </main>

      <footer>
        <div className="wrap">
          Built as a learning + portfolio repo. Dual-use techniques, authorized-use only — see the
          repo's ETHICS.md. · <a href={data.repoUrl} target="_blank" rel="noreferrer">source ↗</a>
        </div>
      </footer>

      <Lightbox
        figures={lb.figures}
        index={lb.index}
        onClose={() => setLb({ figures: null, index: null })}
        onMove={move}
      />
    </>
  );
}
