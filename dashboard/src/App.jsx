import { useMemo, useState } from "react";
import data from "./data/projects.json";
import { useInView, useTheme } from "./hooks.js";
import Nav from "./components/Nav.jsx";
import Hero from "./components/Hero.jsx";
import Highlights from "./components/Highlights.jsx";
import TrackChart from "./components/TrackChart.jsx";
import ProjectCard from "./components/ProjectCard.jsx";
import ProjectModal from "./components/ProjectModal.jsx";
import Lightbox from "./components/Lightbox.jsx";
import Roadmap from "./components/Roadmap.jsx";
import { IconSearch, IconShield } from "./components/icons.jsx";

// Framework tags per track (client-side; keeps the data file lean).
const TRACK_TAGS = {
  "00-foundations": ["MITRE ATT&CK", "MITRE ATLAS"],
  "01-detection-engineering": ["MITRE ATT&CK", "detection-as-code"],
  "02-adversarial-robustness": ["MITRE ATLAS", "evasion"],
  "03-ml-privacy": ["MITRE ATLAS", "differential privacy"],
  "04-llm-security": ["OWASP LLM Top 10"],
  "05-ml-supply-chain": ["MITRE ATLAS", "MLSecOps"],
};
const tagsFor = (p) => TRACK_TAGS[p.track] || [];

function Reveal({ children }) {
  const [ref, inView] = useInView();
  return <div ref={ref} className={`reveal ${inView ? "in" : ""}`}>{children}</div>;
}

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const [track, setTrack] = useState("all");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(null);
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

  const move = (d) =>
    setLb((s) => ({ ...s, index: (s.index + d + s.figures.length) % s.figures.length }));

  return (
    <>
      <Nav theme={theme} onToggleTheme={toggleTheme} repoUrl={data.repoUrl} />
      <Hero data={data} />

      <main className="wrap">
        <section className="section" id="results">
          <Reveal>
            <div className="section-head">
              <span className="kicker">// headline results</span>
              <h2 className="section-title">What the attacks &amp; defenses prove</h2>
              <p className="section-sub">Real before/after numbers pulled straight from each project's metrics.json.</p>
            </div>
            <Highlights highlights={data.highlights} />
          </Reveal>
        </section>

        <section className="section" id="coverage">
          <Reveal>
            <div className="section-head">
              <span className="kicker">// breadth</span>
              <h2 className="section-title">Coverage across the field</h2>
              <p className="section-sub">Balanced from foundations to flagship capstones — not just the flashy attacks.</p>
            </div>
            <TrackChart tracks={data.tracks} max={maxCount} />
          </Reveal>
        </section>

        <section className="section" id="projects">
          <Reveal>
            <div className="section-head">
              <span className="kicker">// {data.projects.length} projects</span>
              <h2 className="section-title">Every project</h2>
              <p className="section-sub">Filter by track or search. Click any card for full metrics and figures.</p>
            </div>
            <div className="controls">
              <div className="filters">
                <button className={`chip-btn ${track === "all" ? "active" : ""}`} onClick={() => setTrack("all")}>
                  all · {data.projects.length}
                </button>
                {data.tracks.map((t) => (
                  <button
                    key={t.id}
                    className={`chip-btn ${track === t.id ? "active" : ""}`}
                    onClick={() => setTrack(t.id)}
                  >
                    {t.name} · {t.projectIds.length}
                  </button>
                ))}
              </div>
              <label className="search-wrap">
                <IconSearch width={15} height={15} />
                <input
                  className="search"
                  placeholder="search projects…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  aria-label="Search projects"
                />
              </label>
            </div>

            {visible.length === 0 ? (
              <p className="empty">No projects match “{query}”.</p>
            ) : (
              <div className="grid">
                {visible.map((p) => (
                  <ProjectCard key={p.id} project={p} tags={tagsFor(p)} repoUrl={data.repoUrl} onOpen={setSelected} />
                ))}
              </div>
            )}
          </Reveal>
        </section>

        <section className="section" id="roadmap">
          <Reveal>
            <div className="section-head">
              <span className="kicker">// what's next</span>
              <h2 className="section-title">How to build on this</h2>
              <p className="section-sub">Each project ships an offline path; here's how it scales up to real data and models.</p>
            </div>
            <Roadmap items={data.roadmap} />
          </Reveal>
        </section>
      </main>

      <footer className="footer">
        <div className="wrap row">
          <span className="warn"><IconShield width={15} height={15} /> Dual-use techniques · authorized-use only (see ETHICS.md)</span>
          <span>
            <a href={data.repoUrl} target="_blank" rel="noreferrer">github.com/N-Lampl/Cyber-Projects</a>
          </span>
        </div>
      </footer>

      <ProjectModal
        project={selected}
        tags={selected ? tagsFor(selected) : []}
        repoUrl={data.repoUrl}
        onClose={() => setSelected(null)}
        onOpenFig={(figures, index) => setLb({ figures, index })}
      />
      <Lightbox
        figures={lb.figures}
        index={lb.index}
        onClose={() => setLb({ figures: null, index: null })}
        onMove={move}
      />
    </>
  );
}
