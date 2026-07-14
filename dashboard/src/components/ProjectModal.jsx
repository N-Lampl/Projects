import { useEffect } from "react";
import { useBodyLock } from "../hooks.js";
import { IconClose, IconExternal } from "./icons.jsx";

const asset = (u) => `${import.meta.env.BASE_URL}${u}`;
const SKIP = new Set(["project", "summary", "figures", "note"]);
const PCT_HINT = /rate|acc|auc|asr|pct|cosine|retention|fraction|reduction|delta/;

function scalars(raw) {
  const out = [];
  for (const [k, v] of Object.entries(raw || {})) {
    if (SKIP.has(k) || v == null || typeof v === "object") continue;
    let val;
    if (typeof v === "boolean") val = v ? "yes" : "no";
    else if (typeof v === "number")
      val = PCT_HINT.test(k) && Math.abs(v) <= 1.0001 ? `${(v * 100).toFixed(0)}%` : `${v}`;
    else val = String(v).slice(0, 60);
    out.push({ k: k.replace(/_/g, " "), v: val });
  }
  return out;
}

export default function ProjectModal({ project, repoUrl, onClose, onOpenFig }) {
  useBodyLock(!!project);
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!project) return null;
  const rows = scalars(project.raw);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={project.name} onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <h3>{project.name}</h3>
            <div className="card-track">{project.trackName}</div>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close details"><IconClose /></button>
        </div>
        <div className="modal-body">
          {(project.kind === "flagship" || project.kind === "seed") && (
            <div className="tags">
              {project.kind === "flagship" && <span className="tag flag">flagship</span>}
              {project.kind === "seed" && <span className="tag seed">seed</span>}
            </div>
          )}

          {project.summary && <p className="lead">{project.summary}</p>}

          {rows.length > 0 && (
            <div className="kv">
              {rows.map((r) => (
                <div className="kv-row" key={r.k}>
                  <div className="k">{r.k}</div>
                  <div className="v">{r.v}</div>
                </div>
              ))}
            </div>
          )}

          {project.figures?.length > 0 && (
            <div className="modal-figs">
              {project.figures.map((f, i) => (
                <img
                  key={f.url}
                  className="modal-fig"
                  src={asset(f.url)}
                  alt={f.name}
                  loading="lazy"
                  onClick={() => onOpenFig(project.figures, i)}
                />
              ))}
            </div>
          )}

          <div className="modal-links">
            <a className="btn" href={`${repoUrl}/tree/main/${project.repoPath}`} target="_blank" rel="noreferrer">
              <IconExternal /> Code
            </a>
            <a className="btn" href={`${repoUrl}/blob/main/${project.repoPath}/README.md`} target="_blank" rel="noreferrer">
              <IconExternal /> Readme
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
