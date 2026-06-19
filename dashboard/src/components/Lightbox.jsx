import { useEffect } from "react";

const asset = (url) => `${import.meta.env.BASE_URL}${url}`;

export default function Lightbox({ figures, index, onClose, onMove }) {
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight") onMove(1);
      if (e.key === "ArrowLeft") onMove(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, onMove]);

  if (index == null || !figures?.length) return null;
  const fig = figures[index];

  return (
    <div className="lb" onClick={onClose}>
      <button className="x" onClick={onClose} aria-label="close">×</button>
      {figures.length > 1 && (
        <button className="nav prev" onClick={(e) => { e.stopPropagation(); onMove(-1); }}>‹</button>
      )}
      <img src={asset(fig.url)} alt={fig.name} onClick={(e) => e.stopPropagation()} />
      {figures.length > 1 && (
        <button className="nav next" onClick={(e) => { e.stopPropagation(); onMove(1); }}>›</button>
      )}
      <div className="cap">{fig.name} — {index + 1}/{figures.length}</div>
    </div>
  );
}
