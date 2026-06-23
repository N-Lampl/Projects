import { useEffect } from "react";
import { IconChevron, IconClose } from "./icons.jsx";

const asset = (u) => `${import.meta.env.BASE_URL}${u}`;

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
      <button className="icon-btn x" onClick={onClose} aria-label="Close image"><IconClose /></button>
      {figures.length > 1 && (
        <button
          className="icon-btn"
          style={{ position: "absolute", left: 12 }}
          onClick={(e) => { e.stopPropagation(); onMove(-1); }}
          aria-label="Previous"
        >
          <IconChevron style={{ transform: "rotate(180deg)" }} />
        </button>
      )}
      <img src={asset(fig.url)} alt={fig.name} onClick={(e) => e.stopPropagation()} />
      {figures.length > 1 && (
        <button
          className="icon-btn"
          style={{ position: "absolute", right: 12 }}
          onClick={(e) => { e.stopPropagation(); onMove(1); }}
          aria-label="Next"
        >
          <IconChevron />
        </button>
      )}
      <div className="cap">{fig.name} — {index + 1}/{figures.length}</div>
    </div>
  );
}
