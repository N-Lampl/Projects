import { useInView } from "../hooks.js";

export default function TrackChart({ tracks, max }) {
  const [ref, inView] = useInView();
  return (
    <div className="coverage" ref={ref}>
      {tracks.map((t) => {
        const pct = (t.projectIds.length / max) * 100;
        return (
          <div className="cov-row" key={t.id}>
            <span className="cov-name">{t.name}</span>
            <div className="cov-track">
              <div className="cov-fill" style={{ width: inView ? `${pct}%` : "0%" }} />
            </div>
            <span className="cov-num">{t.projectIds.length}</span>
          </div>
        );
      })}
    </div>
  );
}
