import { IconArrow } from "./icons.jsx";

const num = (s) => {
  const v = parseFloat(String(s).replace("%", ""));
  return Number.isFinite(v) ? v : 0;
};

export default function Highlights({ highlights }) {
  return (
    <div className="highlights">
      {highlights.map((h) => {
        const b = num(h.before);
        const a = num(h.after);
        const neutralized = b > 0 ? Math.max(0, Math.min(100, ((b - a) / b) * 100)) : 0;
        return (
          <article className="hl" key={h.title}>
            <h3 className="hl-head">{h.title}</h3>
            <div className="hl-ba">
              <span className="metric before">{h.before}</span>
              <span className="arrow"><IconArrow width={16} height={16} /></span>
              <span className="metric after">{h.after}</span>
              <span className="hl-label">{h.label}</span>
            </div>
            <div
              className="hl-bar"
              role="img"
              aria-label={`${Math.round(neutralized)}% of the ${h.label} neutralized`}
            >
              <span style={{ width: `${neutralized}%` }} />
            </div>
            <p className="hl-blurb">{h.blurb}</p>
          </article>
        );
      })}
    </div>
  );
}
