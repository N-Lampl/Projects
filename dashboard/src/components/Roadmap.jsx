export default function Roadmap({ items }) {
  return (
    <div className="roadmap">
      {items.map((r, i) => (
        <div className="rm" key={r.title}>
          <div className="rm-num">{String(i + 1).padStart(2, "0")}</div>
          <h4>{r.title}</h4>
          <p>{r.detail}</p>
        </div>
      ))}
    </div>
  );
}
