import InjectionDemo from "../playground/InjectionDemo.jsx";
import FraudDemo from "../playground/FraudDemo.jsx";
import "../playground/playground.css";

export default function Playground() {
  return (
    <div className="pg-intro">
      <p className="lead">
        Most security portfolios show you screenshots. These two run the{" "}
        <strong>actual trained models</strong> live in your browser — no backend, no API calls.
        The weights were exported straight from the Python projects and the scoring math is
        reproduced in JavaScript, matching the originals to ~1e-6. Try to break them.
      </p>

      <div className="pg-demos">
        <InjectionDemo />
        <FraudDemo />
      </div>
    </div>
  );
}
