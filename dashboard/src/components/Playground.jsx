import InjectionDemo from "../playground/InjectionDemo.jsx";
import FraudDemo from "../playground/FraudDemo.jsx";
import "../playground/playground.css";

export default function Playground() {
  return (
    <div className="pg-intro">
      <p className="lead">
        Most security portfolios just show screenshots. These two run the{" "}
        <strong>actual trained models</strong> live in your browser, with no backend and no
        API calls. The weights come straight from the Python projects, and the scoring is
        reproduced in JavaScript to within 1e-6 of the originals. Try to break them.
      </p>

      <div className="pg-demos">
        <InjectionDemo />
        <FraudDemo />
      </div>
    </div>
  );
}
