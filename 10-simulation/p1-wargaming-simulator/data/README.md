# Data

This project is **fully self-contained**. The three tactical scenarios it ships with are
defined in code (`../src/dow_sim/scenarios.py`) and mirrored as human-readable JSON under
[`scenarios/`](scenarios/) for inspection. No datasets are downloaded and nothing here is
committed except this README and the scenario JSON (`data/` is otherwise git-ignored).

## Scenarios

| File | Setup |
|------|-------|
| `scenarios/meeting_engagement.json` | Symmetric infantry+armor clash on mixed terrain. Baseline — a fair fight lands near 50% win probability, which is how we sanity-check the engine. |
| `scenarios/hold_the_ridge.json` | BLUE defends a hill objective against a larger RED attack. Shows terrain defense value and the payoff of a defensive policy. |
| `scenarios/combined_arms.json` | BLUE armor+artillery+air vs entrenched RED in urban terrain. Exercises every unit type and the game-theoretic payoff policy. |

The JSON and the in-code registry stay in sync; `scenarios.load_scenario(name)` reads the
in-code definition so tests and CI need no files on disk.

> Abstract, illustrative simulation for statistical and decision-analysis education only.
> Not a targeting, planning, or operational tool. See [../../../ETHICS.md](../../../ETHICS.md).
