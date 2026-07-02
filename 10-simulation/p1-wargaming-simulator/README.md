# DOW - Tactical War-Gaming Simulator

A hex-grid tactical combat simulator with a **Monte Carlo analysis** layer that estimates
win probabilities, casualty distributions, and parameter sensitivities across scenarios. Pure
Python, **CPU-only, fully offline, no LLM and no API keys** - the decision-making is a rules
engine plus heuristic and game-theoretic AI policies.

> **Scope & ethics.** This is an *abstract, illustrative* simulation of stylized grid combat for
> statistical and decision-analysis education. It models no real entities, doctrine, or weapon
> systems and is **not** a targeting, planning, or operational tool. See the repo
> [ETHICS.md](../../ETHICS.md).

## What it does

- **Deterministic rules engine** on an axial **hex** grid: terrain (clear/forest/hill/urban/water)
  with move cost, defensive modifiers, and line-of-sight blocking; four unit types
  (infantry, armor, artillery, air) with distinct attack/defense/move/range profiles.
- **Two selectable combat models**, both RNG-driven and reproducible from a seed: a classic
  odds-ratio **Combat Results Table (CRT)** and a continuous **Lanchester** attrition model.
- **Heuristic AI policies**: `aggressive`, `defensive`, `objective`, and a game-theoretic
  `payoff` policy that maximises expected casualty exchange (with a closed-form 2x2 Nash solver
  in [`payoff.py`](src/dow_sim/payoff.py)).
- **Monte Carlo analysis**: win probability with **Wilson confidence intervals**, casualty and
  battle-length distributions, a **sensitivity sweep** over a force-ratio parameter, and an A/B
  policy comparison that uses **common random numbers** for a tight win-rate delta.
- **Interactive Streamlit dashboard** with a turn-by-turn battle visualizer.

Every stochastic call threads an explicit `numpy.random.Generator`, so thousands of battles stay
independent yet fully reproducible - and parallelise across processes.

## Scenarios

| Scenario | Setup |
|----------|-------|
| `meeting_engagement` | Mirror-symmetric infantry+armor clash. A fair fight lands near 50% - the engine's sanity check. |
| `seize_the_ridge` | BLUE must seize and hold a fortified hill for 3 turns against a dug-in RED defense with artillery. |
| `combined_arms` | BLUE armor+artillery+air assaults entrenched RED across a river with a single crossing. |

## Quickstart

```bash
make test            # fast offline smoke tests
make run             # Monte Carlo experiment -> results/metrics.json + figures
make sim   ARGS='meeting_engagement --blue payoff --red defensive'   # one battle, turn-by-turn
make montecarlo ARGS='seize_the_ridge --blue aggressive --n 2000'    # win prob + CI
make dashboard       # launch the Streamlit UI (needs the `dashboard` extra)
```

The CLI is also available directly:

```bash
python scripts/dow.py scenario list
python scripts/dow.py sim combined_arms --blue payoff --red defensive --seed 7
python scripts/dow.py montecarlo meeting_engagement --n 2000
python scripts/dow.py sweep meeting_engagement --param red_count --values 1 2 3
```

## Results

`make run` regenerates [`results/metrics.json`](results/metrics.json) and the figures in
[`results/figures/`](results/figures/) deterministically (seed 42). Headline findings:

- The **symmetric** meeting engagement is a near-even ~50% for BLUE, confirming the engine has no
  built-in side bias (initiative is randomised each turn).
- Taking a **fortified ridge** under a turn limit is much harder - terrain and dug-in defenders
  swing the odds toward the defender.
- Win probability falls monotonically as RED gains units (the sensitivity sweep).
- The cautious **payoff** policy under-performs raw aggression *when forced to attack* on a clock -
  expected-value play alone won't force a decision in time.

## Install

```bash
pip install -r requirements.txt                 # numpy + matplotlib (core)
pip install -e '.[dashboard,dev]'               # + streamlit, pytest, ruff
```

## Layout

```
src/dow_sim/       engine, mechanics (hexgrid/terrain/units/movement/los/combat),
                   policies + payoff, montecarlo + metrics, plots + render
scripts/           run_analysis.py (make run), dow.py (CLI), dashboard.py (Streamlit)
tests/             test_smoke.py (fast, CI), test_full.py (@slow statistical checks)
data/scenarios/    JSON mirrors of the in-code scenario registry
results/           metrics.json + figures (committed evidence)
```
