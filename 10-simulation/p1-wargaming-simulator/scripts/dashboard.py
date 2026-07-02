#!/usr/bin/env python3
"""Streamlit dashboard for the DOW war-gaming simulator.

Four sections: a scenario browser, a turn-by-turn single-battle visualizer, a Monte Carlo panel,
and a sensitivity / what-if panel. Run with ``make dashboard``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from dow_sim.engine import run_battle  # noqa: E402
from dow_sim.montecarlo import monte_carlo, sensitivity_sweep  # noqa: E402
from dow_sim.policies import POLICIES, make_policy  # noqa: E402
from dow_sim.render import render_board  # noqa: E402
from dow_sim.scenarios import list_scenarios, load_scenario  # noqa: E402

POLICY_NAMES = sorted(POLICIES)
SCENARIO_NAMES = [n for n, _ in list_scenarios()]

st.set_page_config(page_title="DOW War-Gaming Simulator", layout="wide")
st.title("DOW - Tactical War-Gaming Simulator")
st.caption(
    "Abstract hex-grid combat with Monte Carlo win-probability analysis. Offline, CPU-only, "
    "no LLM. Illustrative simulation for decision-analysis education only - not operational."
)

# --- sidebar controls ------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    scenario = st.selectbox("Scenario", SCENARIO_NAMES)
    blue = st.selectbox("BLUE policy", POLICY_NAMES, index=POLICY_NAMES.index("aggressive"))
    red = st.selectbox("RED policy", POLICY_NAMES, index=POLICY_NAMES.index("defensive"))
    model = st.selectbox("Combat model", ["crt", "lanchester"])
    seed = st.number_input("Seed", min_value=0, value=42, step=1)
    n_sims = st.slider("Monte Carlo battles", 100, 3000, 500, step=100)

section = st.radio(
    "View",
    ["Scenario browser", "Single battle", "Monte Carlo", "Sensitivity / what-if"],
    horizontal=True,
)


@st.cache_data(show_spinner=False)
def _battle_snapshots(scenario: str, blue: str, red: str, seed: int, model: str):
    state = load_scenario(scenario)
    result, snapshots = run_battle(
        state, make_policy(blue), make_policy(red), seed=int(seed), model=model, record=True
    )
    tiles = state.tiles
    objective = state.objective
    frames = [(s.turn, s.units, list(s.log)) for s in snapshots]
    return result, frames, tiles, objective


@st.cache_data(show_spinner=True)
def _mc(scenario, blue, red, n, seed, model):
    return monte_carlo(scenario, blue, red, n=int(n), base_seed=int(seed), model=model)


if section == "Scenario browser":
    state = load_scenario(scenario)
    st.subheader(scenario)
    st.write(dict(list_scenarios())[scenario])
    fig = render_board(
        state.tiles,
        [(u.uid, u.side, u.kind.value, u.hp, u.pos.q, u.pos.r) for u in state.units.values()],
        state.objective,
        title="Starting deployment",
    )
    st.pyplot(fig)

elif section == "Single battle":
    result, frames, tiles, objective = _battle_snapshots(scenario, blue, red, seed, model)
    st.subheader(f"Result: {result.winner} in {result.turns} turns")
    st.write(
        f"BLUE lost {result.blue_losses} unit(s), RED lost {result.red_losses} unit(s). "
        f"Scrub the turn slider to replay the battle."
    )
    idx = st.slider("Turn", 0, len(frames) - 1, 0)
    turn, units, log = frames[idx]
    col1, col2 = st.columns([3, 2])
    with col1:
        st.pyplot(render_board(tiles, units, objective, title=f"Turn {turn}"))
    with col2:
        st.markdown("**Combat log this turn**")
        st.write("\n".join(f"- {line}" for line in log) or "_no engagements_")

elif section == "Monte Carlo":
    mc = _mc(scenario, blue, red, n_sims, seed, model)
    c1, c2, c3 = st.columns(3)
    c1.metric("BLUE win probability", f"{mc.blue_winrate:.1%}",
              help=f"95% Wilson CI ({mc.ci[0]:.1%}, {mc.ci[1]:.1%})")
    c2.metric("Mean battle length", f"{mc.mean_turns:.1f} turns")
    c3.metric("Draws", f"{mc.draws}/{mc.n}")
    st.write(
        f"Wins - BLUE {mc.blue_wins}, RED {mc.red_wins}, draws {mc.draws}. "
        f"BLUE avg losses {mc.blue_losses['mean']:.2f}, RED avg losses {mc.red_losses['mean']:.2f}."
    )

elif section == "Sensitivity / what-if":
    param = st.selectbox("Parameter", ["red_count", "blue_count", "hold_turns"])
    lo, hi = st.slider("Value range", 1, 5, (1, 3))
    values = list(range(lo, hi + 1))
    if st.button("Run sweep"):
        sw = sensitivity_sweep(
            scenario, param, values, blue, red, n=min(n_sims, 500), base_seed=int(seed), model=model
        )
        st.line_chart({"BLUE win rate": sw["winrate"]})
        st.write(
            {f"{param}={v}": f"{w:.1%}" for v, w in zip(sw["values"], sw["winrate"], strict=True)}
        )
