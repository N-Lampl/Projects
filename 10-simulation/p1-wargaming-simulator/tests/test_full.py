"""Slow statistical tests (excluded from CI via ``-m 'not slow'``).

These run large Monte Carlo batches to check the engine is fair and that the sensitivity sweep and
common-random-number A/B behave as designed.
"""

from __future__ import annotations

import pytest

from dow_sim import monte_carlo, sensitivity_sweep
from dow_sim.montecarlo import compare


@pytest.mark.slow
def test_symmetric_scenario_is_fair():
    """A mirror-symmetric scenario with identical policies should be near 50/50."""
    mc = monte_carlo("meeting_engagement", "aggressive", "aggressive", n=1500)
    assert 0.44 <= mc.blue_winrate <= 0.56


@pytest.mark.slow
def test_sensitivity_is_monotonic_in_red_count():
    """More RED units should monotonically lower BLUE's win probability."""
    sw = sensitivity_sweep(
        "meeting_engagement", "red_count", [1, 2, 3], "aggressive", "defensive", n=600
    )
    rates = sw["winrate"]
    assert rates[0] >= rates[1] >= rates[2]
    assert rates[0] - rates[2] > 0.1  # the effect is clearly visible


@pytest.mark.slow
def test_ab_delta_is_stable_under_common_random_numbers():
    """The A/B win-rate delta should be well-defined and within [-1, 1]."""
    ab = compare(
        "combined_arms", ("aggressive", "defensive"), ("payoff", "defensive"), n=1000
    )
    assert ab["common_random_numbers"] is True
    assert -1.0 <= ab["delta"] <= 1.0
    # Aggression should not do worse than the cautious payoff policy when forced to attack.
    assert ab["arm_a"]["winrate"] >= ab["arm_b"]["winrate"] - 0.05
