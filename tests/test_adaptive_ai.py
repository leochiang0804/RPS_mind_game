import math
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rps_ai_system import RPSAISystem, _simulate


def simulate_profile(
    difficulty: str,
    strategy: str,
    personality: str,
    human_profile: str,
    setup_seed: int,
    sim_seed: int,
    rounds: int,
):
    system = RPSAISystem(seed=setup_seed)
    assert system.set_opponent(difficulty, strategy, personality)
    return _simulate(system, human_profile, rounds=rounds, seed=sim_seed)


def test_random_play_remains_balanced():
    summary = simulate_profile(
        difficulty="rookie",
        strategy="to_win",
        personality="neutral",
        human_profile="random",
        setup_seed=111,
        sim_seed=42,
        rounds=600,
    )
    total = summary["human_win_rate"] + summary["robot_win_rate"] + summary["tie_rate"]
    assert math.isclose(total, 1.0, abs_tol=1e-6)

    baseline = 1.0 / 3.0
    for rate in (summary["human_win_rate"], summary["robot_win_rate"], summary["tie_rate"]):
        assert abs(rate - baseline) < 0.05


def test_fast_switch_human_outpaces_master_aggressive():
    summary = simulate_profile(
        difficulty="master",
        strategy="to_win",
        personality="aggressive",
        human_profile="fast_switch",
        setup_seed=321,
        sim_seed=777,
        rounds=200,
    )
    assert summary["human_win_rate"] > summary["robot_win_rate"] + 0.02
    assert summary["tie_rate"] > 0.55


def test_unpredictable_has_higher_variance_than_cautious():
    unpredictable = simulate_profile(
        difficulty="master",
        strategy="to_win",
        personality="unpredictable",
        human_profile="random",
        setup_seed=321,
        sim_seed=123,
        rounds=400,
    )
    cautious = simulate_profile(
        difficulty="master",
        strategy="to_win",
        personality="cautious",
        human_profile="random",
        setup_seed=321,
        sim_seed=123,
        rounds=400,
    )
    assert unpredictable["robot_win_std"] > cautious["robot_win_std"]


def test_invalid_difficulty_is_rejected():
    system = RPSAISystem(seed=1)
    assert not system.set_opponent("mythic", "to_win", "neutral")
