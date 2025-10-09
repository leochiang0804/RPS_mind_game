import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_autopilot import AutopilotAgent


def _print_log(title, log):
    print(f"\n--- {title} ---")
    for line in log:
        print(line)


def test_autopilot_high_confidence(capsys):
    context = {
        "human_moves": ["rock", "paper", "paper", "scissors"],
        "robot_moves": ["paper", "scissors", "rock", "paper"],
        "results": ["robot", "human", "tie", "robot"],
        "round": 4,
        "opponent_info": {
            "ai_difficulty": "challenger",
            "ai_strategy": "to_win",
            "ai_personality": "neutral",
        },
        "ai_prediction": {
            "human_prediction": [0.18, 0.64, 0.18],
            "confidence": 0.71,
            "metadata": {
                "move_selection": {
                    "adjusted_distribution": [0.08, 0.14, 0.78],
                },
                "human_model": {
                    "pattern_strength": 0.62,
                    "recent_repetition": {"move": "paper", "length": 2},
                    "change_factor": 0.05,
                },
            },
        },
        "game_status": {
            "metrics": {"human_win_rate": 0.25, "robot_win_rate": 0.5, "tie_rate": 0.25}
        },
    }

    agent = AutopilotAgent()
    result = agent.run(context)
    _print_log("High Confidence Scenario", result.log)
    captured = capsys.readouterr().out

    assert "Stage tool_predict" in captured
    assert "Stage tool_policy" in captured
    assert "finalize" in captured
    assert result.predicted_robot_move == "scissors"
    assert result.final_move == "rock"


def test_autopilot_drift_triggers_wildcard(capsys):
    context = {
        "human_moves": ["rock", "rock", "paper", "scissors", "paper", "rock"],
        "robot_moves": ["paper", "paper", "scissors", "rock", "scissors", "paper"],
        "results": ["robot", "robot", "robot", "human", "robot", "robot"],
        "round": 6,
        "opponent_info": {
            "ai_difficulty": "master",
            "ai_strategy": "to_win",
            "ai_personality": "unpredictable",
        },
        "ai_prediction": {
            "human_prediction": [0.34, 0.33, 0.33],
            "confidence": 0.42,
            "metadata": {
                "move_selection": {
                    "adjusted_distribution": [0.36, 0.33, 0.31],
                },
                "human_model": {
                    "pattern_strength": 0.18,
                    "recent_repetition": {"move": "rock", "length": 1},
                    "change_factor": 0.32,
                },
            },
        },
        "game_status": {
            "metrics": {"human_win_rate": 0.17, "robot_win_rate": 0.66, "tie_rate": 0.17}
        },
    }

    agent = AutopilotAgent()
    result = agent.run(context)
    _print_log("Drift Scenario", result.log)
    captured = capsys.readouterr().out

    assert "tool_analytics" in captured
    assert "tool_wildcard" in captured
    assert result.predicted_robot_move == "rock"
    wildcard = result.tool_outputs.get("wildcard", {})
    assert wildcard.get("override_move") == result.final_move


if __name__ == "__main__":
    pytest.main([__file__])
