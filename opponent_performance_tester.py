"""Adaptive opponent performance tester.

This script evaluates the redesigned RPS AI system against several human move
simulators. It produces per-game data, summary statistics, visual reports, and a
statistical analysis to highlight how difficulty, strategy, and personality
influence outcomes under different human play styles.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

from human_simulators import SIMULATOR_REGISTRY, BaseHumanSimulator, SimulatorState
from move_mapping import MOVES, get_counter_move, normalize_move
from rps_ai_system import Difficulty, Personality, RPSAISystem, Strategy

sns.set_theme(style="whitegrid")


@dataclass
class SimulationConfig:
    games_per_opponent: int = 60
    rounds_per_game: int = 120
    seed: int = 20241105
    output_dir: Path = Path("simulation_results")


def determine_outcome(human_move: str, ai_move: str) -> str:
    human_move = normalize_move(human_move)
    ai_move = normalize_move(ai_move)
    if human_move == ai_move:
        return "tie"
    if get_counter_move(ai_move) == human_move:
        return "human"
    if get_counter_move(human_move) == ai_move:
        return "robot"
    return "tie"


def build_rng_seed(sim_name: str, difficulty: str, strategy: str, personality: str, base_seed: int) -> int:
    key = (sim_name, difficulty, strategy, personality, base_seed)
    return abs(hash(key)) % (2**32)


def instantiate_simulator(name: str, rng: np.random.Generator) -> BaseHumanSimulator:
    cls = SIMULATOR_REGISTRY[name]
    return cls(rng)


def play_game(
    simulator_name: str,
    difficulty: str,
    strategy: str,
    personality: str,
    rounds: int,
    base_seed: int,
    game_index: int,
) -> Dict:
    combo_seed = build_rng_seed(simulator_name, difficulty, strategy, personality, base_seed)
    rng = np.random.default_rng(combo_seed + game_index)
    simulator = instantiate_simulator(simulator_name, rng)

    ai_seed = int(combo_seed + game_index * 17)
    ai_system = RPSAISystem(seed=ai_seed)
    if not ai_system.set_opponent(difficulty, strategy, personality):
        raise ValueError(f"Unable to set opponent {difficulty}/{strategy}/{personality}")

    human_history: List[str] = []
    ai_history: List[str] = []
    results: List[str] = []

    human_round_wins = robot_round_wins = tie_rounds = 0

    for round_index in range(rounds):
        probs, ai_move, metadata = ai_system.predict_next_move(list(human_history))
        state = SimulatorState(
            round_index=round_index,
            human_history=human_history.copy(),
            ai_history=ai_history.copy(),
            results=results.copy(),
            ai_metadata=metadata,
            ai_move=ai_move,
        )
        human_move = simulator.select_move(state)
        human_move = normalize_move(human_move)

        outcome = determine_outcome(human_move, ai_move)
        results.append(outcome)
        human_history.append(human_move)
        ai_history.append(ai_move)

        ai_system.update_with_human_move(human_move, ai_move)
        simulator.observe_outcome(
            SimulatorState(
                round_index=round_index,
                human_history=human_history.copy(),
                ai_history=ai_history.copy(),
                results=results.copy(),
                ai_metadata=metadata,
                ai_move=ai_move,
            ),
            outcome,
        )

        if outcome == "human":
            human_round_wins += 1
        elif outcome == "robot":
            robot_round_wins += 1
        else:
            tie_rounds += 1

    total_rounds = human_round_wins + robot_round_wins + tie_rounds
    return {
        "human_simulator": simulator_name,
        "difficulty": difficulty,
        "strategy": strategy,
        "personality": personality,
        "human_round_wins": human_round_wins,
        "robot_round_wins": robot_round_wins,
        "tie_rounds": tie_rounds,
        "human_round_win_rate": human_round_wins / total_rounds if total_rounds else 0.0,
        "robot_round_win_rate": robot_round_wins / total_rounds if total_rounds else 0.0,
        "tie_round_rate": tie_rounds / total_rounds if total_rounds else 0.0,
        "human_game_win": int(human_round_wins > robot_round_wins),
        "robot_game_win": int(robot_round_wins > human_round_wins),
        "tie_game": int(human_round_wins == robot_round_wins),
    }


def run_simulations(config: SimulationConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict] = []
    combos = list(
        itertools.product(
            SIMULATOR_REGISTRY.keys(),
            [d.value for d in Difficulty],
            [s.value for s in Strategy],
            [p.value for p in Personality],
        )
    )
    total_jobs = len(combos) * config.games_per_opponent
    job_counter = 0

    for sim_name, difficulty, strategy, personality in combos:
        for game_index in range(config.games_per_opponent):
            job_counter += 1
            result = play_game(
                simulator_name=sim_name,
                difficulty=difficulty,
                strategy=strategy,
                personality=personality,
                rounds=config.rounds_per_game,
                base_seed=config.seed,
                game_index=game_index,
            )
            result["game_index"] = game_index
            records.append(result)
            if job_counter % 100 == 0 or job_counter == total_jobs:
                print(f"Progress: {job_counter}/{total_jobs} games simulated", flush=True)

    df_games = pd.DataFrame(records)

    summary = (
        df_games
        .groupby(["human_simulator", "difficulty", "strategy", "personality"], as_index=False)
        .agg(
            human_game_win_rate=("human_game_win", "mean"),
            robot_game_win_rate=("robot_game_win", "mean"),
            tie_game_rate=("tie_game", "mean"),
            human_round_win_rate=("human_round_win_rate", "mean"),
            robot_round_win_rate=("robot_round_win_rate", "mean"),
            tie_round_rate=("tie_round_rate", "mean"),
            games_played=("human_game_win", "count"),
        )
    )

    return df_games, summary


def annotate_quartiles(ax: plt.Axes, data: pd.DataFrame, group_field: str, metric: str) -> None:
    positions = {category: idx for idx, category in enumerate(sorted(data[group_field].unique()))}
    for category, idx in positions.items():
        values = data.loc[data[group_field] == category, metric].dropna()
        if values.empty:
            continue
        q1 = values.quantile(0.25)
        median = values.quantile(0.5)
        q3 = values.quantile(0.75)
        ax.text(
            idx,
            min(q3 + 0.03, 1.05),
            f"Q3={q3:.2f}\nMed={median:.2f}\nQ1={q1:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )


def create_boxplots(
    df_summary: pd.DataFrame,
    human_simulator: str,
    group_field: str,
    output_dir: Path,
) -> List[Path]:
    metric_labels = [
        ("human_game_win_rate", "Human game win rate"),
        ("robot_game_win_rate", "Robot game win rate"),
        ("tie_game_rate", "Tie game rate"),
    ]

    subset = df_summary[df_summary["human_simulator"] == human_simulator]
    if subset.empty:
        return []

    paths: List[Path] = []
    fig, axes = plt.subplots(1, len(metric_labels), figsize=(5 * len(metric_labels), 4), sharey=True)

    for ax, (metric, label) in zip(axes, metric_labels):
        sns.boxplot(data=subset, x=group_field, y=metric, ax=ax)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(group_field.title())
        ax.set_ylabel(label)
        annotate_quartiles(ax, subset, group_field, metric)

    fig.suptitle(f"{human_simulator.title()} simulator – grouped by {group_field}")
    fig.tight_layout()
    path = output_dir / f"{human_simulator}_{group_field}_boxplots.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def generate_visualisations(df_summary: pd.DataFrame, output_dir: Path) -> List[Path]:
    generated: List[Path] = []
    for simulator_name in df_summary["human_simulator"].unique():
        for group_field in ["difficulty", "strategy", "personality"]:
            generated.extend(create_boxplots(df_summary, simulator_name, group_field, output_dir))
    return generated


def build_statistical_report(df_summary: pd.DataFrame, output_dir: Path) -> Path:
    metrics = {
        "human_game_win_rate": "Human game win rate",
        "robot_game_win_rate": "Robot game win rate",
        "tie_game_rate": "Tie game rate",
    }
    factors = ["human_simulator", "difficulty", "strategy", "personality"]

    lines: List[str] = []
    lines.append("Statistical analysis of adaptive opponent performance")
    lines.append("")

    for metric, label in metrics.items():
        formula = f"{metric} ~ " + " + ".join(f"C({factor})" for factor in factors)
        model = ols(formula, data=df_summary).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        total_ss = anova_table["sum_sq"].sum()
        anova_table["eta_sq"] = anova_table["sum_sq"] / total_ss if total_ss else 0.0

        lines.append(f"Metric: {label}")
        lines.append(anova_table.to_string())
        lines.append("")

        for factor in factors:
            grouped = df_summary.groupby(factor)[metric].describe()
            lines.append(f"Descriptive statistics by {factor}")
            lines.append(grouped.to_string())
            lines.append("")

        lines.append("-" * 80)

    report_path = output_dir / "statistical_report.txt"
    report_path.write_text("\n".join(lines))
    return report_path


def save_raw_outputs(df_games: pd.DataFrame, df_summary: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path]:
    games_path = output_dir / "game_results.csv"
    summary_path = output_dir / "summary_results.csv"
    df_games.to_csv(games_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    return games_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive opponent simulations")
    parser.add_argument("--games", type=int, default=60, help="Games per opponent configuration")
    parser.add_argument("--rounds", type=int, default=120, help="Rounds per game")
    parser.add_argument("--seed", type=int, default=20241105, help="Global random seed")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output directory (default: simulation_results/<timestamp>)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_output = Path(args.output) if args.output else Path("simulation_results") / f"adaptive_report_{timestamp}"
    base_output.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(
        games_per_opponent=args.games,
        rounds_per_game=args.rounds,
        seed=args.seed,
        output_dir=base_output,
    )

    print("Starting simulations...")
    df_games, df_summary = run_simulations(config)

    print("Saving raw outputs...")
    games_path, summary_path = save_raw_outputs(df_games, df_summary, base_output)

    print("Generating visualisations...")
    plot_paths = generate_visualisations(df_summary, base_output)

    print("Building statistical report...")
    report_path = build_statistical_report(df_summary, base_output)

    manifest = {
        "games_csv": str(games_path),
        "summary_csv": str(summary_path),
        "plots": [str(path) for path in plot_paths],
        "statistical_report": str(report_path),
    }
    manifest_path = base_output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("Simulation complete. Outputs saved to", base_output)


if __name__ == "__main__":
    main()
