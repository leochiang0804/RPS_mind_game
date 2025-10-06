"""Human move simulators for Rock-Paper-Scissors testing.

Each simulator exposes a consistent interface so the performance tester can
instantiate and use them interchangeably. All randomness is routed through the
provided NumPy Generator to keep simulations reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from move_mapping import MOVES, get_counter_move, normalize_move


@dataclass
class SimulatorState:
    """Snapshot of the current match state provided to simulators."""

    round_index: int
    human_history: List[str]
    ai_history: List[str]
    results: List[str]  # 'human', 'robot', or 'tie'
    ai_metadata: Dict
    ai_move: str


class BaseHumanSimulator:
    """Base class with helper utilities for concrete simulators."""

    name: str = "base"

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.reset()

    def reset(self) -> None:
        self.last_move: Optional[str] = None
        self.last_result: Optional[str] = None
        self.loss_streak: int = 0

    def select_move(self, state: SimulatorState) -> str:
        raise NotImplementedError

    def observe_outcome(self, state: SimulatorState, outcome: str) -> None:
        self.last_move = state.human_history[-1] if state.human_history else None
        self.last_result = outcome
        if outcome == 'robot':
            self.loss_streak += 1
        else:
            self.loss_streak = 0

    def _random_choice(self, probabilities: Optional[np.ndarray] = None) -> str:
        if probabilities is None:
            return str(self.rng.choice(MOVES))
        probabilities = np.asarray(probabilities, dtype=float)
        probabilities = probabilities / probabilities.sum()
        idx = int(self.rng.choice(len(MOVES), p=probabilities))
        return MOVES[idx]


class RandomHumanSimulator(BaseHumanSimulator):
    name = "random"

    def select_move(self, state: SimulatorState) -> str:
        return self._random_choice()


class RealisticHumanSimulator(BaseHumanSimulator):
    """Heuristic simulator approximating common human tendencies."""

    name = "realistic"

    def reset(self) -> None:
        super().reset()
        self.preference = np.array([0.42, 0.33, 0.25], dtype=float)  # rock, paper, scissors

    def select_move(self, state: SimulatorState) -> str:
        if not state.human_history:
            return self._random_choice(self.preference)

        last_outcome = self.last_result
        last_move = state.human_history[-1]

        if last_outcome == 'human':
            if self.rng.random() < 0.65:
                return last_move
            return self._random_choice(self.preference)

        if last_outcome == 'robot':
            preferred = get_counter_move(state.ai_history[-1]) if state.ai_history else None
            if preferred and self.rng.random() < 0.7:
                return preferred
            adjusted = self.preference.copy()
            idx = MOVES.index(last_move)
            adjusted[idx] *= 0.6
            return self._random_choice(adjusted)

        # tie
        if self.rng.random() < 0.5:
            return last_move
        return self._random_choice(self.preference)


class StrategicHumanSimulator(BaseHumanSimulator):
    """Simple win-stay, lose-shift with awareness of AI predictions."""

    name = "strategic"

    def select_move(self, state: SimulatorState) -> str:
        metadata = state.ai_metadata or {}
        predicted_probs = metadata.get('human_prediction', [1 / 3] * 3)
        predicted_probs = np.asarray(predicted_probs, dtype=float)
        least_expected = MOVES[int(np.argmin(predicted_probs))]

        if not state.human_history:
            if self.rng.random() < 0.5:
                return least_expected
            return self._random_choice()

        last_move = state.human_history[-1]
        last_outcome = self.last_result

        if last_outcome == 'human':
            if self.rng.random() < 0.7:
                return last_move
            return self._random_choice()

        if last_outcome == 'robot':
            counter = get_counter_move(state.ai_history[-1]) if state.ai_history else None
            if counter and self.rng.random() < 0.8:
                return counter
            return least_expected

        # tie case
        if self.rng.random() < 0.4:
            return least_expected
        if self.rng.random() < 0.5:
            return last_move
        return self._random_choice()


class ExplorerHumanSimulator(BaseHumanSimulator):
    """Optimises expected payoff using the AI move distribution."""

    name = "explorer"

    def select_move(self, state: SimulatorState) -> str:
        metadata = state.ai_metadata or {}
        move_selection = metadata.get('move_selection', {})
        distribution = move_selection.get('adjusted_distribution')

        if not distribution:
            # Fallback to strategic behaviour when distribution is unavailable
            strategic = StrategicHumanSimulator(self.rng)
            strategic.last_move = self.last_move
            strategic.last_result = self.last_result
            strategic.loss_streak = self.loss_streak
            return strategic.select_move(state)

        distribution = np.asarray(distribution, dtype=float)
        distribution = distribution / distribution.sum()

        best_moves: List[str] = []
        best_value = -np.inf
        for idx, human_move in enumerate(MOVES):
            expected_value = 0.0
            for ai_idx, ai_move in enumerate(MOVES):
                probability = float(distribution[ai_idx])
                if human_move == ai_move:
                    payoff = 0.0
                elif get_counter_move(human_move) == ai_move:
                    payoff = -1.0
                elif get_counter_move(ai_move) == human_move:
                    payoff = 1.0
                else:
                    payoff = 0.0
                expected_value += probability * payoff
            if expected_value > best_value + 1e-9:
                best_moves = [human_move]
                best_value = expected_value
            elif abs(expected_value - best_value) <= 1e-9:
                best_moves.append(human_move)

        if not best_moves:
            return self._random_choice()

        if len(best_moves) == 1:
            return best_moves[0]

        # Break ties by favouring moves the AI expects less from the human
        human_prediction = metadata.get('human_prediction')
        if human_prediction:
            human_prediction = np.asarray(human_prediction, dtype=float)
            human_prediction = human_prediction / human_prediction.sum()
            values = [human_prediction[MOVES.index(move)] for move in best_moves]
            min_value = min(values)
            candidates = [move for move, val in zip(best_moves, values) if abs(val - min_value) < 1e-9]
            if len(candidates) == 1:
                return candidates[0]
            best_moves = candidates

        return str(self.rng.choice(best_moves))


SIMULATOR_REGISTRY = {
    RandomHumanSimulator.name: RandomHumanSimulator,
    RealisticHumanSimulator.name: RealisticHumanSimulator,
    StrategicHumanSimulator.name: StrategicHumanSimulator,
    ExplorerHumanSimulator.name: ExplorerHumanSimulator,
}

