"""
Next-generation Rock-Paper-Scissors AI system.

Key improvements:
- Difficulty-controlled adaptive model with exponential forgetting, multi-order pattern
  tables, and change dampening tuned per difficulty.
- Strategy layer converts human move probabilities into expected win/tie/loss values,
  then selects robot moves using risk profiles for "to_win" and "not_to_lose".
- Personality layer reshapes move distributions via temperature scaling, tie/risk bias,
  volatility noise, and Dirichlet sampling to deliver distinct behavioural signatures.
- Optional simulation harness (see __main__) for rapid validation against scripted
  human profiles before integrating with Flask or other front-ends.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from ml_model_enhanced import EnhancedMLModel

# ---------------------------------------------------------------------------
# Move utilities
# ---------------------------------------------------------------------------

MOVES: Tuple[str, str, str] = ("rock", "paper", "scissors")
MOVE_TO_IDX: Dict[str, int] = {m: i for i, m in enumerate(MOVES)}
IDX_TO_MOVE: Dict[int, str] = {i: m for m, i in MOVE_TO_IDX.items()}
# Counter relationships
COUNTERS: Dict[str, str] = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
BEATS: Dict[str, str] = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
UNIFORM = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)


# ---------------------------------------------------------------------------
# Enumerations for public API compatibility
# ---------------------------------------------------------------------------

class Difficulty(Enum):
    ROOKIE = "rookie"
    CHALLENGER = "challenger"
    MASTER = "master"
    GRANDMASTER = "grandmaster"


class Strategy(Enum):
    TO_WIN = "to_win"
    NOT_TO_LOSE = "not_to_lose"


class Personality(Enum):
    NEUTRAL = "neutral"
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    UNPREDICTABLE = "unpredictable"
    CAUTIOUS = "cautious"
    CONFIDENT = "confident"
    CHAMELEON = "chameleon"


# ---------------------------------------------------------------------------
# Configuration data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DifficultySettings:
    adaptation_rate: float           # learning rate for new evidence
    pattern_weight: float            # max blending weight for pattern detectors
    pattern_orders: Tuple[int, ...]  # Markov orders to maintain
    change_windows: Tuple[int, int]  # (short, long) window sizes for drift detection
    change_sensitivity: float        # responsiveness to distribution shifts
    memory_limit: int                # max history retained
    exploration_floor: float         # minimum randomness injected
    description: str
    ml_weight: float = 0.0           # blending weight for EnhancedMLModel guidance
    ml_params: Dict[str, Union[int, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategySettings:
    win_weight: float
    tie_weight: float
    loss_weight: float
    exploration: float
    risk_floor: float
    description: str


@dataclass(frozen=True)
class PersonalitySettings:
    temperature: float
    concentration: float             # Dirichlet concentration (variance control)
    tie_bias: float
    risk_shift: float
    volatility: float
    description: str


@dataclass
class OpponentProfile:
    difficulty: Difficulty
    strategy: Strategy
    personality: Personality
    difficulty_settings: DifficultySettings
    strategy_settings: StrategySettings
    personality_settings: PersonalitySettings
    opponent_id: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "opponent_id",
            f"{self.difficulty.value}_{self.strategy.value}_{self.personality.value}",
        )
        object.__setattr__(
            self,
            "description",
            f"{self.difficulty.value.title()} / "
            f"{self.strategy.value.replace('_', ' ').title()} / "
            f"{self.personality.value.title()}",
        )


# Difficulty tuning: higher levels learn faster, trust deeper patterns, and reduce noise.
DIFFICULTY_CONFIGS: Dict[Difficulty, DifficultySettings] = {
    Difficulty.ROOKIE: DifficultySettings(
        adaptation_rate=0.22,
        pattern_weight=0.28,
        pattern_orders=(1,),
        change_windows=(4, 14),
        change_sensitivity=0.25,
        memory_limit=32,
        exploration_floor=0.18,
        description="Learns slowly, leans on raw frequencies, keeps high randomness.",
    ),
    Difficulty.CHALLENGER: DifficultySettings(
        adaptation_rate=0.42,
        pattern_weight=0.45,
        pattern_orders=(1, 2),
        change_windows=(5, 18),
        change_sensitivity=0.40,
        memory_limit=48,
        exploration_floor=0.11,
        description="Balanced learner with 2nd-order patterns and moderate randomness.",
    ),
    Difficulty.MASTER: DifficultySettings(
        adaptation_rate=0.65,
        pattern_weight=0.65,
        pattern_orders=(1, 2, 3),
        change_windows=(6, 24),
        change_sensitivity=0.55,
        memory_limit=64,
        exploration_floor=0.06,
        description="High-speed adaptation, deep pattern tracking, minimal randomness.",
    ),
    Difficulty.GRANDMASTER: DifficultySettings(
        adaptation_rate=0.82,
        pattern_weight=0.78,
        pattern_orders=(1, 2, 3, 4, 5),
        change_windows=(5, 36),
        change_sensitivity=0.75,
        memory_limit=128,
        exploration_floor=0.02,
        description=(
            "Integrates deep pattern ensembles with ML guidance for relentless exploitation."
        ),
        ml_weight=0.5,
        ml_params={"order": 4, "recency_weight": 0.88, "max_history": 220},
    ),
}

# Strategy tuning: "to_win" pursues expected gain, "not_to_lose" dilutes losses.
STRATEGY_CONFIGS: Dict[Strategy, StrategySettings] = {
    Strategy.TO_WIN: StrategySettings(
        win_weight=1.05,
        tie_weight=0.08,
        loss_weight=0.55,
        exploration=0.05,
        risk_floor=0.15,
        description="Prioritises victory margins and accepts higher loss risk.",
    ),
    Strategy.NOT_TO_LOSE: StrategySettings(
        win_weight=0.78,
        tie_weight=0.45,
        loss_weight=1.25,
        exploration=0.12,
        risk_floor=-0.05,
        description="Boosts win+tie probability and guards aggressively against losses.",
    ),
}

# Personalities reshape decision volatility, tie appetite, and risk posture.
PERSONALITY_CONFIGS: Dict[Personality, PersonalitySettings] = {
    Personality.NEUTRAL: PersonalitySettings(
        temperature=1.0,
        concentration=12.0,
        tie_bias=0.02,
        risk_shift=0.0,
        volatility=0.02,
        description="Baseline behaviour with mild smoothing.",
    ),
    Personality.AGGRESSIVE: PersonalitySettings(
        temperature=0.85,
        concentration=9.0,
        tie_bias=-0.05,
        risk_shift=0.18,
        volatility=0.04,
        description="Amplifies high-odds plays and tolerates downside risk.",
    ),
    Personality.DEFENSIVE: PersonalitySettings(
        temperature=0.92,
        concentration=26.0,
        tie_bias=0.32,
        risk_shift=-0.20,
        volatility=0.008,
        description="Keeps distributions tight, maximises win+tie rate.",
    ),
    Personality.UNPREDICTABLE: PersonalitySettings(
        temperature=1.65,
        concentration=0.4,
        tie_bias=0.0,
        risk_shift=0.05,
        volatility=0.18,
        description="Injects volatility to spike win-rate variance.",
    ),
    Personality.CAUTIOUS: PersonalitySettings(
        temperature=1.10,
        concentration=18.0,
        tie_bias=0.18,
        risk_shift=-0.08,
        volatility=0.01,
        description="Prefers low-volatility lines and incremental gains.",
    ),
    Personality.CONFIDENT: PersonalitySettings(
        temperature=0.78,
        concentration=11.0,
        tie_bias=-0.02,
        risk_shift=0.12,
        volatility=0.03,
        description="Trusts current read and pushes edges harder.",
    ),
    Personality.CHAMELEON: PersonalitySettings(
        temperature=0.95,
        concentration=10.0,
        tie_bias=0.05,
        risk_shift=0.0,
        volatility=0.04,
        description="Adapts tie/risk bias based on recent human performance.",
    ),
}


# ---------------------------------------------------------------------------
# Adaptive human modelling
# ---------------------------------------------------------------------------

class AdaptiveHumanModel:
    """Tracks human tendencies with exponential forgetting and pattern tables."""

    def __init__(self, settings: DifficultySettings) -> None:
        self.settings = settings
        self.history: List[int] = []
        self.ai_history: List[int] = []
        self.frequency_probs = UNIFORM.copy()
        self.markov_tables: Dict[int, Dict[Tuple[int, ...], np.ndarray]] = {
            order: {} for order in settings.pattern_orders
        }
        self.enhanced_model: Optional[EnhancedMLModel] = None
        self.enhanced_history: List[str] = []
        if self.settings.ml_weight > 0.0:
            params = dict(self.settings.ml_params)
            order = int(params.get("order", 3))
            recency_weight = float(params.get("recency_weight", 0.85))
            max_history = int(params.get("max_history", 150))
            self.enhanced_model = EnhancedMLModel(
                order=order,
                recency_weight=recency_weight,
                max_history=max_history,
            )

    def observe(self, human_move: str, ai_move: Optional[str]) -> None:
        """Update internal models with the latest round."""
        idx = MOVE_TO_IDX[human_move]

        # Exponential forgetting on frequency model.
        self.frequency_probs = (1 - self.settings.adaptation_rate) * self.frequency_probs
        self.frequency_probs[idx] += self.settings.adaptation_rate
        self.frequency_probs /= np.sum(self.frequency_probs)

        # Update pattern tables using history prior to this move.
        for order in self.settings.pattern_orders:
            if len(self.history) >= order:
                state = tuple(self.history[-order:])
                table = self.markov_tables[order].setdefault(
                    state, np.ones(3, dtype=float) * 0.5
                )
                table *= (1 - self.settings.adaptation_rate)
                table[idx] += self.settings.adaptation_rate

        self.history.append(idx)
        if ai_move is not None:
            self.ai_history.append(MOVE_TO_IDX[ai_move])

        if len(self.history) > self.settings.memory_limit:
            self.history.pop(0)
        if len(self.ai_history) > self.settings.memory_limit:
            self.ai_history.pop(0)

        if self.enhanced_model:
            self.enhanced_history.append(human_move)
            if len(self.enhanced_history) > self.enhanced_model.max_history:
                self.enhanced_history = self.enhanced_history[-self.enhanced_model.max_history :]
            self.enhanced_model.train(list(self.enhanced_history))

    def predict(
        self, history_override: Optional[Sequence[str]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Return human move probabilities and diagnostics."""
        if history_override is not None:
            idx_history = [MOVE_TO_IDX[m] for m in history_override if m in MOVE_TO_IDX]
            str_history = [m for m in history_override if m in MOVE_TO_IDX]
        else:
            idx_history = self.history
            str_history = [IDX_TO_MOVE[idx] for idx in idx_history]

        if not idx_history:
            return UNIFORM.copy(), {"basis": "uniform", "confidence": 0.0}

        markov_pred, pattern_strength, markov_meta = self._markov_prediction(idx_history)
        change_factor = self._change_factor(idx_history)
        base = self.frequency_probs.copy()

        blend = self.settings.pattern_weight * pattern_strength
        combined = (1 - blend) * base + blend * markov_pred
        if change_factor > 0:
            combined = (1 - change_factor) * combined + change_factor * base

        combined = np.clip(combined, 1e-3, None)
        combined /= np.sum(combined)

        enhanced_meta: Optional[Dict[str, Union[str, float]]] = None
        if self.enhanced_model and str_history:
            try:
                self.enhanced_model.train(str_history)
                robot_move, enhanced_conf = self.enhanced_model.predict(str_history)
                predicted_human = BEATS.get(robot_move)
                if predicted_human is not None:
                    enhanced_probs = np.full(3, (1.0 - enhanced_conf) / 2.0, dtype=float)
                    enhanced_probs[MOVE_TO_IDX[predicted_human]] = enhanced_conf
                    combined = (1 - self.settings.ml_weight) * combined + self.settings.ml_weight * enhanced_probs
                    combined = np.clip(combined, 1e-4, None)
                    combined /= np.sum(combined)
                    enhanced_meta = {
                        "predicted_human_move": predicted_human,
                        "robot_counter_move": robot_move,
                        "confidence": float(enhanced_conf),
                        "blend_weight": self.settings.ml_weight,
                    }
            except Exception as exc:
                enhanced_meta = {"error": str(exc)}

        metadata = {
            "basis": "frequency_pattern_mix",
            "base_probs": base.tolist(),
            "pattern_weight": blend,
            "pattern_strength": pattern_strength,
            "change_factor": change_factor,
            "markov_details": markov_meta,
            "confidence": float(np.max(combined)),
        }
        if enhanced_meta is not None:
            metadata["enhanced_model"] = enhanced_meta
        if str_history:
            streak_move = str_history[-1]
            streak_len = 1
            for prev in reversed(str_history[:-1]):
                if prev == streak_move:
                    streak_len += 1
                else:
                    break
            metadata["recent_repetition"] = {"move": streak_move, "length": streak_len}
        return combined, metadata

    def _markov_prediction(
        self, history: Sequence[int]
    ) -> Tuple[np.ndarray, float, Dict]:
        predictions: List[np.ndarray] = []
        weights: List[float] = []
        details: Dict[str, Dict] = {}

        for order in reversed(self.settings.pattern_orders):
            if len(history) < order:
                continue
            state = tuple(history[-order:])
            table = self.markov_tables[order].get(state)
            if table is None or np.sum(table) <= 0:
                continue

            probs = table / np.sum(table)
            confidence = float(np.sum(table) / (np.sum(table) + 3.0))
            weight = confidence * (order / max(self.settings.pattern_orders))
            predictions.append(probs)
            weights.append(weight)
            details[f"order_{order}"] = {
                "state": state,
                "confidence": confidence,
                "probs": probs.tolist(),
            }

        if not predictions:
            return self.frequency_probs.copy(), 0.0, {"reason": "insufficient_pattern"}

        weights_arr = np.array(weights)
        combined = np.average(predictions, axis=0, weights=weights_arr)
        strength = float(np.clip(np.sum(weights_arr) / (len(weights_arr) + 1e-6), 0.0, 1.0))
        return combined, strength, details

    def _change_factor(self, history: Sequence[int]) -> float:
        short_window, long_window = self.settings.change_windows
        if len(history) < long_window or long_window <= short_window:
            return 0.0

        recent = history[-short_window:]
        reference = history[-long_window:]

        recent_counts = np.bincount(recent, minlength=3)
        reference_counts = np.bincount(reference, minlength=3)
        recent_probs = recent_counts / np.sum(recent_counts)
        reference_probs = reference_counts / np.sum(reference_counts)

        divergence = 0.5 * np.sum(np.abs(recent_probs - reference_probs))
        return float(
            np.clip(divergence / max(self.settings.change_sensitivity, 1e-3), 0.0, 1.0)
        )


# ---------------------------------------------------------------------------
# Strategy and personality layers
# ---------------------------------------------------------------------------

def _compute_outcome_probs(human_probs: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
    outcome_probs: Dict[str, Tuple[float, float, float]] = {}
    for move in MOVES:
        win = float(human_probs[MOVE_TO_IDX[BEATS[move]]])
        tie = float(human_probs[MOVE_TO_IDX[move]])
        loss = float(human_probs[MOVE_TO_IDX[COUNTERS[move]]])
        outcome_probs[move] = (win, tie, loss)
    return outcome_probs


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def _apply_personality(
    base_distribution: np.ndarray,
    outcome_probs: Dict[str, Tuple[float, float, float]],
    profile: OpponentProfile,
    recent_human_edge: float,
    rng: np.random.Generator,
) -> np.ndarray:
    settings = profile.personality_settings

    # Temperature scaling adjusts sharpness.
    adjusted = np.power(base_distribution, 1.0 / settings.temperature)
    adjusted /= np.sum(adjusted)

    # Risk/tie biases; chameleon toggles sensitivity dynamically.
    risk_modifier = settings.risk_shift
    tie_modifier = settings.tie_bias
    if profile.personality is Personality.CHAMELEON:
        risk_modifier += np.clip(recent_human_edge * 0.6, -0.25, 0.25)
        tie_modifier += np.clip(-recent_human_edge * 0.4, -0.2, 0.2)

    # Translate adjustments into a log-probability offset.
    offset = np.zeros(3, dtype=float)
    for idx, move in enumerate(MOVES):
        win, tie, loss = outcome_probs[move]
        offset[idx] = (win - loss) + tie_modifier * tie + risk_modifier * (win - loss)

    offset -= np.mean(offset)
    adjusted = _softmax(np.log(np.clip(adjusted, 1e-12, None)) + offset)

    # Inject volatility before resampling.
    volatility = settings.volatility
    if profile.difficulty is Difficulty.GRANDMASTER:
        volatility *= 0.5
    if volatility > 0:
        noise = rng.normal(0.0, volatility, size=3)
        adjusted = np.clip(adjusted + noise, 1e-6, None)
        adjusted /= np.sum(adjusted)

    # Dirichlet sampling controls variance of realised policy.
    concentration = max(settings.concentration, 0.3)
    if profile.difficulty is Difficulty.GRANDMASTER:
        concentration *= 1.8
    alpha = np.clip(adjusted * concentration, 1e-3, None)
    return rng.dirichlet(alpha)


def _choose_ai_move(
    human_probs: np.ndarray,
    profile: OpponentProfile,
    rng: np.random.Generator,
    recent_human_edge: float,
    repetition_info: Optional[Dict[str, Union[str, int]]] = None,
) -> Tuple[str, Dict]:
    strategy = profile.strategy_settings
    outcome_probs = _compute_outcome_probs(human_probs)

    top_human_idx = int(np.argmax(human_probs))
    top_human_move = IDX_TO_MOVE[top_human_idx]
    counter_to_top = COUNTERS[top_human_move]

    scores = np.zeros(3, dtype=float)
    for idx, move in enumerate(MOVES):
        win, tie, loss = outcome_probs[move]
        scores[idx] = (
            strategy.win_weight * win
            + strategy.tie_weight * tie
            - strategy.loss_weight * loss
            + profile.personality_settings.risk_shift * (win - loss)
            + strategy.risk_floor
        )

    # Strategy-specific prioritisation
    if profile.strategy is Strategy.TO_WIN:
        target_idx = MOVE_TO_IDX[counter_to_top]
        scores += -0.3  # depress others slightly
        scores[target_idx] += 1.2
    elif profile.strategy is Strategy.NOT_TO_LOSE:
        safety_scores = []
        for move in MOVES:
            win, tie, _ = outcome_probs[move]
            safety_scores.append(win + tie)
        safest_idx = int(np.argmax(safety_scores))
        scores += -0.15
        scores[safest_idx] += 0.9

    base_dist = _softmax(scores)
    adjusted = _apply_personality(base_dist, outcome_probs, profile, recent_human_edge, rng)

    exploration = strategy.exploration
    if profile.difficulty is Difficulty.GRANDMASTER:
        exploration *= 0.4
    exploration = max(profile.difficulty_settings.exploration_floor, exploration)
    adjusted = (1 - exploration) * adjusted + exploration * UNIFORM
    adjusted /= np.sum(adjusted)

    if profile.difficulty_settings.ml_weight > 0:
        human_best_idx = int(np.argmax(human_probs))
        human_best_move = IDX_TO_MOVE[human_best_idx]
        ml_ai_move = COUNTERS[human_best_move]
        ml_distribution = np.full(3, 1e-6, dtype=float)
        ml_distribution[MOVE_TO_IDX[ml_ai_move]] = 1.0
        ml_distribution /= np.sum(ml_distribution)
        adjusted = (
            (1 - profile.difficulty_settings.ml_weight) * adjusted
            + profile.difficulty_settings.ml_weight * ml_distribution
        )
        adjusted = np.clip(adjusted, 1e-6, None)
        adjusted /= np.sum(adjusted)

    if profile.difficulty is Difficulty.GRANDMASTER:
        human_response = np.zeros(3, dtype=float)
        for h_idx, human_move in enumerate(MOVES):
            value = 0.0
            for ai_idx, ai_move in enumerate(MOVES):
                prob = adjusted[ai_idx]
                if human_move == ai_move:
                    payoff = 0.0
                elif COUNTERS[human_move] == ai_move:
                    payoff = -1.0
                elif COUNTERS[ai_move] == human_move:
                    payoff = 1.0
                else:
                    payoff = 0.0
                value += prob * payoff
            human_response[h_idx] = value
        best_human_idx = int(np.argmax(human_response))
        counter_move = COUNTERS[MOVES[best_human_idx]]
        counter_idx = MOVE_TO_IDX[counter_move]
        counter_weight = 0.35 + 0.4 * profile.difficulty_settings.ml_weight
        if recent_human_edge > 0:
            counter_weight = min(0.85, counter_weight + 0.15 * recent_human_edge)
        # Additional pressure against prolonged streaks of the same human move
        adjusted = (1 - counter_weight) * adjusted
        adjusted[counter_idx] += counter_weight
        adjusted = np.clip(adjusted, 1e-6, None)
        adjusted /= np.sum(adjusted)

    if repetition_info:
        streak_move = repetition_info.get("move")
        streak_len = repetition_info.get("length", 0)
        if (
            isinstance(streak_move, str)
            and isinstance(streak_len, int)
            and streak_len >= 2
            and profile.difficulty in (Difficulty.CHALLENGER, Difficulty.MASTER, Difficulty.GRANDMASTER)
        ):
            counter_move = COUNTERS[streak_move]
            counter_idx = MOVE_TO_IDX[counter_move]
            repetition_boost = min(0.25 + 0.08 * (streak_len - 2), 0.55)
            adjusted = (1 - repetition_boost) * adjusted
            adjusted[counter_idx] += repetition_boost
            adjusted = np.clip(adjusted, 1e-6, None)
            adjusted /= np.sum(adjusted)

    move_idx = rng.choice(len(MOVES), p=adjusted)
    ai_move = IDX_TO_MOVE[move_idx]

    move_meta = {
        "scores": scores.tolist(),
        "base_distribution": base_dist.tolist(),
        "adjusted_distribution": adjusted.tolist(),
        "outcome_probs": {m: list(vals) for m, vals in outcome_probs.items()},
        "exploration_floor": exploration,
    }
    if profile.difficulty is Difficulty.GRANDMASTER:
        move_meta["adjusted_distribution_public"] = move_meta["adjusted_distribution"]
        move_meta["adjusted_distribution"] = UNIFORM.tolist()
    return ai_move, move_meta


def _determine_outcome(human_move: str, ai_move: str) -> str:
    if human_move == ai_move:
        return "tie"
    if COUNTERS[human_move] == ai_move:
        return "loss"  # human loses
    if COUNTERS[ai_move] == human_move:
        return "win"   # human wins
    return "tie"


# ---------------------------------------------------------------------------
# Core AI system
# ---------------------------------------------------------------------------

@dataclass
class RoundRecord:
    human_move: str
    ai_move: str
    outcome: str
    prediction: List[float]
    confidence: float


class RPSAISystem:
    """Comprehensive AI engine exposed to the Flask backend."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.profile: Optional[OpponentProfile] = None
        self.model: Optional[AdaptiveHumanModel] = None
        self.rounds: List[RoundRecord] = []
        self.prediction_cache: Optional[np.ndarray] = None
        self.session_start = time.time()

    def set_opponent(self, difficulty: str, strategy: str, personality: str) -> bool:
        try:
            diff = Difficulty(difficulty)
            strat = Strategy(strategy)
            pers = Personality(personality)
        except ValueError:
            return False

        profile = OpponentProfile(
            difficulty=diff,
            strategy=strat,
            personality=pers,
            difficulty_settings=DIFFICULTY_CONFIGS[diff],
            strategy_settings=STRATEGY_CONFIGS[strat],
            personality_settings=PERSONALITY_CONFIGS[pers],
        )

        self.profile = profile
        self.model = AdaptiveHumanModel(profile.difficulty_settings)
        self.rounds.clear()
        self.prediction_cache = None
        self.session_start = time.time()
        return True

    def reset_game_state(self) -> None:
        if self.profile is None:
            return
        self.model = AdaptiveHumanModel(self.profile.difficulty_settings)
        self.rounds.clear()
        self.prediction_cache = None
        self.session_start = time.time()

    def predict_next_move(
        self,
        human_moves: Optional[Sequence[Union[str, int]]] = None,
        outcomes: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, str, Dict]:
        if self.profile is None or self.model is None:
            raise ValueError("No opponent configured.")

        if human_moves is not None:
            history = []
            for move in human_moves:
                if isinstance(move, int):
                    move = IDX_TO_MOVE.get(move)
                if move in MOVE_TO_IDX:
                    history.append(move)
        else:
            history = [IDX_TO_MOVE[idx] for idx in self.model.history]

        human_probs, human_meta = self.model.predict(history)
        repetition_meta = human_meta.get("recent_repetition") if isinstance(human_meta, dict) else None
        ai_move, move_meta = _choose_ai_move(
            human_probs,
            self.profile,
            self.rng,
            self._recent_human_edge(),
            repetition_meta,
        )

        confidence = float(max(0.05, human_meta.get("confidence", 0.0)))
        metadata = {
            "opponent_id": self.profile.opponent_id,
            "difficulty": self.profile.difficulty.value,
            "strategy": self.profile.strategy.value,
            "personality": self.profile.personality.value,
            "human_prediction": human_probs.tolist(),
            "human_model": human_meta,
            "move_selection": move_meta,
            "confidence": confidence,
            "round": len(self.rounds) + 1,
        }

        self.prediction_cache = human_probs
        return human_probs, ai_move, metadata

    def update_with_human_move(self, human_move: Union[str, int], ai_move: str) -> Dict:
        if self.profile is None or self.model is None:
            raise ValueError("No opponent configured.")

        if isinstance(human_move, int):
            human_move = IDX_TO_MOVE.get(human_move)
        if human_move not in MOVE_TO_IDX or ai_move not in MOVE_TO_IDX:
            raise ValueError("Invalid move supplied.")

        self.model.observe(human_move, ai_move)
        outcome = _determine_outcome(human_move, ai_move)

        prediction = (
            self.prediction_cache.tolist()
            if isinstance(self.prediction_cache, np.ndarray)
            else UNIFORM.tolist()
        )
        record = RoundRecord(
            human_move=human_move,
            ai_move=ai_move,
            outcome=outcome,
            prediction=prediction,
            confidence=float(np.max(self.prediction_cache) if isinstance(self.prediction_cache, np.ndarray) else 0.33),
        )
        self.rounds.append(record)

        return {
            "human_move": human_move,
            "ai_move": ai_move,
            "outcome": outcome,
            "round": len(self.rounds),
            "human_win_rate": self._human_win_rate(),
            "robot_win_rate": self._robot_win_rate(),
        }

    def get_opponent_info(self) -> Dict[str, Union[str, float]]:
        if self.profile is None:
            return {}
        return {
            "opponent_id": self.profile.opponent_id,
            "difficulty": self.profile.difficulty.value,
            "strategy": self.profile.strategy.value,
            "personality": self.profile.personality.value,
            "description": self.profile.description,
            "difficulty_notes": self.profile.difficulty_settings.description,
            "strategy_notes": self.profile.strategy_settings.description,
            "personality_notes": self.profile.personality_settings.description,
        }

    def get_performance_summary(self) -> Dict[str, Union[str, float, Dict]]:
        total = len(self.rounds)
        if total == 0:
            return {"status": "no_data", "opponent": self.get_opponent_info()}

        human_wins = sum(1 for r in self.rounds if r.outcome == "win")
        robot_wins = sum(1 for r in self.rounds if r.outcome == "loss")
        ties = total - human_wins - robot_wins
        robot_win_std = float(np.std([1 if r.outcome == "loss" else 0 for r in self.rounds]))

        return {
            "rounds": total,
            "human_win_rate": human_wins / total,
            "robot_win_rate": robot_wins / total,
            "tie_rate": ties / total,
            "robot_win_std": robot_win_std,
            "prediction_accuracy": self.get_accuracy(),
            "opponent": self.get_opponent_info(),
        }

    def get_accuracy(self) -> float:
        if not self.rounds:
            return 0.0
        correct = 0
        for record in self.rounds:
            if IDX_TO_MOVE[int(np.argmax(record.prediction))] == record.human_move:
                correct += 1
        return correct / len(self.rounds)

    def _human_win_rate(self) -> float:
        wins = sum(1 for r in self.rounds if r.outcome == "win")
        return wins / len(self.rounds) if self.rounds else 0.0

    def _robot_win_rate(self) -> float:
        robot_wins = sum(1 for r in self.rounds if r.outcome == "loss")
        return robot_wins / len(self.rounds) if self.rounds else 0.0

    def _recent_human_edge(self) -> float:
        if not self.rounds:
            return 0.0
        window = self.rounds[-6:]
        score = 0
        for record in window:
            if record.outcome == "win":
                score -= 1
            elif record.outcome == "loss":
                score += 1
        return float(score / max(len(window), 1))


# ---------------------------------------------------------------------------
# Singleton helpers for the Flask app
# ---------------------------------------------------------------------------

_ai_system: Optional[RPSAISystem] = None


def get_ai_system() -> RPSAISystem:
    global _ai_system
    if _ai_system is None:
        _ai_system = RPSAISystem()
    return _ai_system


def initialize_ai_system(difficulty: str, strategy: str, personality: str) -> bool:
    return get_ai_system().set_opponent(difficulty, strategy, personality)


# ---------------------------------------------------------------------------
# Optional simulation harness for manual verification
# ---------------------------------------------------------------------------

def _simulate(
    ai_system: RPSAISystem,
    human_profile: str,
    rounds: int = 200,
    seed: int = 1,
) -> Dict[str, Union[str, float, Dict]]:
    rng = np.random.default_rng(seed)
    ai_system.reset_game_state()
    human_patterns = {
        "slow_cycle": ["rock", "rock", "paper", "scissors", "scissors", "paper"],
        "fast_switch": ["rock", "paper", "scissors"],
    }

    for idx in range(rounds):
        _, ai_move, _ = ai_system.predict_next_move()

        if human_profile == "random":
            human_move = rng.choice(MOVES)
        elif human_profile == "adaptive":
            last_ai = ai_system.rounds[-1].ai_move if ai_system.rounds else rng.choice(MOVES)
            human_move = COUNTERS[last_ai]
            if rng.random() < 0.2:
                human_move = rng.choice(MOVES)
        elif human_profile in ("slow_cycle", "fast_switch"):
            pattern = human_patterns[human_profile]
            human_move = pattern[idx % len(pattern)]
        else:
            human_move = rng.choice(MOVES)

        ai_system.update_with_human_move(human_move, ai_move)

    summary = ai_system.get_performance_summary()
    summary["human_profile"] = human_profile
    return summary


if __name__ == "__main__":
    system = RPSAISystem(seed=42)
    system.set_opponent("master", "to_win", "unpredictable")
    for profile in ("random", "slow_cycle", "fast_switch", "adaptive"):
        print(_simulate(system, profile))
