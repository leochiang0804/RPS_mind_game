"""
Agentic autopilot controller for Rock-Paper-Scissors using a LangGraph-style
workflow. The agent consumes a game_context snapshot, lets a lightweight LLM
planner (Qwen 0.5B local stub) decide which analytical tool to invoke, and
produces a recommended human move with rationale.

The real LangGraph package is optional. If it is unavailable in the environment,
the module provides a minimal fallback that honours the same entry-point and
conditional-edge semantics so tests can execute without external dependencies.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from collections import Counter

try:
    from sbc_backend import get_sbc_backend
except Exception:  # pragma: no cover - sbc backend unavailable
    get_sbc_backend = None  # type: ignore

# Import strategic knowledge from our existing systems
try:
    from optimized_strategies import ToWinStrategy, NotToLoseStrategy
    from markov_predictor import MarkovPredictor
    STRATEGIC_MODULES_AVAILABLE = True
    MarkovPredictorClass = MarkovPredictor
except Exception:
    STRATEGIC_MODULES_AVAILABLE = False
    MarkovPredictorClass = None

# ---------------------------------------------------------------------------
# Minimal LangGraph fallback (used when the real package is absent)
# ---------------------------------------------------------------------------

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - exercised when langgraph is unavailable

    START = "__start__"
    END = "__end__"

    class _CompiledGraph:
        def __init__(
            self,
            nodes: Dict[str, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]],
            edges: Dict[str, List[str]],
            conditional_edges: Dict[str, Callable[[Dict[str, Any]], str]],
            entry_point: str,
            default_max_steps: int = 32,
        ) -> None:
            self.nodes = nodes
            self.edges = edges
            self.conditional_edges = conditional_edges
            self.entry_point = entry_point
            self.default_max_steps = default_max_steps

        def invoke(
            self,
            state: Dict[str, Any],
            config: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            max_steps = self.default_max_steps
            if config and "max_steps" in config:
                max_steps = int(config["max_steps"])
            current = self.entry_point
            steps = 0
            while current != END:
                if steps >= max_steps:
                    raise RuntimeError("LangGraph fallback exceeded max_steps")
                steps += 1
                node_fn = self.nodes[current]
                result = node_fn(state)
                if isinstance(result, dict):
                    state.update(result)
                if current in self.conditional_edges:
                    current = self.conditional_edges[current](state)
                    continue
                next_edges = self.edges.get(current, [])
                current = next_edges[0] if next_edges else END
            return state

    class StateGraph:
        def __init__(self, _state_type: Any) -> None:
            self._nodes: Dict[str, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = {}
            self._edges: Dict[str, List[str]] = {}
            self._conditional_edges: Dict[str, Callable[[Dict[str, Any]], str]] = {}
            self._entry_point: Optional[str] = None
            self._max_steps = 32

        def add_node(
            self,
            name: str,
            fn: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
        ) -> None:
            self._nodes[name] = fn

        def add_edge(self, source: str, target: str) -> None:
            self._edges.setdefault(source, []).append(target)

        def add_conditional_edges(
            self, source: str, router: Callable[[Dict[str, Any]], str]
        ) -> None:
            self._conditional_edges[source] = router

        def set_entry_point(self, name: str) -> None:
            self._entry_point = name

        def compile(self) -> _CompiledGraph:
            if self._entry_point is None:
                raise ValueError("Entry point not set for LangGraph fallback")
            return _CompiledGraph(
                self._nodes,
                self._edges,
                self._conditional_edges,
                self._entry_point,
                self._max_steps,
            )


# ---------------------------------------------------------------------------
# Domain utilities
# ---------------------------------------------------------------------------

MOVES: Tuple[str, str, str] = ("rock", "paper", "scissors")
COUNTERS: Dict[str, str] = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
BEATS: Dict[str, str] = {"rock": "scissors", "paper": "rock", "scissors": "paper"}


def _normalise(probabilities: Iterable[float]) -> List[float]:
    probs = [max(float(p), 0.0) for p in probabilities]
    total = sum(probs)
    if total <= 0:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in probs]


def _entropy(probs: Iterable[float]) -> float:
    values = [p for p in probs if p > 0]
    return -sum(p * math.log(p, 2) for p in values)


def _recent_loss_streak(results: Iterable[str]) -> int:
    streak = 0
    for outcome in reversed(list(results)):
        if outcome == "robot":
            streak += 1
        else:
            break
    return streak


def _expected_payoff(human_move: str, robot_distribution: Dict[str, float]) -> float:
    payoff = 0.0
    for robot_move, probability in robot_distribution.items():
        if human_move == robot_move:
            contribution = 0.0
        elif BEATS[human_move] == robot_move:
            contribution = 1.0
        elif BEATS[robot_move] == human_move:
            contribution = -1.0
        else:
            contribution = 0.0
        payoff += probability * contribution
    return payoff


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_strategic_analysis(context: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced strategic analysis using our sophisticated prediction systems"""
    human_moves = context.get("human_moves", [])
    robot_moves = context.get("robot_moves", [])
    results = context.get("results", [])
    
    # Get opponent configuration
    opponent_info = context.get("opponent_info", {})
    ai_difficulty = opponent_info.get("ai_difficulty", "intermediate")
    ai_strategy = opponent_info.get("ai_strategy", "to_win")
    ai_personality = opponent_info.get("ai_personality", "neutral")
    
    insights = {
        "name": "strategic_analysis",
        "confidence": 0.3,
        "recommended_move": "paper",  # Default
        "reasoning": "fallback strategy",
        "pattern_analysis": {},
        "counter_strategy": {}
    }
    
    if len(human_moves) < 2:
        return insights
    
    try:
        # Use our advanced strategic knowledge if available
        if STRATEGIC_MODULES_AVAILABLE and MarkovPredictorClass and len(human_moves) >= 3:
            try:
                # Initialize Markov predictor for pattern detection
                markov = MarkovPredictorClass(order=2, smoothing_factor=0.5)
                
                # Train on human move history
                for move in human_moves:
                    markov.update(move)
                
                # Get Markov prediction of next human move
                prediction_result = markov.predict()
                
                # Handle different return formats from MarkovPredictor
                human_pred = None
                if isinstance(prediction_result, tuple):
                    pred_move = prediction_result[0]
                    if hasattr(pred_move, 'item'):
                        human_pred = str(pred_move.item())
                    else:
                        human_pred = str(pred_move)
                elif isinstance(prediction_result, str):
                    human_pred = prediction_result
                
                # Ensure the prediction is a valid move
                if human_pred and human_pred.lower() in ['rock', 'paper', 'scissors']:
                    human_pred = human_pred.lower()
                    # Counter the predicted human move
                    counter_to_human = COUNTERS[human_pred]
                    
                    # But the robot will likely also try to counter us
                    # So we need to think one step ahead
                    if ai_strategy == "to_win" and ai_difficulty in ["challenger", "master", "grandmaster"]:
                        # Smart robot: assume it will counter our counter
                        robot_likely_move = COUNTERS[counter_to_human]
                        # So we counter its counter
                        our_move = COUNTERS[robot_likely_move]
                        reasoning = f"Level-2 thinking: predict human {human_pred} → robot counters our counter {counter_to_human} with {robot_likely_move} → we play {our_move}"
                    else:
                        # Simple robot: just counter the predicted human move
                        our_move = counter_to_human
                        reasoning = f"Level-1 thinking: predict human {human_pred} → play {our_move} to counter"
                    
                    insights["recommended_move"] = our_move
                    insights["reasoning"] = reasoning
                    insights["confidence"] = min(0.8, 0.5 + len(human_moves) * 0.05)
            except Exception as e:
                # If Markov fails, continue with other analysis
                pass
            
            # Analyze robot patterns if we have robot move history
            if len(robot_moves) >= 3:
                robot_counter = Counter(robot_moves[-8:])  # Recent robot moves
                most_common_robot = robot_counter.most_common(1)[0][0]
                
                # Check if robot has a bias
                total_recent = sum(robot_counter.values())
                robot_bias = robot_counter[most_common_robot] / total_recent
                
                if robot_bias > 0.5:  # Strong bias detected
                    # Counter the robot's bias
                    counter_robot_bias = COUNTERS[most_common_robot]
                    insights["recommended_move"] = counter_robot_bias
                    insights["reasoning"] = f"Robot bias detected: {robot_bias:.1%} {most_common_robot} → counter with {counter_robot_bias}"
                    insights["confidence"] = min(0.9, robot_bias * 1.2)
        
        # Additional pattern analysis using our own logic
        if len(human_moves) >= 4:
            # Detect cycles and repetitions
            recent_4 = human_moves[-4:]
            
            # Check for alternating pattern
            if len(set(recent_4[::2])) == 1 and len(set(recent_4[1::2])) == 1:
                # Strong alternating pattern detected
                last_move = human_moves[-1]
                predicted_next = recent_4[0] if human_moves[-1] == recent_4[1] else recent_4[1]
                counter_move = COUNTERS[predicted_next]
                
                insights["recommended_move"] = counter_move
                insights["reasoning"] = f"Alternating pattern: {recent_4[0]}-{recent_4[1]} detected → predict {predicted_next} → counter with {counter_move}"
                insights["confidence"] = 0.85
                insights["pattern_analysis"]["type"] = "alternating"
                insights["pattern_analysis"]["strength"] = 0.85
            
            # Check for repetition pattern
            elif len(set(recent_4)) == 1:
                # Strong repetition detected
                repeated_move = recent_4[0]
                counter_move = COUNTERS[repeated_move]
                
                insights["recommended_move"] = counter_move  
                insights["reasoning"] = f"Repetition pattern: {repeated_move}×4 detected → counter with {counter_move}"
                insights["confidence"] = 0.9
                insights["pattern_analysis"]["type"] = "repetition"
                insights["pattern_analysis"]["strength"] = 0.9
        
        # Performance-based adjustments
        if len(results) >= 5:
            recent_results = results[-5:]
            human_wins = recent_results.count("human")
            robot_wins = recent_results.count("robot")
            
            if robot_wins >= 4:
                # Robot is dominating - we need to break the pattern
                insights["confidence"] = min(1.0, insights["confidence"] * 1.2)
                insights["reasoning"] += " [high-priority: breaking losing streak]"
            elif human_wins >= 4:
                # We're winning - be more conservative
                insights["confidence"] = insights["confidence"] * 0.9
                insights["reasoning"] += " [conservative: maintaining winning streak]"
    
    except Exception as e:
        # Fallback if strategic modules fail
        insights["reasoning"] = f"strategic analysis failed: {e}"
    
    return insights


def tool_predict_opponent_move(context: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced prediction tool that better utilizes game context"""
    ai_pred = context.get("ai_prediction", {})
    metadata = ai_pred.get("metadata", {}) or context.get("ai_metadata", {})
    move_sel = metadata.get("move_selection", {})
    
    # Get human move history for pattern analysis
    human_moves = context.get("human_moves", [])
    robot_moves = context.get("robot_moves", [])
    results = context.get("results", [])
    
    # Get opponent configuration for strategic analysis
    opponent_info = context.get("opponent_info", {})
    ai_difficulty = opponent_info.get("ai_difficulty", "unknown")
    ai_strategy = opponent_info.get("ai_strategy", "unknown")
    ai_personality = opponent_info.get("ai_personality", "unknown")
    
    # Enhanced human prediction with more history
    human_prediction = ai_pred.get("human_prediction")
    if not human_prediction and len(human_moves) > 0:
        # Calculate frequency distribution with recency weighting
        counts = {move: 1.0 for move in MOVES}  # Base frequency
        
        # Weight recent moves more heavily
        for i, move in enumerate(human_moves[-12:]):
            if move in counts:
                weight = 1.0 + (i / 12.0)  # More recent = higher weight
                counts[move] += weight
        
        total = sum(counts.values())
        human_prediction = [counts[move] / total for move in MOVES]
    
    if not human_prediction:
        human_prediction = [1.0/3, 1.0/3, 1.0/3]  # Uniform if no data
    
    human_prediction = _normalise(human_prediction)

    # Enhanced opponent prediction with difficulty/strategy context
    opponent_dist = move_sel.get("adjusted_distribution")
    if opponent_dist:
        opponent_dist = _normalise(opponent_dist)
    else:
        # IMPROVED: Use actual robot move history to build better predictions
        if len(robot_moves) >= 3:
            # Analyze robot's actual pattern
            robot_counts = {move: 1.0 for move in MOVES}  # Base frequency
            
            # Weight recent robot moves more heavily
            for i, move in enumerate(robot_moves[-8:]):
                if move in robot_counts:
                    weight = 1.0 + (i / 8.0)  # More recent = higher weight
                    robot_counts[move] += weight
            
            # Analyze robot's response patterns to human moves
            if len(human_moves) == len(robot_moves):
                for i in range(min(6, len(human_moves))):
                    h_move = human_moves[-(i+1)]
                    r_move = robot_moves[-(i+1)]
                    
                    # Boost probability if robot shows counter-patterns
                    if COUNTERS[h_move] == r_move:
                        robot_counts[r_move] += 0.5  # Robot likes to counter
                    elif h_move == r_move:
                        robot_counts[r_move] += 0.3  # Robot sometimes mirrors
            
            total = sum(robot_counts.values())
            opponent_dist = [robot_counts[move] / total for move in MOVES]
        else:
            # Fallback: Strategic counter-prediction based on opponent type
            derived = [0.0, 0.0, 0.0]
            
            # Base counter-strategy
            for idx, move in enumerate(MOVES):
                counter = COUNTERS[move]
                counter_idx = MOVES.index(counter)
                derived[counter_idx] += human_prediction[idx]
            
            opponent_dist = _normalise(derived)
        # Adjust based on opponent personality and strategy
        if ai_strategy == "to_win" and ai_difficulty in ["master", "grandmaster"]:
            # Aggressive opponents might use more complex counter-strategies
            # Add some unpredictability
            for i in range(len(opponent_dist)):
                opponent_dist[i] = opponent_dist[i] * 0.8 + 0.2 * (1.0/3)
        elif ai_strategy == "not_to_lose":
            # Defensive opponents might be more predictable
            # Strengthen the counter pattern
            max_idx = max(range(len(opponent_dist)), key=opponent_dist.__getitem__)
            for i in range(len(opponent_dist)):
                if i == max_idx:
                    opponent_dist[i] = min(0.7, opponent_dist[i] * 1.2)
                else:
                    opponent_dist[i] = opponent_dist[i] * 0.9
        
        opponent_dist = _normalise(opponent_dist)

    # Enhanced confidence calculation based on distribution strength
    top_idx = max(range(len(opponent_dist)), key=opponent_dist.__getitem__)
    top_move = MOVES[top_idx]
    
    # Calculate base confidence from distribution strength
    max_prob = max(opponent_dist)
    min_prob = min(opponent_dist)
    distribution_strength = max_prob - (1.0 / len(MOVES))  # How much above uniform
    
    # Base confidence from AI prediction or distribution strength
    ai_confidence = float(ai_pred.get("confidence", 0.33))
    distribution_confidence = min(0.95, max(0.15, distribution_strength * 2.0))  # Scale to 0.15-0.95
    
    # Use the higher of the two confidence sources
    base_confidence = max(ai_confidence, distribution_confidence)
    
    # Boost confidence based on data quality and game state
    confidence_factors = []
    
    # Factor 1: Amount of data
    data_rounds = len(human_moves)
    if data_rounds >= 10:
        confidence_factors.append(1.0)
    elif data_rounds >= 5:
        confidence_factors.append(0.8)
    else:
        confidence_factors.append(0.6)
    
    # Factor 2: Pattern strength in recent moves
    if len(human_moves) >= 3:
        recent_3 = human_moves[-3:]
        if len(set(recent_3)) == 1:  # All same
            confidence_factors.append(1.2)
        elif len(set(recent_3)) == 2:  # Pattern emerging
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.8)
    
    # Factor 3: Performance trend
    if len(results) >= 5:
        recent_results = results[-5:]
        robot_recent_wins = recent_results.count("robot")
        if robot_recent_wins >= 4:  # Robot is winning
            confidence_factors.append(1.1)  # High confidence in pattern
        elif robot_recent_wins <= 1:  # Human is winning  
            confidence_factors.append(0.9)  # Medium confidence
        else:
            confidence_factors.append(1.0)
    
    # Calculate weighted confidence
    if confidence_factors:
        confidence_multiplier = sum(confidence_factors) / len(confidence_factors)
        final_confidence = base_confidence * confidence_multiplier
    else:
        final_confidence = base_confidence
    
    # Improved confidence bounds - don't artificially cap strong predictions
    final_confidence = min(0.95, max(0.15, final_confidence))

    return {
        "name": "predict",
        "opponent_distribution": {move: opponent_dist[i] for i, move in enumerate(MOVES)},
        "human_distribution": {move: human_prediction[i] for i, move in enumerate(MOVES)},
        "confidence": float(final_confidence),
        "top_robot_move": top_move,
        "signals": {
            "pattern_strength": metadata.get("human_model", {}).get("pattern_strength"),
            "recent_repetition": metadata.get("human_model", {}).get("recent_repetition"),
            "change_factor": metadata.get("human_model", {}).get("change_factor"),
            "data_rounds": data_rounds,
            "opponent_type": f"{ai_difficulty}_{ai_strategy}_{ai_personality}",
        },
    }


def tool_detect_behavior_shift(context: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced behavior shift detection with better confidence scoring"""
    human_moves = context.get("human_moves", [])
    results = context.get("results", [])
    
    if len(human_moves) < 3:
        return {
            "name": "analytics",
            "drift": False,
            "confidence": 0.0,
            "label": "insufficient_data",
            "features": {},
            "action_suggestion": "gather_more_data",
        }

    # Enhanced pattern analysis with variable window sizes
    if len(human_moves) >= 6:
        # Compare recent vs previous patterns
        recent = human_moves[-3:]
        previous = human_moves[-6:-3]
        freq_recent = {move: recent.count(move) / len(recent) for move in MOVES}
        freq_previous = {move: previous.count(move) / len(previous) for move in MOVES}
        divergence = 0.5 * sum(abs(freq_recent[m] - freq_previous[m]) for m in MOVES)
    else:
        # Fall back to simple analysis for shorter histories
        recent = human_moves[-2:] if len(human_moves) >= 2 else human_moves
        previous = human_moves[:-2] if len(human_moves) >= 4 else []
        
        if previous:
            freq_recent = {move: recent.count(move) / len(recent) for move in MOVES}
            freq_previous = {move: previous.count(move) / len(previous) for move in MOVES}
            divergence = 0.5 * sum(abs(freq_recent[m] - freq_previous[m]) for m in MOVES)
        else:
            freq_recent = {move: recent.count(move) / len(recent) for move in MOVES}
            freq_previous = {move: 0.33 for move in MOVES}  # Assume uniform baseline
            divergence = 0.5 * sum(abs(freq_recent[m] - freq_previous[m]) for m in MOVES)

    # Enhanced loss streak analysis
    loss_streak = _recent_loss_streak(results)
    
    # Detect repetition patterns
    repetition_score = 0.0
    if len(human_moves) >= 3:
        last_3 = human_moves[-3:]
        if len(set(last_3)) == 1:  # All same move
            repetition_score = 1.0
        elif len(set(last_3)) == 2:  # Two different moves
            repetition_score = 0.5
    
    # Detect oscillation patterns (rock-paper-rock-paper...)
    oscillation_score = 0.0
    if len(human_moves) >= 4:
        last_4 = human_moves[-4:]
        if last_4[0] == last_4[2] and last_4[1] == last_4[3] and last_4[0] != last_4[1]:
            oscillation_score = 1.0
    
    # Performance pressure analysis
    performance_pressure = 0.0
    if len(results) >= 5:
        recent_results = results[-5:]
        human_wins = recent_results.count("human")
        performance_pressure = max(0.0, (3 - human_wins) / 3.0)  # Higher when losing more
    
    # Determine drift and confidence
    drift_indicators = []
    
    # Divergence-based drift
    if divergence > 0.4:
        drift_indicators.append(("major_shift", 0.9))
    elif divergence > 0.25:
        drift_indicators.append(("pattern_shift", 0.7))
    
    # Loss streak drift
    if loss_streak >= 4:
        drift_indicators.append(("pressure_shift", 0.8))
    elif loss_streak >= 3:
        drift_indicators.append(("losing_streak", 0.6))
    
    # Repetition drift
    if repetition_score >= 0.5:
        drift_indicators.append(("repetition_pattern", 0.5))
    
    # Oscillation drift  
    if oscillation_score >= 1.0:
        drift_indicators.append(("oscillation_pattern", 0.6))
    
    # Calculate overall drift and confidence
    if drift_indicators:
        drift = True
        # Use the strongest indicator for confidence
        best_indicator = max(drift_indicators, key=lambda x: x[1])
        label = best_indicator[0] 
        confidence = best_indicator[1]
        
        # Adjust confidence based on data quality
        data_quality = min(1.0, len(human_moves) / 10.0)
        confidence = confidence * (0.5 + 0.5 * data_quality)
    else:
        drift = False
        label = "stable"
        confidence = min(0.8, 0.3 + 0.5 * min(1.0, len(human_moves) / 10.0))

    # Determine action suggestion
    if loss_streak >= 4:
        action = "major_strategy_change"
    elif loss_streak >= 3:
        action = "increase_exploration"
    elif drift and label in ["major_shift", "pattern_shift"]:
        action = "refresh_prediction"
    elif repetition_score >= 0.5:
        action = "counter_repetition"
    elif oscillation_score >= 1.0:
        action = "break_oscillation"
    else:
        action = "maintain_strategy"

    return {
        "name": "analytics",
        "drift": drift,
        "confidence": float(confidence),
        "label": label,
        "features": {
            "freq_recent": freq_recent,
            "freq_previous": freq_previous,
            "divergence": divergence,
            "loss_streak": loss_streak,
            "repetition_score": repetition_score,
            "oscillation_score": oscillation_score,
            "performance_pressure": performance_pressure,
            "data_rounds": len(human_moves),
        },
        "action_suggestion": action,
    }


def tool_choose_human_move(
    opponent_distribution: Dict[str, float],
    objective: str,
) -> Dict[str, Any]:
    utilities: Dict[str, float] = {}
    tie_bonus = 0.15
    top_robot_move = max(opponent_distribution.items(), key=lambda kv: kv[1])[0]
    for move in MOVES:
        payoff = _expected_payoff(move, opponent_distribution)
        not_loss = 1.0 - sum(
            prob for r_move, prob in opponent_distribution.items() if BEATS[r_move] == move
        )
        win_plus_tie = (
            sum(prob for r_move, prob in opponent_distribution.items() if BEATS[move] == r_move)
            + tie_bonus
            * sum(prob for r_move, prob in opponent_distribution.items() if r_move == move)
        )
        if objective == "maximize_win":
            utilities[move] = payoff
        elif objective == "minimize_loss":
            utilities[move] = not_loss
        elif objective == "maximize_win_plus_tie":
            utilities[move] = win_plus_tie
        elif objective == "robust":
            shrink = {
                r_move: max(prob - 0.08, 0.0) for r_move, prob in opponent_distribution.items()
            }
            shrink = _normalise(shrink.values())
            shrink_dist = {m: shrink[i] for i, m in enumerate(MOVES)}
            utilities[move] = min(
                _expected_payoff(move, opponent_distribution),
                _expected_payoff(move, shrink_dist),
            )
        else:
            utilities[move] = payoff

    ranked = sorted(utilities.items(), key=lambda kv: kv[1], reverse=True)
    best_move, best_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else best_score
    tie_breaker: Optional[str] = None

    if len(ranked) > 1 and abs(best_score - runner_up_score) <= 1e-6:
        best_move = COUNTERS[top_robot_move]
        best_score = _expected_payoff(best_move, opponent_distribution)
        tie_breaker = "counter_top_robot"

    if best_move == top_robot_move:
        best_move = COUNTERS[top_robot_move]
        best_score = _expected_payoff(best_move, opponent_distribution)
        tie_breaker = tie_breaker or "avoid_mirroring"

    other_scores = [score for move, score in utilities.items() if move != best_move]
    runner_up_after = max(other_scores) if other_scores else best_score
    confidence = max(0.0, min(1.0, abs(best_score - runner_up_after)))

    return {
        "name": "policy",
        "objective": objective,
        "candidate_move": best_move,
        "utilities": utilities,
        "confidence": confidence,
        "best_score": best_score,
        "tie_breaker": tie_breaker,
        "top_robot_move": top_robot_move,
    }


def tool_wildcard_override(state: Dict[str, Any]) -> Dict[str, Any]:
    predict_output = state["tool_outputs"].get("predict")
    policy_output = state["tool_outputs"].get("policy")
    if not predict_output or not policy_output:
        # Improved fallback: if we have prediction, counter it; otherwise be smart about fallback
        if predict_output:
            top_robot = predict_output["top_robot_move"]
            fallback_move = COUNTERS[top_robot]  # Always counter predicted move
            reason = f"counter predicted {top_robot}"
        else:
            # If no prediction at all, use a balanced approach
            fallback_move = policy_output["candidate_move"] if policy_output else "paper"
            reason = "balanced fallback"
        
        return {
            "name": "wildcard",
            "override_move": fallback_move,
            "reason": reason,
            "valid_for_rounds": 1,
        }

    top_robot = predict_output["top_robot_move"]
    primary_move = policy_output["candidate_move"]
    # Level-k+1 counter: assume robot counters our primary move, so beat that counter.
    expected_robot_adjustment = COUNTERS[primary_move]
    wildcard_move = COUNTERS[expected_robot_adjustment]

    if wildcard_move == primary_move:
        wildcard_move = COUNTERS[top_robot]

    return {
        "name": "wildcard",
        "override_move": wildcard_move,
        "reason": f"anticipating robot counter to {primary_move}",
        "valid_for_rounds": 1,
    }


def tool_risk_controller(
    candidate_move: str,
    drift: bool,
    confidence: float,
) -> Dict[str, Any]:
    base_distribution = {move: 0.05 for move in MOVES}
    emphasis = 0.85
    if drift:
        emphasis = max(0.55, emphasis - 0.2)
    if confidence < 0.15:
        emphasis = max(0.5, emphasis - 0.25)
    residue = 1.0 - emphasis
    share = residue / (len(MOVES) - 1)
    for move in MOVES:
        base_distribution[move] = share
    base_distribution[candidate_move] = emphasis
    base_distribution = {move: round(prob, 6) for move, prob in base_distribution.items()}
    return {
        "name": "risk_controller",
        "final_move": candidate_move,
        "distribution": base_distribution,
        "params": {
            "emphasis": emphasis,
            "drift": drift,
            "model_confidence": confidence,
        },
    }


def tool_coaching_message(state: Dict[str, Any]) -> str:
    predict_output = state["tool_outputs"].get("predict", {})
    policy_output = state["tool_outputs"].get("policy", {})
    wildcard_output = state["tool_outputs"].get("wildcard")
    final_move = state.get("final_move")
    top_robot = predict_output.get("top_robot_move", "unknown")
    robot_prob = predict_output.get("opponent_distribution", {}).get(top_robot, 0.0)
    objective = policy_output.get("objective", "maximize_win")
    objective_label = objective.replace("_", " ")
    policy_conf = policy_output.get("confidence", 0.0)
    
    # Create better distribution analysis
    distribution_note: str
    if robot_prob >= 0.55:
        distribution_note = "Strong pattern detected"
    elif robot_prob >= 0.4:
        distribution_note = "Moderate pattern detected"
    else:
        distribution_note = "No dominant pattern"
    
    counter_move = COUNTERS.get(top_robot)
    if wildcard_output and wildcard_output.get("override_move") == final_move:
        rationale = wildcard_output.get("reason", "wildcard override active")
    elif final_move == counter_move:
        rationale = f"counter {top_robot}"
    elif final_move == top_robot:
        rationale = "mirror for tie control"
    else:
        rationale = f"press edge with {final_move}"
    
    confidence_pct = int(max(0.0, min(1.0, policy_conf)) * 100)
    if wildcard_output and wildcard_output.get("override_move") == final_move:
        rationale = f"{rationale} (wildcard)"
    
    final_move_label = (final_move or "--").upper()
    
    # Improved coaching message format without misleading percentages
    return (
        f"{distribution_note} - predicted {top_robot.upper()}. "
        f"Play {final_move_label} to {rationale} [{objective_label}, {confidence_pct}% margin]"
    )


# ---------------------------------------------------------------------------
# Lightweight LLM planner (rule-driven stand-in for Qwen 0.5B)
# ---------------------------------------------------------------------------

class QwenMiniPlanner:
    """Heuristic planner that mimics a compact LLM deciding which tool to call."""

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["iterations"] = state.get("iterations", 0) + 1
        if state["iterations"] >= self.max_turns:
            return {
                "next_action": "finalize",
                "objective": state["meta"].get("objective", "maximize_win"),
                "reason": "safety_stop",
            }

        tool_outputs = state["tool_outputs"]
        context = state["context"]
        loss_streak = _recent_loss_streak(context.get("results", []))

        if "predict" not in tool_outputs:
            return {
                "next_action": "call_tool_predict",
                "objective": "maximize_win",
                "reason": "need_opponent_distribution",
            }

        # NEW: Strategic analysis with advanced pattern detection
        if ("strategic_analysis" not in tool_outputs and 
            len(context.get("human_moves", [])) >= 2 and
            tool_outputs.get("predict", {}).get("confidence", 0) < 0.6):
            return {
                "next_action": "call_tool_strategic",
                "objective": "maximize_win_plus_tie", 
                "reason": "apply_advanced_strategic_analysis",
            }

        predict_conf = tool_outputs["predict"]["confidence"]
        if predict_conf < 0.55 and "analytics" not in tool_outputs:
            return {
                "next_action": "call_tool_analytics",
                "objective": "maximize_win",
                "reason": "low_confidence_requesting_drift_analysis",
            }

        drift_flag = tool_outputs.get("analytics", {}).get("drift", False)
        objective = "robust" if drift_flag else "maximize_win"
        state["meta"]["objective"] = objective

        if "policy" not in tool_outputs:
            return {
                "next_action": "call_tool_policy",
                "objective": objective,
                "reason": "need_candidate_move",
            }

        policy_conf = tool_outputs["policy"].get("confidence", 0.0)
        if (
            policy_conf < 0.12
            and "wildcard" not in tool_outputs
            and loss_streak >= 2
        ):
            return {
                "next_action": "call_tool_wildcard",
                "objective": objective,
                "reason": "low_margin_and_losses_trigger_wildcard",
            }

        return {
            "next_action": "finalize",
            "objective": objective,
            "reason": "sufficient_confidence_to_commit",
        }


class QwenLLMPlanner:
    """Planner that delegates decisions to the same LLM backend used for coaching."""

    def __init__(self, fallback: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None) -> None:
        self.backend = get_sbc_backend() if callable(get_sbc_backend) else None
        self.model_adapter = getattr(self.backend, "model_adapter", None)
        self.personality = "professor"
        self.fallback = fallback or QwenMiniPlanner()

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Safety check to prevent infinite loops
        state["iterations"] = state.get("iterations", 0) + 1
        if state["iterations"] >= 8:  # Maximum iterations to prevent recursion
            return {
                "next_action": "finalize",
                "objective": state["meta"].get("objective", "maximize_win"),
                "reason": "safety_stop_max_iterations",
            }

        if self.model_adapter is None:
            return self.fallback(state)

        prompt = self._build_prompt(state)
        try:
            response = self.model_adapter.generate_response(prompt, self.personality)
            decision = self._parse_response(response)
            if not decision:
                raise ValueError("LLM response missing required keys")
            
            # Additional safety check - if we have minimum tools, prefer finalize
            tool_outputs = state.get("tool_outputs", {})
            if (decision["next_action"] != "finalize" and 
                "predict" in tool_outputs and "policy" in tool_outputs and
                state["iterations"] >= 5):
                return {
                    "next_action": "finalize",
                    "objective": decision.get("objective", "maximize_win"),
                    "reason": "safety_override_sufficient_tools",
                }
            
            return decision
        except Exception:
            return self.fallback(state)

    def _build_prompt(self, state: Dict[str, Any]) -> str:
        context = state.get("context", {})
        tool_outputs = state.get("tool_outputs", {})
        objective = state.get("meta", {}).get("objective", "maximize_win")

        # Extract key game context information
        round_num = context.get("round") or len(context.get("human_moves", []))
        human_moves = context.get("human_moves", [])
        robot_moves = context.get("robot_moves", [])
        results = context.get("results", [])
        
        # Calculate game statistics for better decision making
        total_rounds = len(results)
        human_wins = results.count("human") if results else 0
        robot_wins = results.count("robot") if results else 0
        ties = results.count("tie") if results else 0
        
        human_win_rate = (human_wins / total_rounds * 100) if total_rounds > 0 else 0
        robot_win_rate = (robot_wins / total_rounds * 100) if total_rounds > 0 else 0
        
        loss_streak = _recent_loss_streak(results)
        
        # Analyze recent patterns
        recent_human = human_moves[-5:] if len(human_moves) >= 5 else human_moves
        recent_robot = robot_moves[-5:] if len(robot_moves) >= 5 else robot_moves
        
        # Get opponent info for strategic context
        opponent_info = context.get("opponent_info", {})
        ai_difficulty = opponent_info.get("ai_difficulty", "unknown")
        ai_strategy = opponent_info.get("ai_strategy", "unknown") 
        ai_personality = opponent_info.get("ai_personality", "unknown")

        # Tool output summaries with more detail
        predict_summary = "missing"
        predict_confidence = 0.0
        if "predict" in tool_outputs:
            pred = tool_outputs["predict"]
            dist = pred.get("opponent_distribution", {})
            dist_text = ", ".join(f"{move}:{prob:.0%}" for move, prob in dist.items())
            predict_confidence = pred.get("confidence", 0.0)
            predict_summary = (
                f"top_robot={pred.get('top_robot_move')} "
                f"confidence={predict_confidence:.2f} dist=({dist_text})"
            )
        
        analytics_summary = "missing"
        analytics_confidence = 0.0
        if "analytics" in tool_outputs:
            an = tool_outputs["analytics"]
            analytics_confidence = an.get("confidence", 0.0)
            analytics_summary = (
                f"drift={an.get('drift')} confidence={analytics_confidence:.2f} "
                f"label={an.get('label')} action={an.get('action_suggestion')}"
            )
        
        policy_summary = "missing"
        policy_confidence = 0.0
        if "policy" in tool_outputs:
            pol = tool_outputs["policy"]
            policy_confidence = pol.get("confidence", 0.0)
            utilities = ", ".join(f"{m}:{v:.2f}" for m, v in pol.get("utilities", {}).items())
            policy_summary = (
                f"move={pol.get('candidate_move')} gap={policy_confidence:.2f} "
                f"utilities=({utilities}) objective={pol.get('objective')}"
            )

        # Decision logic guidelines
        decision_guidance = ""
        if round_num <= 2:
            decision_guidance = "Early game: Focus on prediction to establish patterns."
        elif loss_streak >= 3:
            decision_guidance = "Long loss streak: Consider analytics for strategy shift, then policy."
        elif predict_confidence < 0.4 and "predict" in tool_outputs:
            decision_guidance = "Low prediction confidence: Try analytics for behavior analysis."
        elif "predict" in tool_outputs and "analytics" not in tool_outputs:
            decision_guidance = "Have prediction: Get analytics to detect patterns/drift."
        elif "predict" in tool_outputs and "analytics" in tool_outputs and "policy" not in tool_outputs:
            decision_guidance = "Have prediction + analytics: Get policy for move selection."
        elif all(key in tool_outputs for key in ["predict", "analytics", "policy"]):
            if min(predict_confidence, analytics_confidence, policy_confidence) < 0.3:
                decision_guidance = "All tools complete but low confidence: Consider wildcard or finalize with caution."
            else:
                decision_guidance = "All tools complete with reasonable confidence: Ready to finalize."

        prompt = f"""
You are an intelligent decision planner for a Rock-Paper-Scissors autopilot agent. Your goal is to make strategic decisions 
that lead to the best possible human move against a {ai_difficulty} {ai_personality} robot using {ai_strategy} strategy.

Game Context:
- Round: {round_num} | Win Rate: Human {human_win_rate:.1f}% vs Robot {robot_win_rate:.1f}%
- Recent loss streak: {loss_streak} | Total games: {total_rounds}
- Recent human moves: {recent_human}
- Recent robot moves: {recent_robot}
- Opponent: {ai_difficulty} difficulty, {ai_strategy} strategy, {ai_personality} personality

Current Analysis:
- Tools completed: {list(tool_outputs.keys())}
- Prediction: {predict_summary}
- Analytics: {analytics_summary} 
- Policy: {policy_summary}
- Current objective: {objective}

Strategic Guidance: {decision_guidance}

Available Actions:
- call_tool_predict: Analyze opponent patterns and predict next robot move
- call_tool_analytics: Detect behavioral shifts and recommend strategy changes  
- call_tool_policy: Choose optimal human move based on prediction and objective
- call_tool_wildcard: Apply counter-prediction strategies for advanced play
- finalize: Lock in final move decision (requires predict + policy at minimum)

Available Objectives:
- maximize_win: Focus purely on winning (aggressive)
- minimize_loss: Avoid losses (defensive) 
- maximize_win_plus_tie: Win or tie acceptable (balanced)
- robust: Conservative play against uncertain predictions

Choose your next action strategically. Consider the opponent type, current confidence levels, and missing analysis.
Respond with valid JSON only:

{{"next_action": "<action>", "objective": "<objective>", "reason": "<strategic reasoning>"}}
"""
        return prompt.strip()

    @staticmethod
    def _parse_response(response_text: str) -> Optional[Dict[str, Any]]:
        if not response_text:
            return None
        json_start = response_text.find("{")
        json_end = response_text.rfind("}")
        if json_start == -1 or json_end == -1 or json_end <= json_start:
            return None
        try:
            payload = json.loads(response_text[json_start : json_end + 1])
        except json.JSONDecodeError:
            return None

        allowed_actions = {
            "call_tool_predict",
            "call_tool_analytics",
            "call_tool_policy",
            "call_tool_wildcard",
            "finalize",
        }
        allowed_objectives = {
            "maximize_win",
            "minimize_loss",
            "maximize_win_plus_tie",
            "robust",
        }

        next_action = payload.get("next_action")
        objective = payload.get("objective")
        reason = payload.get("reason")

        if next_action not in allowed_actions:
            return None
        if objective not in allowed_objectives:
            objective = "maximize_win"
        if not isinstance(reason, str):
            reason = "LLM-decision"

        return {
            "next_action": next_action,
            "objective": objective,
            "reason": reason,
        }


# ---------------------------------------------------------------------------
# Agent state / result containers
# ---------------------------------------------------------------------------

@dataclass
class AutopilotResult:
    final_move: str
    predicted_robot_move: str
    log: List[str]
    tool_outputs: Dict[str, Dict[str, Any]]
    context: Dict[str, Any]
    events: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Autopilot agent orchestrated by LangGraph
# ---------------------------------------------------------------------------

class AutopilotAgent:
    def __init__(
        self,
        llm_planner: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        max_graph_steps: int = 24,
    ) -> None:
        self.llm_planner = llm_planner or QwenLLMPlanner()
        self.max_graph_steps = max_graph_steps
        self.graph = self._build_graph()

    # Graph construction -----------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(dict)
        graph.set_entry_point("planner")
        graph.add_node("planner", self._node_planner)
        graph.add_node("call_tool_predict", self._node_tool_predict)
        graph.add_node("call_tool_analytics", self._node_tool_analytics)
        graph.add_node("call_tool_strategic", self._node_tool_strategic)
        graph.add_node("call_tool_policy", self._node_tool_policy)
        graph.add_node("call_tool_wildcard", self._node_tool_wildcard)
        graph.add_node("finalize", self._node_finalize)

        graph.add_conditional_edges("planner", self._planner_router)
        graph.add_edge("call_tool_predict", "planner")
        graph.add_edge("call_tool_analytics", "planner")
        graph.add_edge("call_tool_strategic", "planner")
        graph.add_edge("call_tool_policy", "planner")
        graph.add_edge("call_tool_wildcard", "planner")
        graph.add_edge("finalize", END)

        return graph.compile()

    def _record_event(
        self,
        state: Dict[str, Any],
        stage: str,
        status: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        events = state.setdefault("events", [])
        entry: Dict[str, Any] = {
            "stage": stage,
            "status": status,
        }
        start_time = state.get("start_time")
        if isinstance(start_time, float):
            entry["time_ms"] = int((time.perf_counter() - start_time) * 1000)
        if detail is not None:
            entry["detail"] = detail
        events.append(entry)

    # Node implementations ---------------------------------------------------
    def _node_planner(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._record_event(state, "planner", "entered")
        decision = self.llm_planner(state)
        state["last_decision"] = decision
        state["meta"]["objective"] = decision.get(
            "objective", state["meta"].get("objective", "maximize_win")
        )
        message = (
            f"[Stage planner] decision={decision['next_action']} "
            f"objective={state['meta']['objective']} because {decision['reason']}"
        )
        state["log"].append(message)
        detail = {
            "next_action": decision.get("next_action"),
            "objective": state["meta"].get("objective"),
            "reason": decision.get("reason"),
        }
        self._record_event(state, "planner", "completed", detail)
        return state

    def _planner_router(self, state: Dict[str, Any]) -> str:
        action = state.get("last_decision", {}).get("next_action", "finalize")
        mapping = {
            "call_tool_predict": "call_tool_predict",
            "call_tool_analytics": "call_tool_analytics",
            "call_tool_strategic": "call_tool_strategic",
            "call_tool_policy": "call_tool_policy",
            "call_tool_wildcard": "call_tool_wildcard",
            "finalize": "finalize",
        }
        return mapping.get(action, "finalize")

    def _node_tool_predict(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._record_event(state, "tool_predict", "entered")
        output = tool_predict_opponent_move(state["context"])
        state["tool_outputs"]["predict"] = output
        state["predicted_robot_move"] = output["top_robot_move"]
        distribution = ", ".join(
            f"{move}:{prob:.0%}" for move, prob in output["opponent_distribution"].items()
        )
        state["log"].append(
            f"[Stage tool_predict] top_robot={output['top_robot_move']} "
            f"confidence={output['confidence']:.2f} dist=({distribution})"
        )
        self._record_event(
            state,
            "tool_predict",
            "completed",
            {
                "top_robot": output["top_robot_move"],
                "confidence": round(output.get("confidence", 0.0), 3),
            },
        )
        return state

    def _node_tool_analytics(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._record_event(state, "tool_analytics", "entered")
        output = tool_detect_behavior_shift(state["context"])
        state["tool_outputs"]["analytics"] = output
        state["log"].append(
            f"[Stage tool_analytics] drift={output['drift']} "
            f"confidence={output['confidence']:.2f} label={output['label']}"
        )
        self._record_event(
            state,
            "tool_analytics",
            "completed",
            {
                "drift": output.get("drift"),
                "label": output.get("label"),
                "confidence": round(output.get("confidence", 0.0), 3),
            },
        )
        return state

    def _node_tool_strategic(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._record_event(state, "tool_strategic", "entered")
        output = tool_strategic_analysis(state["context"])
        state["tool_outputs"]["strategic_analysis"] = output
        state["log"].append(
            f"[Stage tool_strategic] move={output['recommended_move']} "
            f"confidence={output['confidence']:.2f} reason={output['reasoning'][:50]}..."
        )
        self._record_event(
            state,
            "tool_strategic",
            "completed",
            {
                "recommended_move": output.get("recommended_move"),
                "confidence": round(output.get("confidence", 0.0), 3),
                "reasoning": output.get("reasoning", "")[:100]
            },
        )
        return state

    def _node_tool_policy(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._record_event(state, "tool_policy", "entered")
        predict_output = state["tool_outputs"].get("predict")
        if not predict_output:
            # Emergency prediction if missing
            predict_output = tool_predict_opponent_move(state["context"])
            state["tool_outputs"]["predict"] = predict_output
            state["predicted_robot_move"] = predict_output["top_robot_move"]
            state["log"].append("[Emergency] Generated prediction in policy")
            
        objective = state["meta"].get("objective", "maximize_win")
        output = tool_choose_human_move(
            opponent_distribution=predict_output["opponent_distribution"],
            objective=objective
        )
        state["tool_outputs"]["policy"] = output
        utilities = ", ".join(
            f"{move}:{score:.2f}" for move, score in output["utilities"].items()
        )
        state["log"].append(
            f"[Stage tool_policy] candidate={output['candidate_move']} "
            f"confidence_gap={output['confidence']:.2f} utilities=({utilities})"
        )
        self._record_event(
            state,
            "tool_policy",
            "completed",
            {
                "candidate": output.get("candidate_move"),
                "objective": output.get("objective"),
                "tie_breaker": output.get("tie_breaker"),
                "top_robot_move": output.get("top_robot_move"),
                "confidence": round(output.get("confidence", 0.0), 3),
            },
        )
        return state

    def _node_tool_wildcard(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._record_event(state, "tool_wildcard", "entered")
        output = tool_wildcard_override(state)
        state["tool_outputs"]["wildcard"] = output
        state["log"].append(
            f"[Stage tool_wildcard] override={output['override_move']} "
            f"reason={output['reason']}"
        )
        self._record_event(
            state,
            "tool_wildcard",
            "completed",
            {
                "override": output.get("override_move"),
                "reason": output.get("reason"),
            },
        )
        return state

    def _node_finalize(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._record_event(state, "finalize", "entered")
        
        # Ensure we have minimum required tools
        predict_output = state["tool_outputs"].get("predict")
        policy_output = state["tool_outputs"].get("policy")
        
        # If missing critical tools, run them with fallback logic
        if not predict_output:
            predict_output = tool_predict_opponent_move(state["context"])
            state["tool_outputs"]["predict"] = predict_output
            state["predicted_robot_move"] = predict_output["top_robot_move"]
            state["log"].append("[Emergency] Generated prediction in finalize")
        
        if not policy_output:
            policy_output = tool_choose_human_move(
                opponent_distribution=predict_output["opponent_distribution"],
                objective=state["meta"].get("objective", "maximize_win")
            )
            state["tool_outputs"]["policy"] = policy_output
            state["log"].append("[Emergency] Generated policy in finalize")

        candidate_move = policy_output["candidate_move"]
        analytics = state["tool_outputs"].get("analytics", {})
        strategic_analysis = state["tool_outputs"].get("strategic_analysis", {})
        wildcard_output = state["tool_outputs"].get("wildcard")
        
        # Prioritize strategic analysis if it has high confidence
        if (strategic_analysis and 
            strategic_analysis.get("confidence", 0) > 0.7 and
            strategic_analysis.get("confidence", 0) > policy_output.get("confidence", 0)):
            candidate_move = strategic_analysis["recommended_move"]
            state["log"].append(f"[Strategic Override] Using strategic move: {candidate_move} ({strategic_analysis['reasoning'][:60]}...)")
        
        if wildcard_output:
            candidate_move = wildcard_output.get("override_move", candidate_move)

        # CRITICAL FIX: Safety check to ensure we never mirror the predicted robot move
        # when trying to maximize wins - always counter it
        predicted_robot = predict_output["top_robot_move"]
        if (candidate_move == predicted_robot and 
            state["meta"].get("objective", "maximize_win") in ["maximize_win", "maximize_win_plus_tie"]):
            candidate_move = COUNTERS[predicted_robot]
            state["log"].append(f"[Safety] Corrected {predicted_robot} mirror → {candidate_move} counter")

        risk_output = tool_risk_controller(
            candidate_move,
            analytics.get("drift", False),
            policy_output.get("confidence", 0.0),
        )
        state["tool_outputs"]["risk_controller"] = risk_output
        state["final_move"] = risk_output["final_move"]
        state["predicted_robot_move"] = predict_output["top_robot_move"]

        message = (
            f"[Stage finalize] predicted_robot={state['predicted_robot_move']} "
            f"→ final_move={state['final_move']}"
        )
        state["log"].append(message)
        coaching = tool_coaching_message(state)
        state["log"].append(f"[Stage coaching] {coaching}")
        self._record_event(
            state,
            "finalize",
            "completed",
            {
                "final_move": state.get("final_move"),
                "predicted_robot": state.get("predicted_robot_move"),
                "distribution": risk_output.get("distribution"),
            },
        )
        return state

    # Public API -------------------------------------------------------------
    def run(self, context: Dict[str, Any], verbose: bool = False) -> AutopilotResult:
        initial_state = {
            "context": context,
            "tool_outputs": {},
            "meta": {},
            "log": [],
            "iterations": 0,
            "events": [],
            "start_time": time.perf_counter(),
        }
        self._record_event(initial_state, "agent", "started")
        final_state = self.graph.invoke(
            initial_state,
            config={"max_steps": self.max_graph_steps},
        )
        self._record_event(
            final_state,
            "agent",
            "completed",
            {
                "final_move": final_state.get("final_move"),
                "predicted_robot_move": final_state.get("predicted_robot_move"),
            },
        )
        result = AutopilotResult(
            final_move=final_state.get("final_move") or "",
            predicted_robot_move=final_state.get("predicted_robot_move") or "",
            log=list(final_state.get("log", [])),
            tool_outputs=dict(final_state.get("tool_outputs", {})),
            context=context,
            events=list(final_state.get("events", [])),
        )
        if verbose:
            for line in result.log:
                print(line)
        return result


__all__ = ["AutopilotAgent", "AutopilotResult"]


if __name__ == "__main__":
    SAMPLE_CONTEXT = {
        "human_moves": ["rock", "paper", "scissors", "scissors"],
        "robot_moves": ["paper", "scissors", "rock", "paper"],
        "results": ["robot", "human", "tie", "robot"],
        "round": 4,
        "opponent_info": {
            "ai_difficulty": "challenger",
            "ai_strategy": "to_win",
            "ai_personality": "neutral",
        },
        "ai_prediction": {
            "human_prediction": [0.3, 0.5, 0.2],
            "confidence": 0.62,
            "metadata": {
                "move_selection": {
                    "adjusted_distribution": [0.1, 0.28, 0.62],
                },
                "human_model": {
                    "pattern_strength": 0.48,
                    "recent_repetition": {"move": "scissors", "length": 2},
                    "change_factor": 0.12,
                },
            },
        },
        "game_status": {
            "metrics": {
                "human_win_rate": 0.25,
                "robot_win_rate": 0.5,
                "tie_rate": 0.25,
            }
        },
    }

    agent = AutopilotAgent()
    print("Running autopilot demo...\n")
    agent.run(SAMPLE_CONTEXT, verbose=True)
