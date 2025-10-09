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

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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

def tool_predict_opponent_move(context: Dict[str, Any]) -> Dict[str, Any]:
    ai_pred = context.get("ai_prediction", {})
    metadata = ai_pred.get("metadata", {}) or context.get("ai_metadata", {})
    move_sel = metadata.get("move_selection", {})
    human_prediction = ai_pred.get("human_prediction")
    if not human_prediction:
        counts = {move: 1.0 for move in MOVES}
        history = context.get("human_moves", [])
        for move in history[-12:]:
            if move in counts:
                counts[move] += 1.0
        total = sum(counts.values())
        human_prediction = [counts[move] / total for move in MOVES]
    human_prediction = _normalise(human_prediction)

    opponent_dist = move_sel.get("adjusted_distribution")
    if opponent_dist:
        opponent_dist = _normalise(opponent_dist)
    else:
        derived = [0.0, 0.0, 0.0]
        for idx, move in enumerate(MOVES):
            counter = COUNTERS[move]
            counter_idx = MOVES.index(counter)
            derived[counter_idx] += human_prediction[idx]
        opponent_dist = _normalise(derived)

    top_idx = max(range(len(opponent_dist)), key=opponent_dist.__getitem__)
    top_move = MOVES[top_idx]

    return {
        "name": "predict",
        "opponent_distribution": {move: opponent_dist[i] for i, move in enumerate(MOVES)},
        "human_distribution": {move: human_prediction[i] for i, move in enumerate(MOVES)},
        "confidence": float(ai_pred.get("confidence", max(human_prediction))),
        "top_robot_move": top_move,
        "signals": {
            "pattern_strength": metadata.get("human_model", {}).get("pattern_strength"),
            "recent_repetition": metadata.get("human_model", {}).get("recent_repetition"),
            "change_factor": metadata.get("human_model", {}).get("change_factor"),
        },
    }


def tool_detect_behavior_shift(context: Dict[str, Any]) -> Dict[str, Any]:
    human_moves = context.get("human_moves", [])
    if len(human_moves) < 6:
        return {
            "name": "analytics",
            "drift": False,
            "confidence": 0.0,
            "label": "insufficient_data",
            "features": {},
            "action_suggestion": "maintain_strategy",
        }

    recent = human_moves[-3:]
    previous = human_moves[-6:-3]
    freq_recent = {move: recent.count(move) / len(recent) for move in MOVES}
    freq_previous = {move: previous.count(move) / len(previous) for move in MOVES}
    divergence = 0.5 * sum(abs(freq_recent[m] - freq_previous[m]) for m in MOVES)

    loss_streak = _recent_loss_streak(context.get("results", []))
    drift = divergence > 0.25 or loss_streak >= 3
    label = "bias_shift" if divergence > 0.25 else "losing_streak" if loss_streak >= 3 else "stable"

    action = "increase_exploration" if loss_streak >= 3 else "refresh_prediction"
    confidence = min(1.0, max(divergence * 1.3, loss_streak * 0.25))

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
        },
        "action_suggestion": action,
    }


def tool_choose_human_move(
    opponent_distribution: Dict[str, float],
    objective: str,
) -> Dict[str, Any]:
    utilities: Dict[str, float] = {}
    tie_bonus = 0.15
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
    confidence = max(0.0, min(1.0, abs(best_score - runner_up_score)))

    return {
        "name": "policy",
        "objective": objective,
        "candidate_move": best_move,
        "utilities": utilities,
        "confidence": confidence,
        "best_score": best_score,
    }


def tool_wildcard_override(state: Dict[str, Any]) -> Dict[str, Any]:
    predict_output = state["tool_outputs"].get("predict")
    policy_output = state["tool_outputs"].get("policy")
    if not predict_output or not policy_output:
        return {
            "name": "wildcard",
            "override_move": policy_output["candidate_move"] if policy_output else MOVES[0],
            "reason": "fallback",
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
    if wildcard_output and wildcard_output.get("override_move") == final_move:
        rationale = wildcard_output.get("reason", "wildcard override active")
    else:
        rationale = f"optimising {objective}"
    return (
        f"Robot likely plays {top_robot} ({robot_prob:.0%}); "
        f"choose {final_move} to {rationale}."
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


# ---------------------------------------------------------------------------
# Autopilot agent orchestrated by LangGraph
# ---------------------------------------------------------------------------

class AutopilotAgent:
    def __init__(
        self,
        llm_planner: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        max_graph_steps: int = 24,
    ) -> None:
        self.llm_planner = llm_planner or QwenMiniPlanner()
        self.max_graph_steps = max_graph_steps
        self.graph = self._build_graph()

    # Graph construction -----------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(dict)
        graph.set_entry_point("planner")
        graph.add_node("planner", self._node_planner)
        graph.add_node("call_tool_predict", self._node_tool_predict)
        graph.add_node("call_tool_analytics", self._node_tool_analytics)
        graph.add_node("call_tool_policy", self._node_tool_policy)
        graph.add_node("call_tool_wildcard", self._node_tool_wildcard)
        graph.add_node("finalize", self._node_finalize)

        graph.add_conditional_edges("planner", self._planner_router)
        graph.add_edge("call_tool_predict", "planner")
        graph.add_edge("call_tool_analytics", "planner")
        graph.add_edge("call_tool_policy", "planner")
        graph.add_edge("call_tool_wildcard", "planner")
        graph.add_edge("finalize", END)

        return graph.compile()

    # Node implementations ---------------------------------------------------
    def _node_planner(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
        return state

    def _planner_router(self, state: Dict[str, Any]) -> str:
        action = state.get("last_decision", {}).get("next_action", "finalize")
        mapping = {
            "call_tool_predict": "call_tool_predict",
            "call_tool_analytics": "call_tool_analytics",
            "call_tool_policy": "call_tool_policy",
            "call_tool_wildcard": "call_tool_wildcard",
            "finalize": "finalize",
        }
        return mapping.get(action, "finalize")

    def _node_tool_predict(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
        return state

    def _node_tool_analytics(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        output = tool_detect_behavior_shift(state["context"])
        state["tool_outputs"]["analytics"] = output
        state["log"].append(
            f"[Stage tool_analytics] drift={output['drift']} "
            f"confidence={output['confidence']:.2f} label={output['label']}"
        )
        return state

    def _node_tool_policy(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        predict_output = state["tool_outputs"].get("predict")
        if not predict_output:
            raise RuntimeError("Policy tool requires prediction output")
        objective = state["meta"].get("objective", "maximize_win")
        output = tool_choose_human_move(
            predict_output["opponent_distribution"],
            objective,
        )
        state["tool_outputs"]["policy"] = output
        utilities = ", ".join(
            f"{move}:{score:.2f}" for move, score in output["utilities"].items()
        )
        state["log"].append(
            f"[Stage tool_policy] candidate={output['candidate_move']} "
            f"confidence_gap={output['confidence']:.2f} utilities=({utilities})"
        )
        return state

    def _node_tool_wildcard(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        output = tool_wildcard_override(state)
        state["tool_outputs"]["wildcard"] = output
        state["log"].append(
            f"[Stage tool_wildcard] override={output['override_move']} "
            f"reason={output['reason']}"
        )
        return state

    def _node_finalize(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        predict_output = state["tool_outputs"].get("predict")
        policy_output = state["tool_outputs"].get("policy")
        if not predict_output or not policy_output:
            raise RuntimeError("Finalize requires prediction and policy outputs")

        candidate_move = policy_output["candidate_move"]
        analytics = state["tool_outputs"].get("analytics", {})
        wildcard_output = state["tool_outputs"].get("wildcard")
        if wildcard_output:
            candidate_move = wildcard_output.get("override_move", candidate_move)

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
        return state

    # Public API -------------------------------------------------------------
    def run(self, context: Dict[str, Any], verbose: bool = False) -> AutopilotResult:
        initial_state = {
            "context": context,
            "tool_outputs": {},
            "meta": {},
            "log": [],
            "iterations": 0,
        }
        final_state = self.graph.invoke(
            initial_state,
            config={"max_steps": self.max_graph_steps},
        )
        result = AutopilotResult(
            final_move=final_state.get("final_move"),
            predicted_robot_move=final_state.get("predicted_robot_move"),
            log=list(final_state.get("log", [])),
            tool_outputs=dict(final_state.get("tool_outputs", {})),
            context=context,
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
