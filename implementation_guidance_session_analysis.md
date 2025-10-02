# Implementation Guidance: Session Analysis, AI Strategies, and Validation Tolerances

## 1. Time Windows for Session Analysis

**For end-game coaching, use a hybrid approach:**

### Recommended Strategy
- **Core metrics**: Use entire session for overall performance assessment
- **Trend analysis**: Use rolling windows (last 10-15 moves) to show recent behavior
- **Pattern detection**: Use different window sizes based on metric type

#### Specific Recommendations

**Use Entire Session For:**
- Overall win rate
- Total entropy across session
- Complete pattern catalog
- Session-level Nash distance

**Use Rolling Windows For:**
- Recent adaptation rate (last 10 moves)
- Current predictability trend (last 15 moves)
- Recent decision complexity (last 8-12 moves)

**Use Aggregation Methods:**
- **Mean**: For stability metrics (consistency, risk tolerance)
- **Final**: For current state metrics (recent entropy, latest adaptation rate)
- **Trend**: For improvement indicators (slope of win rate over time)

##### Implementation Example
```python
# Session-level coaching metrics
session_metrics = {
    'overall_performance': calculate_entire_session(moves),
    'recent_trends': calculate_rolling_window(moves, window=10),
    'improvement_trajectory': calculate_trend_analysis(moves),
    'current_state': get_final_window_values(moves, window=5)
}
```

---

## 2. AI Strategy Implementation

**Use simplified approximations for data generation:**

### Recommended Approach
- **Don't implement full AI algorithms** – too complex for training data generation
- **Use statistical approximations** that capture the essence of each strategy
- **Focus on realistic human sequences** and generate plausible AI responses

#### Simplified Strategy Implementations
```python
def generate_ai_response_approximation(strategy_type, human_history):
    if strategy_type == 'frequency_based':
        # Counter most frequent human move
        return counter_to_most_frequent(human_history)
    elif strategy_type == 'markov_chain':
        # Counter predicted next move based on bigrams
        return counter_to_predicted_next(human_history)
    elif strategy_type == 'adaptive':
        # Switch strategies based on recent success
        return adaptive_counter_approximation(human_history)
    elif strategy_type == 'random':
        # Pure random with slight bias toward rock
        return weighted_random(['rock', 'paper', 'scissors'], [0.35, 0.33, 0.32])
```

#### Training Data Focus
1. **Generate realistic human sequences** first (based on psychological profiles)
2. **Apply simplified AI approximations** to create plausible robot responses
3. **Ensure win rates are realistic** (30-45% for most human players)

---

## 3. Validation Tolerances

**The tolerances shown are practical recommendations, not absolute requirements:**

### Actual Tolerances from Codebase Analysis

**Strict Tolerances (< 0.01):**
- Win rate calculations (must be precise)
- Probability distributions (must sum to 1.0)
- Nash equilibrium distances

**Moderate Tolerances (< 0.05):**
- Entropy calculations
- Mutual information values
- Compression ratios

**Flexible Tolerances (< 0.1):**
- Psychological metrics (inherently noisy)
- Adaptation rates (context-dependent)
- Decision complexity scores

#### Recommended Validation Tolerances
```python
VALIDATION_TOLERANCES = {
    # Core metrics - strict
    'win_rate': 0.01,
    'move_distribution': 0.01,
    'nash_distance': 0.01,
    # Information theory - moderate  
    'entropy': 0.05,
    'mutual_information': 0.05,
    'compression_ratio': 0.05,
    # Psychological - flexible
    'consistency_score': 0.1,
    'impulsiveness': 0.1,
    'risk_tolerance': 0.1,
    # Advanced analytics - moderate
    'adaptation_rate': 0.05,
    'predictability': 0.05,
    'decision_complexity': 0.08
}
```

#### Context-Dependent Adjustments
- **Early game** (< 10 moves): Use looser tolerances (+50%)
- **Short sessions** (< 20 moves): Use looser tolerances (+30%)
- **Long sessions** (> 100 moves): Use stricter tolerances (-20%)

##### Practical Implementation
```python
def validate_metric(metric_name, calculated_value, expected_value, game_length):
    base_tolerance = VALIDATION_TOLERANCES[metric_name]
    # Adjust tolerance based on game length
    if game_length < 10:
        tolerance = base_tolerance * 1.5
    elif game_length < 20:
        tolerance = base_tolerance * 1.3
    elif game_length > 100:
        tolerance = base_tolerance * 0.8
    else:
        tolerance = base_tolerance
    return abs(calculated_value - expected_value) <= tolerance
```

---

## Summary

- **Session Analysis**: Use hybrid approach (entire session + rolling windows + trends)
- **AI Strategies**: Use simplified approximations, focus on realistic human sequences
- **Validation**: Use the provided tolerances as baselines, adjust for context

These recommendations will give you realistic training data that matches the existing codebase behavior while being practical to implement.
