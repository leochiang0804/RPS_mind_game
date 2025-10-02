# Advanced Metrics Calculation Guide for RPS Coaching System

## Overview
This document provides detailed formulas and algorithms for calculating advanced metrics in the Rock-Paper-Scissors coaching system. These calculations ensure realistic and meaningful data generation for training.

---

## Advanced Analytics Metrics

### 1. Decision Complexity
**Definition**: Measures the cognitive load and complexity of decision-making patterns.

**Calculation**:
```python
def calculate_decision_complexity(moves, timing_data=None, win_history=None):
    """
    Calculate decision complexity based on multiple factors:
    - Move variety and unpredictability
    - Response time variation (if available)
    - Context switching frequency
    - Strategic depth indicators
    """
    if len(moves) < 3:
        return 0.5  # Default for insufficient data
    
    # Factor 1: Move entropy (30% weight)
    entropy = calculate_entropy(moves)
    max_entropy = 1.585  # log2(3)
    entropy_factor = min(entropy / max_entropy, 1.0)
    
    # Factor 2: Pattern breaking frequency (25% weight)
    pattern_breaks = count_pattern_breaks(moves)
    total_opportunities = max(len(moves) - 2, 1)
    pattern_break_factor = min(pattern_breaks / total_opportunities, 1.0)
    
    # Factor 3: Context adaptation (25% weight)
    adaptation_factor = calculate_context_adaptation(moves, win_history)
    
    # Factor 4: Sequence complexity (20% weight)
    sequence_complexity = calculate_sequence_complexity(moves)
    
    complexity = (
        0.30 * entropy_factor +
        0.25 * pattern_break_factor +
        0.25 * adaptation_factor +
        0.20 * sequence_complexity
    )
    
    return min(max(complexity, 0.0), 1.0)

def count_pattern_breaks(moves):
    """Count how often player breaks from predictable patterns"""
    breaks = 0
    for i in range(2, len(moves)):
        # Check for repetition break
        if moves[i-2] == moves[i-1] and moves[i] != moves[i-1]:
            breaks += 1
        # Check for alternation break
        elif (i >= 3 and moves[i-3] != moves[i-2] and 
              moves[i-2] != moves[i-1] and moves[i] == moves[i-2]):
            breaks += 1
    return breaks

def calculate_context_adaptation(moves, win_history):
    """Measure adaptation to game context"""
    if not win_history or len(moves) < 5:
        return 0.5
    
    adaptations = 0
    for i in range(3, len(moves)):
        recent_performance = sum(win_history[i-3:i]) / 3
        if recent_performance < 0.3:  # Poor performance
            # Check if strategy changed
            recent_moves = moves[i-3:i]
            if len(set(recent_moves)) >= 2:  # Strategy diversification
                adaptations += 1
    
    return min(adaptations / max(len(moves) - 3, 1), 1.0)

def calculate_sequence_complexity(moves):
    """Measure complexity of move sequences"""
    if len(moves) < 4:
        return 0.0
    
    # Calculate sequence uniqueness
    sequences = []
    for i in range(len(moves) - 2):
        sequences.append(tuple(moves[i:i+3]))
    
    unique_ratio = len(set(sequences)) / len(sequences)
    return unique_ratio
```

### 2. Strategy Consistency
**Definition**: Measures how consistent the player's strategic approach is over time.

**Calculation**:
```python
def calculate_strategy_consistency(moves, window_size=5):
    """
    Calculate consistency of strategic approach across time windows
    """
    if len(moves) < window_size * 2:
        return 0.5
    
    consistencies = []
    
    for i in range(window_size, len(moves) - window_size + 1):
        window1 = moves[i-window_size:i]
        window2 = moves[i:i+window_size]
        
        # Calculate distribution similarity
        dist1 = get_move_distribution(window1)
        dist2 = get_move_distribution(window2)
        
        # Jensen-Shannon divergence for distribution similarity
        similarity = 1 - jensen_shannon_divergence(dist1, dist2)
        consistencies.append(similarity)
    
    return sum(consistencies) / len(consistencies) if consistencies else 0.5

def get_move_distribution(moves):
    """Get probability distribution of moves"""
    from collections import Counter
    counts = Counter(moves)
    total = len(moves)
    return {move: counts.get(move, 0) / total for move in ['rock', 'paper', 'scissors']}

def jensen_shannon_divergence(p, q):
    """Calculate Jensen-Shannon divergence between two distributions"""
    import math
    
    def kl_divergence(p, q):
        return sum(p[x] * math.log2(p[x] / q[x]) for x in p if p[x] > 0 and q[x] > 0)
    
    # Average distribution
    m = {x: (p[x] + q[x]) / 2 for x in p}
    
    # JS divergence
    js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return js_div / math.log2(2)  # Normalize to [0,1]
```

### 3. Adaptation Rate
**Definition**: Measures how quickly a player adapts their strategy in response to AI behavior.

**Calculation**:
```python
def calculate_adaptation_rate(moves, ai_moves, win_history, window_size=5):
    """
    Calculate adaptation rate based on strategy changes after losses
    """
    if len(moves) < window_size * 2:
        return 0.5
    
    adaptations = []
    
    for i in range(window_size, len(moves) - window_size):
        # Identify if adaptation was needed (poor recent performance)
        recent_wins = sum(win_history[i-window_size:i])
        win_rate = recent_wins / window_size
        
        if win_rate < 0.4:  # Poor performance threshold
            # Measure strategy change
            before_strategy = analyze_strategy_pattern(moves[i-window_size:i])
            after_strategy = analyze_strategy_pattern(moves[i:i+window_size])
            
            adaptation_strength = calculate_strategy_distance(before_strategy, after_strategy)
            adaptations.append(adaptation_strength)
    
    return sum(adaptations) / len(adaptations) if adaptations else 0.5

def analyze_strategy_pattern(moves):
    """Analyze strategic pattern in move sequence"""
    patterns = {
        'repetition': count_repetitions(moves),
        'alternation': count_alternations(moves),
        'cycling': count_cycles(moves),
        'randomness': calculate_entropy(moves)
    }
    return patterns

def calculate_strategy_distance(strategy1, strategy2):
    """Calculate distance between two strategy patterns"""
    total_distance = 0
    for key in strategy1:
        total_distance += abs(strategy1[key] - strategy2[key])
    return min(total_distance / 4, 1.0)  # Normalize
```

### 4. Nash Distance
**Definition**: Distance from Nash equilibrium (optimal mixed strategy of 1/3 each move).

**Calculation**:
```python
def calculate_nash_distance(moves):
    """
    Calculate distance from Nash equilibrium
    Nash equilibrium: 1/3 probability for each move
    """
    from collections import Counter
    
    if not moves:
        return 0.33  # Maximum distance
    
    counts = Counter(moves)
    total = len(moves)
    
    # Calculate actual probabilities
    actual_probs = {
        'rock': counts.get('rock', 0) / total,
        'paper': counts.get('paper', 0) / total,
        'scissors': counts.get('scissors', 0) / total
    }
    
    # Nash equilibrium probabilities
    nash_prob = 1/3
    
    # Calculate L1 distance (Manhattan distance)
    distance = sum(abs(prob - nash_prob) for prob in actual_probs.values())
    
    # Normalize to [0, 2/3] range, then scale to [0, 0.5]
    max_distance = 2/3  # Maximum possible L1 distance
    normalized_distance = distance / max_distance * 0.5
    
    return min(normalized_distance, 0.5)
```

### 5. Exploitability
**Definition**: Measures how exploitable a player's strategy is by an optimal opponent.

**Calculation**:
```python
def calculate_exploitability(moves, window_size=10):
    """
    Calculate exploitability based on predictable patterns
    """
    if len(moves) < window_size:
        return calculate_simple_exploitability(moves)
    
    exploitabilities = []
    
    for i in range(window_size, len(moves) + 1):
        window = moves[i-window_size:i]
        exploit_score = calculate_window_exploitability(window)
        exploitabilities.append(exploit_score)
    
    return sum(exploitabilities) / len(exploitabilities)

def calculate_window_exploitability(moves):
    """Calculate exploitability for a window of moves"""
    # Factor 1: Frequency bias (40% weight)
    freq_exploit = calculate_frequency_exploitability(moves)
    
    # Factor 2: Pattern predictability (35% weight)
    pattern_exploit = calculate_pattern_exploitability(moves)
    
    # Factor 3: Sequence predictability (25% weight)
    sequence_exploit = calculate_sequence_exploitability(moves)
    
    total_exploit = (
        0.40 * freq_exploit +
        0.35 * pattern_exploit +
        0.25 * sequence_exploit
    )
    
    return min(total_exploit, 1.0)

def calculate_frequency_exploitability(moves):
    """Exploitability based on move frequency bias"""
    from collections import Counter
    counts = Counter(moves)
    total = len(moves)
    
    # Calculate expected gain for optimal counter-strategy
    max_freq = max(counts.values()) / total
    optimal_gain = max_freq - 1/3  # Gain above random play
    
    return min(optimal_gain * 3, 1.0)  # Scale to [0,1]

def calculate_pattern_exploitability(moves):
    """Exploitability based on detectable patterns"""
    if len(moves) < 3:
        return 0.0
    
    pattern_scores = []
    
    # Check for repetition patterns
    for i in range(2, len(moves)):
        if moves[i-2] == moves[i-1]:  # Repetition detected
            if moves[i] == moves[i-1]:  # Continued repetition
                pattern_scores.append(0.8)
            else:
                pattern_scores.append(0.3)  # Broken repetition
    
    # Check for alternation patterns
    alternations = 0
    for i in range(2, len(moves)):
        if moves[i-2] != moves[i-1] and moves[i-1] != moves[i] and moves[i] == moves[i-2]:
            alternations += 1
    
    if alternations > len(moves) * 0.3:
        pattern_scores.append(0.6)
    
    return sum(pattern_scores) / len(moves) if pattern_scores else 0.0
```

### 6. Mutual Information
**Definition**: Information shared between consecutive moves (measures move independence).

**Calculation**:
```python
def calculate_mutual_information(moves):
    """
    Calculate mutual information between consecutive moves
    MI(X_{t}, X_{t+1}) measures how much knowing current move tells us about next move
    """
    if len(moves) < 2:
        return 0.0
    
    import math
    from collections import Counter, defaultdict
    
    # Create pairs of consecutive moves
    pairs = [(moves[i], moves[i+1]) for i in range(len(moves)-1)]
    
    # Count occurrences
    pair_counts = Counter(pairs)
    current_counts = Counter(moves[:-1])
    next_counts = Counter(moves[1:])
    
    total_pairs = len(pairs)
    mutual_info = 0.0
    
    for (current, next_move), pair_count in pair_counts.items():
        # P(X_t, X_{t+1})
        p_joint = pair_count / total_pairs
        
        # P(X_t) and P(X_{t+1})
        p_current = current_counts[current] / total_pairs
        p_next = next_counts[next_move] / total_pairs
        
        # MI contribution: P(x,y) * log2(P(x,y) / (P(x) * P(y)))
        if p_joint > 0 and p_current > 0 and p_next > 0:
            mi_term = p_joint * math.log2(p_joint / (p_current * p_next))
            mutual_info += mi_term
    
    return max(mutual_info, 0.0)  # Ensure non-negative
```

### 7. Compression Ratio
**Definition**: Measures how compressible the move sequence is (lower compression = more random).

**Calculation**:
```python
def calculate_compression_ratio(moves):
    """
    Calculate compression ratio using run-length encoding
    Lower ratio indicates more randomness
    """
    if len(moves) < 2:
        return 1.0
    
    # Method 1: Run-length encoding compression
    rle_compressed = run_length_encode(moves)
    rle_ratio = len(rle_compressed) / len(moves)
    
    # Method 2: Pattern-based compression
    pattern_compressed = pattern_compress(moves)
    pattern_ratio = len(pattern_compressed) / len(moves)
    
    # Method 3: Dictionary-based compression (simplified LZ77)
    dict_compressed = dictionary_compress(moves)
    dict_ratio = len(dict_compressed) / len(moves)
    
    # Average of compression methods
    avg_ratio = (rle_ratio + pattern_ratio + dict_ratio) / 3
    return min(avg_ratio, 1.0)

def run_length_encode(moves):
    """Simple run-length encoding"""
    if not moves:
        return []
    
    encoded = []
    current_move = moves[0]
    count = 1
    
    for move in moves[1:]:
        if move == current_move:
            count += 1
        else:
            encoded.append((current_move, count))
            current_move = move
            count = 1
    
    encoded.append((current_move, count))
    return encoded

def pattern_compress(moves, max_pattern_length=4):
    """Compress based on repeating patterns"""
    compressed = []
    i = 0
    
    while i < len(moves):
        best_pattern = None
        best_length = 0
        
        # Look for repeating patterns
        for pattern_len in range(1, min(max_pattern_length, len(moves) - i) + 1):
            pattern = moves[i:i + pattern_len]
            repetitions = 1
            
            # Count repetitions
            pos = i + pattern_len
            while pos + pattern_len <= len(moves) and moves[pos:pos + pattern_len] == pattern:
                repetitions += 1
                pos += pattern_len
            
            if repetitions > 1 and repetitions * pattern_len > best_length:
                best_pattern = pattern
                best_length = repetitions * pattern_len
        
        if best_pattern and len(best_pattern) > 1:
            compressed.append(('pattern', best_pattern, best_length // len(best_pattern)))
            i += best_length
        else:
            compressed.append(moves[i])
            i += 1
    
    return compressed
```

---

## Psychological Metrics

### 1. Impulsiveness
**Definition**: Tendency to make quick, reactive decisions without strategic consideration.

**Calculation**:
```python
def calculate_impulsiveness(moves, timing_data=None, win_history=None):
    """
    Calculate impulsiveness based on:
    - Reaction to immediate losses
    - Quick strategy changes
    - Lack of planning indicators
    """
    if len(moves) < 5:
        return 0.5
    
    impulsive_indicators = []
    
    # Factor 1: Immediate reaction to losses (40% weight)
    if win_history:
        loss_reactions = calculate_loss_reactions(moves, win_history)
        impulsive_indicators.append(('loss_reaction', loss_reactions, 0.4))
    
    # Factor 2: Move switching frequency (30% weight)
    switch_frequency = calculate_move_switching(moves)
    impulsive_indicators.append(('switching', switch_frequency, 0.3))
    
    # Factor 3: Pattern abandonment (30% weight)
    pattern_abandonment = calculate_pattern_abandonment(moves)
    impulsive_indicators.append(('abandonment', pattern_abandonment, 0.3))
    
    # Weighted average
    total_weight = sum(weight for _, _, weight in impulsive_indicators)
    weighted_sum = sum(score * weight for _, score, weight in impulsive_indicators)
    
    return min(weighted_sum / total_weight, 1.0) if total_weight > 0 else 0.5

def calculate_loss_reactions(moves, win_history):
    """Calculate tendency to change strategy immediately after losses"""
    reactions = 0
    opportunities = 0
    
    for i in range(1, len(moves)):
        if i < len(win_history) and win_history[i-1] == 0:  # Previous round was a loss
            opportunities += 1
            if moves[i] != moves[i-1]:  # Changed move after loss
                reactions += 1
    
    return reactions / opportunities if opportunities > 0 else 0.0

def calculate_move_switching(moves):
    """Calculate frequency of move changes"""
    if len(moves) < 2:
        return 0.0
    
    switches = sum(1 for i in range(1, len(moves)) if moves[i] != moves[i-1])
    max_switches = len(moves) - 1
    
    return switches / max_switches
```

### 2. Consistency Score
**Definition**: Different from strategy_consistency - measures psychological consistency in decision-making approach.

**Calculation**:
```python
def calculate_consistency_score(moves, timing_data=None):
    """
    Psychological consistency - stability of decision-making approach
    Different from strategy_consistency which measures strategic patterns
    """
    if len(moves) < 8:
        return 0.5
    
    # Factor 1: Response time consistency (if available)
    time_consistency = 0.5
    if timing_data and len(timing_data) == len(moves):
        time_consistency = calculate_timing_consistency(timing_data)
    
    # Factor 2: Decision confidence indicators
    confidence_consistency = calculate_decision_confidence_consistency(moves)
    
    # Factor 3: Behavioral pattern stability
    behavioral_consistency = calculate_behavioral_stability(moves)
    
    overall_consistency = (
        0.3 * time_consistency +
        0.4 * confidence_consistency +
        0.3 * behavioral_consistency
    )
    
    return min(max(overall_consistency, 0.0), 1.0)

def calculate_timing_consistency(timing_data):
    """Calculate consistency in decision timing"""
    import statistics
    
    if len(timing_data) < 3:
        return 0.5
    
    # Calculate coefficient of variation (std/mean)
    mean_time = statistics.mean(timing_data)
    std_time = statistics.stdev(timing_data)
    
    if mean_time == 0:
        return 0.0
    
    cv = std_time / mean_time
    # Lower CV indicates higher consistency
    consistency = max(0, 1 - cv)
    return min(consistency, 1.0)

def calculate_decision_confidence_consistency(moves):
    """Infer decision confidence from move patterns"""
    # Assumption: Confident decisions show less second-guessing
    windows = []
    window_size = 4
    
    for i in range(len(moves) - window_size + 1):
        window = moves[i:i + window_size]
        # Calculate "decisiveness" of window
        unique_moves = len(set(window))
        repetitions = sum(1 for j in range(1, len(window)) if window[j] == window[j-1])
        
        # Higher unique moves and fewer repetitions indicate confidence
        decisiveness = (unique_moves / window_size) * (1 - repetitions / (window_size - 1))
        windows.append(decisiveness)
    
    if not windows:
        return 0.5
    
    # Consistency is low variance in decisiveness
    import statistics
    if len(windows) == 1:
        return windows[0]
    
    variance = statistics.variance(windows)
    consistency = max(0, 1 - variance * 4)  # Scale factor
    return min(consistency, 1.0)
```

### 3. Risk Tolerance
**Definition**: Willingness to take strategic risks vs. playing safe strategies.

**Calculation**:
```python
def calculate_risk_tolerance(moves, win_history=None, ai_strategy=None):
    """
    Calculate risk tolerance based on:
    - Willingness to break winning patterns
    - Response to losing streaks
    - Strategy diversity under pressure
    """
    if len(moves) < 5:
        return {'level': 'moderate', 'score': 0.5}
    
    risk_indicators = []
    
    # Factor 1: Breaking winning patterns (high risk)
    if win_history:
        pattern_breaking = calculate_winning_pattern_breaks(moves, win_history)
        risk_indicators.append(pattern_breaking)
    
    # Factor 2: Strategy diversity under pressure
    pressure_diversity = calculate_pressure_diversity(moves, win_history)
    risk_indicators.append(pressure_diversity)
    
    # Factor 3: Exploration vs exploitation
    exploration_ratio = calculate_exploration_ratio(moves)
    risk_indicators.append(exploration_ratio)
    
    avg_risk = sum(risk_indicators) / len(risk_indicators)
    
    # Categorize risk tolerance
    if avg_risk < 0.3:
        level = 'conservative'
    elif avg_risk < 0.7:
        level = 'moderate'
    else:
        level = 'aggressive'
    
    return {'level': level, 'score': avg_risk}

def calculate_winning_pattern_breaks(moves, win_history):
    """Calculate tendency to change strategy while winning"""
    breaks = 0
    opportunities = 0
    
    for i in range(3, len(moves)):
        # Check if player was on a winning streak
        recent_wins = sum(win_history[i-3:i])
        if recent_wins >= 2:  # At least 2 wins in last 3 rounds
            opportunities += 1
            # Check if they changed their pattern anyway (risky)
            if moves[i] != moves[i-1]:
                breaks += 1
    
    return breaks / opportunities if opportunities > 0 else 0.5

def calculate_pressure_diversity(moves, win_history):
    """Calculate strategy diversity under losing pressure"""
    if not win_history or len(moves) < 6:
        return 0.5
    
    pressure_windows = []
    
    for i in range(3, len(moves) - 2):
        # Identify pressure situations (recent losses)
        recent_performance = sum(win_history[i-3:i]) / 3
        if recent_performance < 0.4:  # Under pressure
            # Measure diversity in next few moves
            next_moves = moves[i:i+3] if i+3 <= len(moves) else moves[i:]
            diversity = len(set(next_moves)) / len(next_moves)
            pressure_windows.append(diversity)
    
    return sum(pressure_windows) / len(pressure_windows) if pressure_windows else 0.5
```

### 4. Emotional Indicators
**Definition**: Patterns that indicate emotional states during gameplay.

**Calculation**:
```python
def calculate_emotional_indicators(moves, win_history, timing_data=None):
    """
    Detect emotional states from gameplay patterns
    """
    indicators = {}
    
    if len(moves) < 5:
        return {'state': 'neutral', 'confidence': 0.5, 'stability': 0.5}
    
    # Indicator 1: Tilt detection (emotional distress after losses)
    tilt_score = detect_tilt_patterns(moves, win_history)
    indicators['tilt'] = tilt_score
    
    # Indicator 2: Confidence level
    confidence_level = detect_confidence_level(moves, win_history)
    indicators['confidence'] = confidence_level
    
    # Indicator 3: Emotional stability
    stability = calculate_emotional_stability(moves, win_history, timing_data)
    indicators['stability'] = stability
    
    # Indicator 4: Frustration patterns
    frustration = detect_frustration_patterns(moves, win_history)
    indicators['frustration'] = frustration
    
    # Overall emotional state
    primary_state = determine_primary_emotional_state(indicators)
    
    return {
        'state': primary_state,
        'indicators': indicators,
        'confidence': confidence_level,
        'stability': stability
    }

def detect_tilt_patterns(moves, win_history):
    """Detect emotional tilt from loss-reaction patterns"""
    if not win_history or len(moves) < 4:
        return 0.0
    
    tilt_indicators = 0
    total_loss_sequences = 0
    
    # Look for sequences following losses
    for i in range(2, len(moves)):
        if i < len(win_history) and win_history[i-1] == 0:  # Previous loss
            total_loss_sequences += 1
            
            # Tilt indicators:
            # 1. Immediate strategy flip
            if moves[i] != moves[i-1]:
                tilt_indicators += 0.3
            
            # 2. Erratic pattern in next few moves
            if i + 2 < len(moves):
                next_three = moves[i:i+3]
                if len(set(next_three)) == 3:  # All different moves
                    tilt_indicators += 0.4
            
            # 3. Abandoning previously successful patterns
            if i >= 3:
                prev_pattern = moves[i-3:i]
                if calculate_pattern_success_rate(prev_pattern, win_history[i-3:i]) > 0.6:
                    if moves[i] not in prev_pattern[-2:]:  # Abandoned successful pattern
                        tilt_indicators += 0.5
    
    return min(tilt_indicators / max(total_loss_sequences, 1), 1.0)

def detect_confidence_level(moves, win_history):
    """Detect confidence level from move patterns"""
    if len(moves) < 5:
        return 0.5
    
    confidence_indicators = []
    
    # Factor 1: Consistency in successful periods
    for i in range(4, len(moves)):
        recent_performance = sum(win_history[i-4:i]) / 4
        if recent_performance > 0.6:  # Good performance period
            # Check for consistent strategy during success
            recent_moves = moves[i-4:i]
            consistency = 1 - (len(set(recent_moves)) / len(recent_moves))
            confidence_indicators.append(consistency * 0.8)  # High weight for success consistency
    
    # Factor 2: Response to winning streaks
    for i in range(3, len(moves)):
        if i >= 2 and sum(win_history[i-2:i]) == 2:  # Two consecutive wins
            # Confident players maintain strategy
            if i + 1 < len(moves) and moves[i+1] == moves[i]:
                confidence_indicators.append(0.7)
            else:
                confidence_indicators.append(0.3)
    
    return sum(confidence_indicators) / len(confidence_indicators) if confidence_indicators else 0.5

def determine_primary_emotional_state(indicators):
    """Determine primary emotional state from indicators"""
    tilt = indicators.get('tilt', 0)
    confidence = indicators.get('confidence', 0.5)
    frustration = indicators.get('frustration', 0)
    stability = indicators.get('stability', 0.5)
    
    if tilt > 0.6:
        return 'tilted'
    elif frustration > 0.6:
        return 'frustrated'
    elif confidence > 0.7 and stability > 0.6:
        return 'confident'
    elif confidence < 0.3 and stability < 0.4:
        return 'uncertain'
    else:
        return 'neutral'
```

---

## Advanced Context Variables

### 1. Cognitive Patterns
**Definition**: High-level thinking patterns detected in gameplay.

**Detection Algorithm**:
```python
def detect_cognitive_patterns(moves, win_history, ai_strategy=None):
    """
    Detect cognitive patterns in decision-making
    """
    patterns = {}
    
    # Pattern 1: Forward thinking (planning ahead)
    forward_thinking = detect_forward_thinking(moves, win_history)
    patterns['forward_thinking'] = forward_thinking
    
    # Pattern 2: Reactive vs proactive decision making
    reactivity = calculate_reactivity_score(moves, win_history)
    patterns['reactivity'] = reactivity
    
    # Pattern 3: Learning adaptation
    learning_pattern = detect_learning_patterns(moves, win_history)
    patterns['learning_adaptation'] = learning_pattern
    
    # Pattern 4: Meta-game awareness
    meta_awareness = detect_meta_game_awareness(moves, ai_strategy)
    patterns['meta_awareness'] = meta_awareness
    
    return patterns

def detect_forward_thinking(moves, win_history):
    """Detect evidence of planning ahead vs immediate reactions"""
    if len(moves) < 6:
        return 0.5
    
    planning_evidence = 0
    total_decisions = 0
    
    for i in range(3, len(moves) - 2):
        total_decisions += 1
        
        # Look for decisions that sacrifice short-term for long-term
        current_move = moves[i]
        recent_pattern = moves[i-3:i]
        
        # If breaking a recently successful pattern for strategic reasons
        if calculate_pattern_success_rate(recent_pattern, win_history[i-3:i]) > 0.6:
            if current_move not in recent_pattern:  # Breaking successful pattern
                # Check if this leads to future success
                future_performance = sum(win_history[i:i+3]) if i+3 <= len(win_history) else 0
                if future_performance >= 2:  # Good future performance
                    planning_evidence += 1
    
    return planning_evidence / total_decisions if total_decisions > 0 else 0.5
```

### 2. Strategic Opportunities
**Definition**: Algorithmically identified opportunities for improvement.

**Identification Algorithm**:
```python
def identify_strategic_opportunities(moves, win_history, ai_strategy):
    """
    Identify specific strategic improvement opportunities
    """
    opportunities = []
    
    # Opportunity 1: Exploitable patterns in player's moves
    pattern_opportunities = identify_pattern_opportunities(moves)
    opportunities.extend(pattern_opportunities)
    
    # Opportunity 2: Suboptimal responses to AI strategy
    response_opportunities = identify_response_opportunities(moves, ai_strategy)
    opportunities.extend(response_opportunities)
    
    # Opportunity 3: Timing-based opportunities
    timing_opportunities = identify_timing_opportunities(moves, win_history)
    opportunities.extend(timing_opportunities)
    
    # Opportunity 4: Meta-strategy opportunities
    meta_opportunities = identify_meta_opportunities(moves, win_history, ai_strategy)
    opportunities.extend(meta_opportunities)
    
    # Rank opportunities by impact potential
    ranked_opportunities = rank_opportunities_by_impact(opportunities, moves, win_history)
    
    return ranked_opportunities[:5]  # Return top 5 opportunities

def identify_pattern_opportunities(moves):
    """Identify opportunities to break exploitable patterns"""
    opportunities = []
    
    # Check for repetition patterns
    repetition_score = calculate_repetition_vulnerability(moves)
    if repetition_score > 0.6:
        opportunities.append({
            'type': 'pattern_breaking',
            'description': 'Break repetitive move patterns',
            'impact': 'high',
            'urgency': 'immediate',
            'specific_advice': 'Vary your moves more frequently to avoid predictability'
        })
    
    # Check for alternation patterns
    alternation_score = calculate_alternation_vulnerability(moves)
    if alternation_score > 0.5:
        opportunities.append({
            'type': 'alternation_breaking',
            'description': 'Reduce alternating patterns',
            'impact': 'medium',
            'urgency': 'near_term',
            'specific_advice': 'Avoid simple back-and-forth patterns'
        })
    
    return opportunities

def rank_opportunities_by_impact(opportunities, moves, win_history):
    """Rank opportunities by potential impact on win rate"""
    current_win_rate = sum(win_history) / len(win_history) if win_history else 0.5
    
    for opportunity in opportunities:
        # Estimate potential win rate improvement
        if opportunity['type'] == 'pattern_breaking' and current_win_rate < 0.4:
            opportunity['estimated_improvement'] = 0.15  # 15% win rate boost
        elif opportunity['type'] == 'response_optimization':
            opportunity['estimated_improvement'] = 0.10
        else:
            opportunity['estimated_improvement'] = 0.05
    
    # Sort by estimated improvement
    return sorted(opportunities, key=lambda x: x['estimated_improvement'], reverse=True)
```

### 3. Educational Focus
**Definition**: Determination of what the player should focus on learning next.

**Algorithm**:
```python
def determine_educational_focus(moves, win_history, current_skill_level=None):
    """
    Determine what the player should focus on learning next
    """
    focus_areas = []
    
    # Assess current skill level if not provided
    if current_skill_level is None:
        current_skill_level = assess_skill_level(moves, win_history)
    
    # Focus area 1: Randomization (fundamental skill)
    randomization_need = assess_randomization_need(moves)
    if randomization_need > 0.6:
        focus_areas.append({
            'area': 'randomization',
            'priority': 'high',
            'description': 'Learning to play more unpredictably',
            'skill_level': 'beginner',
            'exercises': ['practice mixed strategies', 'use randomization devices']
        })
    
    # Focus area 2: Pattern recognition (intermediate skill)
    if current_skill_level >= 2:  # Intermediate or above
        pattern_recognition_need = assess_pattern_recognition_need(moves, win_history)
        if pattern_recognition_need > 0.5:
            focus_areas.append({
                'area': 'pattern_recognition',
                'priority': 'medium',
                'description': 'Learning to recognize and exploit opponent patterns',
                'skill_level': 'intermediate',
                'exercises': ['analyze opponent frequencies', 'track pattern sequences']
            })
    
    # Focus area 3: Meta-game strategy (advanced skill)
    if current_skill_level >= 3:  # Advanced
        meta_game_need = assess_meta_game_need(moves, win_history)
        if meta_game_need > 0.4:
            focus_areas.append({
                'area': 'meta_game',
                'priority': 'low',
                'description': 'Advanced strategic thinking and adaptation',
                'skill_level': 'advanced',
                'exercises': ['study game theory', 'practice deliberate deception']
            })
    
    # Rank by priority and current need
    ranked_focus = rank_educational_priorities(focus_areas, current_skill_level)
    
    return ranked_focus[0] if ranked_focus else {
        'area': 'general_improvement',
        'priority': 'medium',
        'description': 'Continue practicing and maintaining current skills'
    }

def assess_skill_level(moves, win_history):
    """Assess player's current skill level (1-5 scale)"""
    if len(moves) < 10:
        return 1  # Insufficient data
    
    # Factor 1: Win rate against adaptive AI
    win_rate = sum(win_history) / len(win_history) if win_history else 0.33
    
    # Factor 2: Unpredictability
    entropy = calculate_entropy(moves)
    max_entropy = 1.585
    unpredictability = entropy / max_entropy
    
    # Factor 3: Strategic consistency
    consistency = calculate_strategy_consistency(moves)
    
    # Factor 4: Adaptation ability
    adaptation = calculate_adaptation_rate(moves, [], win_history)
    
    # Weighted score
    skill_score = (
        0.3 * win_rate +
        0.3 * unpredictability +
        0.2 * consistency +
        0.2 * adaptation
    )
    
    # Convert to 1-5 scale
    if skill_score < 0.2:
        return 1  # Beginner
    elif skill_score < 0.4:
        return 2  # Novice
    elif skill_score < 0.6:
        return 3  # Intermediate
    elif skill_score < 0.8:
        return 4  # Advanced
    else:
        return 5  # Expert
```

---

## Implementation Notes

### Data Generation Guidelines

1. **Consistency Checks**: Always ensure calculated metrics are mathematically consistent with the generated move sequences.

2. **Realistic Ranges**: Use the provided formulas to generate metrics within realistic ranges based on actual human gameplay patterns.

3. **Temporal Coherence**: Ensure that metrics evolve logically over time (e.g., adaptation_rate should change based on actual performance changes).

4. **AI Strategy Simulation**: When simulating AI responses, ensure the AI behaves according to its stated strategy type.

### Validation Methods

```python
def validate_metrics_consistency(moves, ai_moves, win_history, metrics):
    """
    Validate that calculated metrics are consistent with actual game data
    """
    validations = {}
    
    # Validate win rate
    calculated_win_rate = sum(win_history) / len(win_history)
    reported_win_rate = metrics.get('win_rate', 0)
    validations['win_rate'] = abs(calculated_win_rate - reported_win_rate) < 0.01
    
    # Validate entropy
    calculated_entropy = calculate_entropy(moves)
    reported_entropy = metrics.get('entropy', 0)
    validations['entropy'] = abs(calculated_entropy - reported_entropy) < 0.05
    
    # Validate move distribution
    from collections import Counter
    actual_dist = Counter(moves)
    total = len(moves)
    calculated_dist = {move: actual_dist[move] / total for move in ['rock', 'paper', 'scissors']}
    reported_dist = metrics.get('move_distribution', {})
    
    dist_error = sum(abs(calculated_dist.get(move, 0) - reported_dist.get(move, 0)) 
                     for move in ['rock', 'paper', 'scissors'])
    validations['move_distribution'] = dist_error < 0.05
    
    return validations
```

This comprehensive guide provides the mathematical foundation for generating realistic and consistent metrics for training data generation.

---

## Codebase Analysis and Implementation Guidelines

### Existing Implementations in the Project

Based on analysis of the current codebase (`ai_coach_metrics.py`), here are the answers to key implementation questions:

#### 1. **Psychological Metrics - Existing Implementations**

**Impulsiveness Indicator**:
```python
def _assess_impulsiveness(self, human_history):
    """Already implemented in ai_coach_metrics.py"""
    # Based on immediate strategy changes after losses
    # Calculated as: adaptations / loss_opportunities
    # Range: 0.0 (never changes after loss) to 1.0 (always changes)
```

**Consistency Score**:
```python
def _assess_consistency(self, human_history):
    """Existing implementation uses variance-based approach"""
    # Calculates variance in move patterns over windows
    # Formula: 1.0 - min(avg_variance * 3, 1.0)
    # Range: 0.0 (highly inconsistent) to 1.0 (very consistent)
```

**Risk Tolerance**:
```python
def _assess_risk_tolerance(self, human_history, result_history):
    """Implemented based on move variety under pressure"""
    # Analyzes move diversity during losing streaks
    # Higher diversity under pressure = higher risk tolerance
```

#### 2. **Mutual Information - Exact Implementation**

The project already has a working implementation:
```python
def _calculate_mutual_information(self, game_state: Dict[str, Any]) -> float:
    """Calculate mutual information between consecutive moves"""
    moves = game_state.get('human_history', [])
    
    if len(moves) < 3:
        return self._format_metric(0.0)
    
    # Calculate joint probability of consecutive moves
    bigrams = self._count_bigrams(moves)  # Format: "move1->move2"
    total_bigrams = len(moves) - 1
    
    # Individual move probabilities
    move_dist = self._calculate_move_distribution(moves)
    
    # Mutual information calculation
    mutual_info = 0.0
    for bigram, count in bigrams.items():
        if '->' in bigram:
            move1, move2 = bigram.split('->')
            p_joint = count / total_bigrams
            p_move1 = move_dist.get(move1, 0)
            p_move2 = move_dist.get(move2, 0)
            
            if p_joint > 0 and p_move1 > 0 and p_move2 > 0:
                mutual_info += p_joint * math.log2(p_joint / (p_move1 * p_move2))
    
    return self._format_metric(mutual_info)
```

#### 3. **Compression Ratio - Simplified Implementation**

Current implementation uses run-length encoding:
```python
def _calculate_compression_ratio(self, moves: List[str]) -> float:
    """Calculate compression ratio using simple RLE"""
    if len(moves) < 5:
        return self._format_metric(1.0)
    
    # Simple run-length encoding simulation
    compressed_length = 1
    current_move = moves[0]
    
    for move in moves[1:]:
        if move == current_move:
            run_length += 1  # Same move continues
        else:
            compressed_length += 1  # New run starts
            current_move = move
    
    compression_ratio = compressed_length / len(moves)
    return self._format_metric(compression_ratio)
    
    # Lower ratio = more repetitive patterns
    # Higher ratio = more random/complex patterns
```

#### 4. **Adaptation Rate - Optimal Time Windows**

The existing implementation uses these time windows:
```python
def _calculate_adaptation_rate(self, game_state: Dict[str, Any]) -> float:
    """Optimal time windows based on existing implementation"""
    moves = game_state.get('human_history', [])
    results = game_state.get('result_history', [])
    
    # Minimum window: 10 moves (too short gives unreliable data)
    if len(moves) < 10 or len(results) < 10:
        return self._format_metric(0.5)
    
    # Analysis window: Single-move reaction (immediate adaptation)
    # Look at strategy changes after each loss
    adaptations = 0
    loss_opportunities = 0
    
    for i in range(1, min(len(moves), len(results))):
        if results[i-1] == 'robot':  # Previous move was a loss
            loss_opportunities += 1
            if moves[i] != moves[i-1]:  # Player changed strategy
                adaptations += 1
    
    return adaptations / loss_opportunities if loss_opportunities > 0 else 0.5
```

**Recommended Time Windows for Different Analyses**:
- **Immediate Adaptation**: 1-move lag (current implementation)
- **Short-term Adaptation**: 3-5 move windows
- **Medium-term Adaptation**: 5-10 move windows  
- **Long-term Adaptation**: 10+ move windows

#### 5. **Advanced Context Variables - Existing Algorithms**

**Cognitive Patterns**:
```python
def _assess_pattern_awareness(self, game_state):
    """Detects if player recognizes and counters AI patterns"""
    # Analyzes response effectiveness to AI strategy changes
    
def _assess_meta_cognition(self, game_state):
    """Measures thinking about thinking"""
    # Looks for deliberate pattern breaks after successful AI counters
```

**Strategic Opportunities**:
```python
def _identify_strategic_opportunities(self, game_state):
    """Algorithmic opportunity detection"""
    # 1. Pattern vulnerability analysis
    # 2. Nash equilibrium deviation opportunities  
    # 3. AI exploitation opportunities
    # 4. Timing-based opportunities
```

**Educational Focus**:
```python
def _determine_educational_focus(self, game_state):
    """Determines learning priorities"""
    # Based on skill assessment algorithm:
    # - Entropy < 1.0 → Focus on randomization
    # - Predictability > 0.7 → Focus on pattern breaking
    # - Adaptation rate < 0.3 → Focus on responsiveness
    # - Nash distance > 0.4 → Focus on balance
```

### Implementation Recommendations

#### **For Data Generation**:

1. **Use Existing Implementations**: The project already has robust implementations for mutual information, compression ratio, and basic psychological metrics. Use these directly.

2. **Time Window Standards**:
   - **Real-time metrics**: 1-3 move windows
   - **Adaptation analysis**: 5-10 move windows  
   - **Session analysis**: Full sequence analysis

3. **Metric Consistency**: Always use the `_format_metric()` method to ensure consistent formatting:
   ```python
   def _format_metric(self, value, decimals=3):
       """Format metric to consistent decimal places"""
       return round(float(value), decimals)
   ```

4. **Simplified vs Complex Implementations**:
   - **Use simplified versions** for real-time calculations
   - **Use complex versions** for post-game analysis
   - **Maintain consistency** between both versions

#### **For Training Data Generation**:

1. **Start with existing implementations** from `ai_coach_metrics.py`
2. **Validate consistency** using the provided validation methods
3. **Use realistic value ranges** based on observed data in the codebase
4. **Ensure temporal coherence** - metrics should evolve logically over time

#### **Missing Implementations to Create**:

1. **Decision Complexity**: Use the comprehensive formula provided in this guide
2. **Strategy Consistency**: Enhance the existing variance-based approach with Jensen-Shannon divergence
3. **Exploitability**: Implement the multi-factor approach from this guide
4. **Enhanced Psychological Metrics**: Extend beyond the basic implementations

### Validation Against Existing Code

The metrics calculated should match these existing patterns:
- **Entropy**: 0.0 to 1.585 (log₂(3))
- **Mutual Information**: 0.0 to 1.585 (matches entropy range)
- **Compression Ratio**: 0.1 to 1.0 (lower = more patterns)
- **Adaptation Rate**: 0.0 to 1.0 (frequency of strategy changes)
- **Nash Distance**: 0.0 to ~0.57 (Euclidean distance in 3D space)

This ensures your generated training data will be consistent with the existing codebase and realistic for the coaching domain.


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
