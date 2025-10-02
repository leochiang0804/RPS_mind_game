# Rock-Paper-Scissors AI Coach Prompt Templates & Metrics Documentation

## Overview
This document provides comprehensive documentation for knowledge distillation training of an AI coaching assistant for Rock-Paper-Scissors gameplay. It includes all prompt templates, metric definitions, and variable explanations needed to generate realistic training data.

## Coaching Style Instructions

### Style Instruction Options
The `{style_instruction}` variable can take two primary values:

- **"easy-to-understand and encouraging"**: For casual players who need accessible advice
- **"scientific and detailed"**: For advanced players who want technical analysis

### Examples of Style Variations:
- Easy Style: "Try mixing up your moves more! You're being too predictable."
- Scientific Style: "Your entropy score of 0.845 indicates suboptimal randomization. Consider implementing a mixed strategy closer to Nash equilibrium."

---

## Real-Time Coaching Prompt Template

### Template Structure
```
You are an expert Rock-Paper-Scissors coach. A player needs your help with their game strategy.

CURRENT SITUATION:
{context}

COACHING APPROACH: Provide {style_instruction} guidance that's helpful and encouraging.

Based on the game data above, please provide personalized coaching advice. Your response MUST be structured using the following clear section headers (in this order):

---
**IMMEDIATE TIPS**
List 2-3 specific things the player can do right now to improve.

**PATTERN INSIGHTS**
Describe what you notice about their playing style, including any habits or tendencies.

**STRATEGIC ADVICE**
Explain how they can adapt to beat the AI's current strategy.

**ENCOURAGEMENT**
Give positive reinforcement about what the player is doing well.
---

Please use these section headers verbatim and keep your advice concise, actionable, and easy to follow. Respond in a natural, conversational way as if you're coaching them in person. Be specific about the numbers and patterns you see, but explain them in an accessible way.
```

### Context Variables for Real-Time Coaching

The `{context}` variable contains a comprehensive set of metrics formatted as natural language. Here's what each metric means:

#### Core Game Metrics
- **current_round**: Current round number (integer, 1+)
- **total_moves**: Total moves played so far (integer, 0+)
- **win_rate**: Human player's win percentage (float, 0.0-1.0)
- **recent_moves**: Array of last 5 human moves (e.g., ['rock', 'paper', 'scissors', 'rock', 'paper'])
- **recent_results**: Array of last 5 game results (e.g., ['human', 'robot', 'tie', 'human', 'robot'])

#### Pattern Analysis Metrics
- **entropy**: Measure of randomness in moves (float, 0.0-1.585)
  - 0.0 = completely predictable
  - 1.585 = maximum theoretical randomness
  - Typical values: 0.5-1.4
- **predictability**: How predictable the player's moves are (float, 0.0-1.0)
  - 0.0 = completely unpredictable
  - 1.0 = completely predictable
  - Good range: 0.2-0.5
- **move_distribution**: Frequency of each move (e.g., {'rock': 0.4, 'paper': 0.3, 'scissors': 0.3})
- **sequence_patterns**: Detected patterns in move sequences
- **cycling_patterns**: Whether player cycles through moves predictably

#### Advanced Analytics
- **decision_complexity**: Cognitive load of decision-making (float, 0.0-1.0)
- **strategy_consistency**: How consistent the strategy is (float, 0.0-1.0)
- **adaptation_rate**: How quickly player adapts to AI (float, 0.0-1.0)
- **nash_distance**: Distance from Nash equilibrium (float, 0.0-0.5)
  - 0.0 = perfect Nash equilibrium (33.33% each move)
  - 0.5 = maximum deviation
- **exploitability**: How exploitable the player's strategy is (float, 0.0-1.0)
- **mutual_information**: Information shared between consecutive moves (float, 0.0+)
- **compression_ratio**: How compressible the move sequence is (float, 0.0-1.0)

#### Psychological Metrics
- **impulsiveness**: Tendency to make impulsive decisions (float, 0.0-1.0)
- **consistency_score**: Psychological consistency (float, 0.0-1.0)
- **risk_tolerance**: Risk-taking behavior analysis
- **emotional_indicators**: Current emotional state indicators
- **cognitive_patterns**: Cognitive decision-making patterns

#### AI Behavior Analysis
- **robot_strategy**: AI's current strategy ('frequency_based', 'markov_chain', 'adaptive', 'random')
- **human_strategy_label**: Detected human strategy pattern
- **ai_confidence**: AI's confidence in its predictions (float, 0.0-1.0)
- **ai_adaptation**: How the AI is adapting to human play
- **prediction_patterns**: AI's prediction accuracy patterns

#### Performance Data
- **streaks**: Current win/loss streaks and longest streaks
- **momentum**: Current momentum direction and score
- **recent_performance**: Performance trend over recent rounds

#### Temporal Analysis
- **game_phase**: Current phase ('early', 'mid', 'late')
- **performance_timeline**: Performance changes over time

#### Strategic Context
- **current_strategy_assessment**: Assessment of current human strategy
- **strategic_opportunities**: Identified opportunities for improvement
- **strategic_weaknesses**: Current strategic weaknesses
- **educational_focus**: Recommended learning focus areas

### Example Context Data
```
PLAYER'S CURRENT SITUATION:
• Round 15 with 45.2% win rate
• Recent moves: ['rock', 'paper', 'rock', 'scissors', 'rock']
• Entropy (randomness): 1.234 out of 1.585 maximum
• Predictability: 0.567 (lower is better)
• Nash distance: 0.123 (measures optimal play balance)
• AI opponent using: frequency_based strategy
• Decision complexity: 0.678
• Pattern consistency: 0.456

ANALYSIS CONTEXT:
The player's move distribution shows {'rock': 0.45, 'paper': 0.28, 'scissors': 0.27}. Their recent performance trend is {'momentum_direction': 'declining', 'momentum_score': -0.23}.
```

---

## Comprehensive Post-Game Analysis Prompt Template

### Template Structure
```
You are an expert Rock-Paper-Scissors coach conducting a comprehensive post-game analysis for a player.

COMPLETE SESSION DATA:
{session_data}

ANALYSIS STYLE: Provide {style_instruction} insights that help the player understand their strategic development.

Your response MUST be structured using the following section headers, in this order. For each section, provide clear, concise, and actionable feedback. Write in a conversational but well-organized and professional tone.

---
**PSYCHOLOGICAL PATTERNS**
Analyze the player's decision-making style, risk tolerance, and any psychological tendencies observed during the session.

**STRATEGIC EVOLUTION**
Describe how the player's strategy developed throughout the session. Note any adaptations, shifts in approach, or learning moments.

**DECISION ANALYSIS**
Identify patterns in the player's choices. Are their moves predictable or random? Highlight any habits or exploitable tendencies.

**PERFORMANCE INSIGHTS**
Interpret key metrics (entropy, predictability, win rate, etc.) and explain what they reveal about the player's gameplay.

**LEARNING OPPORTUNITIES**
Suggest specific areas the player should focus on to improve, based on your analysis.

**FASCINATING DISCOVERIES**
Share any interesting or unexpected patterns, behaviors, or turning points you noticed.

**EDUCATIONAL SUMMARY**
Summarize the main lessons and connect them to relevant game theory or strategy principles.

**PERSONALIZED ROADMAP**
Provide a step-by-step improvement plan tailored to the player's current skill level and playing style.

---

Please use these section headers verbatim and keep your advice well-structured, easy to follow, and actionable. Avoid emojis. Respond as if you are having a thoughtful conversation with the player about their game.
```

### Session Data Variables for Comprehensive Analysis

The `{session_data}` variable contains a complete summary of the entire game session:

#### Session Overview
- **Total Rounds**: Total number of rounds played
- **Final Performance**: Final win rate percentage
- **Session Duration**: Time spent playing (seconds)

#### Pattern Analysis Summary
- **Pattern Type**: Overall pattern classification ('high_variation', 'low_variation', 'mixed_pattern', 'single_move_repetition')
- **Final Entropy**: Final entropy score
- **Final Predictability**: Final predictability score
- **Most Exploited Pattern**: The pattern most exploited by the AI

#### Performance Evolution
- **Starting Performance**: Early game performance description
- **Ending Performance**: Late game performance description
- **Best Streak**: Longest winning streak
- **Worst Streak**: Longest losing streak
- **Overall Trend**: Performance trend ('improving', 'declining', 'stable')

#### Strategic Development
- **AI Strategy Faced**: The AI strategy encountered
- **Player Adaptation**: How well the player adapted
- **Learning Indicators**: Evidence of learning during the session
- **Key Turning Point**: Major momentum shifts or breakthroughs

#### Educational Insights
- **Main Weakness Exploited**: Primary weakness the AI exploited
- **Biggest Improvement Area**: Top recommendation for improvement
- **Strategic Lesson Learned**: Key strategic insight from the session

#### Psychological Profile
- **Decision Making Style**: Psychological decision-making patterns
- **Consistency Throughout**: How consistency evolved
- **Emotional Resilience**: Emotional stability during difficult periods

### Example Session Data
```
📊 SESSION OVERVIEW:
Total Rounds: 25
Final Performance: 52.0% win rate
Session Duration: 180 seconds

🎯 PATTERN ANALYSIS:
Pattern Type: mixed_pattern patterns detected
Final Entropy: 1.342 (max 1.585)
Final Predictability: 0.423 (lower is better)
Most Exploited Pattern: High predictability in cycling patterns (67.5%)

📈 PERFORMANCE EVOLUTION:
Starting Performance: Early rounds showed developing strategy
Ending Performance: Strong finish with improving play
Best Streak: 4 wins
Worst Streak: 3 losses
Overall Trend: improving

🧠 STRATEGIC DEVELOPMENT:
AI Strategy Faced: frequency_based
Player Adaptation: Successfully adapted and improved throughout session
Learning Indicators: Strong learning - improved randomness and performance
Key Turning Point: Mid-session win streak of 4 showed strategic breakthrough

🎓 EDUCATIONAL INSIGHTS:
Main Weakness Exploited: Tendency to repeat same moves too frequently
Biggest Improvement Area: Focus on unpredictability - break patterns and increase randomness
Strategic Lesson Learned: Successfully countered frequency analysis by varying move patterns

💡 PSYCHOLOGICAL PROFILE:
Decision Making Style: {'impulsiveness_indicator': 0.34, 'consistency_score': 0.67, 'risk_tolerance': 'moderate'}
Consistency Throughout: Maintained good strategic consistency throughout
Emotional Resilience: Strong emotional control - maintained positive momentum
```

---

## Metric Calculation Guidelines for Data Generation

### Creating Realistic Metrics
When generating training data, ensure metrics are mathematically consistent:

#### Win Rate Calculation
```python
win_rate = human_wins / total_rounds
# Should match the actual sequence of moves and outcomes
```

#### Entropy Calculation
```python
import math
from collections import Counter

def calculate_entropy(moves):
    counts = Counter(moves)
    total = len(moves)
    entropy = 0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy
```

#### Predictability Score
```python
def calculate_predictability(moves):
    # Based on pattern detection algorithms
    # Higher score = more predictable
    # Consider: repetitions, cycles, alternations
    # Range: 0.0 (random) to 1.0 (completely predictable)
```

#### Nash Distance
```python
def calculate_nash_distance(move_distribution):
    ideal_prob = 1/3  # 33.33% for each move
    distance = 0
    for move in ['rock', 'paper', 'scissors']:
        prob = move_distribution.get(move, 0)
        distance += abs(prob - ideal_prob)
    return distance / 2  # Normalize to 0-0.5 range
```

### AI Strategy Behaviors
- **frequency_based**: AI tracks move frequencies and counters most common moves
- **markov_chain**: AI analyzes move sequences and predicts next moves
- **adaptive**: AI changes strategy based on player adaptation
- **random**: AI plays randomly

### Realistic Value Ranges
- **Entropy**: 0.5-1.4 (most players), 1.4+ (very good randomization)
- **Predictability**: 0.3-0.8 (typical range), <0.3 (excellent), >0.8 (poor)
- **Win Rate**: 20-70% (typical range against adaptive AI)
- **Decision Complexity**: 0.2-0.8 (based on move variety and timing)
- **Nash Distance**: 0.05-0.4 (closer to 0 is better)

---

## Training Data Generation Guidelines

### Sequence Generation
1. Generate realistic move sequences (15-50 rounds)
2. Implement AI counter-strategies based on patterns
3. Calculate all metrics based on actual sequence
4. Ensure temporal consistency (early vs late game metrics)

### Response Quality Standards
- **Immediate Tips**: 2-3 specific, actionable suggestions
- **Pattern Insights**: Accurate analysis of actual patterns in data
- **Strategic Advice**: Counter-strategies for detected AI behavior
- **Encouragement**: Positive reinforcement based on actual performance

### Consistency Checks
- Metrics must mathematically match the generated sequences
- Performance trends should align with calculated win rates
- Pattern descriptions should match entropy/predictability scores
- AI strategy labels should match the AI's actual behavior in the sequence

This documentation ensures that generated training data will be realistic and analytically sound for knowledge distillation training.