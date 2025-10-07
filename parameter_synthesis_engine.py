"""
Parameter-Synthesis Engine (PSE) for RPS AI
===========================================

Generates 42 distinct AI opponents using systematic parameter            Difficulty.CHALLENGER: {
                'markov_order': 2,
                'smoothing_factor': 0.6,
                'ensemble_weights': {1: 0.3, 2: 0.7},
                'lambda_influence': 0.25,   # Moderate reduction for balanced exploitation
                'bias_weights': {'RA': 0.25, 'WSLS': 0.28, 'ALT': 0.17, 'CYC': 0.24, 'META': 0.06},
                'bias_params': {
                    'RA': {'rho': 0.03},
                    'WSLS': {'delta_WS': 0.04, 'delta_LS': 0.03, 'delta_T': 0.02},
                    'ALT': {'delta_ALT': 0.03},
                    'CYC': {'delta_CYC': 0.03},
                    'META': {'delta_META': 0.02}
                },
                'epsilon': 0.12,            # Balanced exploration-exploitation
                'gamma': 0.80,              # Moderate exploitation focus
                'expected_win_rate': 0.62,  # Increased for better challenge
                'computational_complexity': 'medium'
            } Difficulties: Rookie, Challenger, Master, Grandmaster
- 2 Strategies: To-Win (α≈0.85), Not-to-Lose (α≈0.15)  
- 7 Personalities: Neutral, Aggressive, Defensive, Unpredictable, Cautious, Confident, Chameleon

Each combination produces unique Markov + HLBM parameter sets for diverse gameplay.

Author: AI Assistant
Created: 2025-10-03
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json


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


@dataclass
class OpponentParameters:
    """Complete parameter set for a single opponent configuration."""
    
    # Basic identification
    opponent_id: str
    difficulty: Difficulty
    strategy: Strategy
    personality: Personality
    
    # Markov Predictor Parameters
    markov_order: int
    smoothing_factor: float
    ensemble_weights: Dict[int, float]  # {order: weight}
    pattern_memory_limit: int  # Memory limit for pattern detection
    pattern_detection_speed: float  # NEW: How fast patterns are detected (difficulty-based)
    
    # Strategy Parameters (decision-making logic only)
    alpha: float  # To-win vs not-to-lose balance
    epsilon: float  # Exploration factor  
    gamma: float  # Exploitation factor
    
    # Personality Parameters (marginal probability adjustments only)
    personality_influence: float  # How much personality affects probabilities (very small)
    
    # Metadata
    description: str
    expected_win_rate: float
    computational_complexity: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'opponent_id': self.opponent_id,
            'difficulty': self.difficulty.value,
            'strategy': self.strategy.value,
            'personality': self.personality.value,
            'markov_order': self.markov_order,
            'smoothing_factor': self.smoothing_factor,
            'ensemble_weights': self.ensemble_weights,
            'pattern_memory_limit': self.pattern_memory_limit,
            'pattern_detection_speed': self.pattern_detection_speed,
            'alpha': self.alpha,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'personality_influence': self.personality_influence,
            'description': self.description,
            'expected_win_rate': self.expected_win_rate,
            'computational_complexity': self.computational_complexity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OpponentParameters':
        """Create from dictionary."""
        return cls(
            difficulty=Difficulty(data['difficulty']),
            strategy=Strategy(data['strategy']),
            personality=Personality(data['personality']),
            opponent_id=data['opponent_id'],
            markov_order=data['markov_order'],
            smoothing_factor=data['smoothing_factor'],
            ensemble_weights=data['ensemble_weights'],
            pattern_memory_limit=data.get('pattern_memory_limit', 25),  # Default for backward compatibility
            pattern_detection_speed=data.get('pattern_detection_speed', 1.0),  # Default
            alpha=data['alpha'],
            epsilon=data['epsilon'],
            gamma=data['gamma'],
            personality_influence=data.get('personality_influence', 0.05),  # Default
            description=data['description'],
            expected_win_rate=data['expected_win_rate'],
            computational_complexity=data['computational_complexity']
        )


class ParameterSynthesisEngine:
    """
    Parameter-Synthesis Engine that generates 42 distinct AI opponents.
    
    Uses systematic parameter combinations with difficulty presets,
    strategy layers, and personality adjustments.
    """
    
    def __init__(self):
        """Initialize PSE with base parameter templates."""
        self.opponents = {}  # {opponent_id: OpponentParameters}
        self._init_base_parameters()
        self.generate_all_opponents()  # Generate all 42 opponents
        
    def _init_base_parameters(self) -> None:
        """Initialize base parameter templates for each difficulty."""
        
        # Difficulty-based base parameters - ENHANCED SYSTEM per AI_DIFFICULTY_ENHANCEMENT_PLAN
        # Implementing multi-order Markov chains and improved exploitation parameters
        self.difficulty_presets = {
            Difficulty.ROOKIE: {
                # ROOKIE_PARAMS from enhancement plan: markov_orders [1,2], exploitation_strength 0.4, randomness_factor 0.3
                'markov_order': 2,  # Highest order for ensemble
                'smoothing_factor': 2.0,  # Higher smoothing = less aggressive pattern exploitation
                'ensemble_weights': {1: 0.7, 2: 0.3},  # Simple patterns only [1,2]
                'pattern_memory_limit': 15,  # Remember last 15 moves
                'pattern_detection_speed': 0.3,  # Slow adaptation ('slow' adaptation_speed)
                'epsilon': 0.30,  # 30% randomness (randomness_factor 0.3)
                'gamma': 0.40,    # Moderate exploitation (exploitation_strength 0.4)
                'personality_influence': 0.08,  # Higher personality influence for more variability
                'expected_win_rate': 0.35,  # Target 30-40% AI win rate
                'computational_complexity': 'low'
            },
            
            Difficulty.CHALLENGER: {
                # CHALLENGER_PARAMS from enhancement plan: markov_orders [1,2,3,5], exploitation_strength 0.7, randomness_factor 0.2
                'markov_order': 5,  # Highest order for ensemble
                'smoothing_factor': 0.8,  # Medium smoothing
                'ensemble_weights': {1: 0.15, 2: 0.35, 3: 0.35, 5: 0.15},  # Multi-order [1,2,3,5]
                'pattern_memory_limit': 25,  # Remember last 25 moves
                'pattern_detection_speed': 0.7,  # Medium adaptation ('medium' adaptation_speed)
                'epsilon': 0.20,  # 20% randomness (randomness_factor 0.2)
                'gamma': 0.70,    # Higher exploitation (exploitation_strength 0.7)
                'personality_influence': 0.05,  # Medium personality influence
                'expected_win_rate': 0.55,  # Target 50-60% AI win rate
                'computational_complexity': 'medium'
            },
            
            Difficulty.MASTER: {
                # MASTER_PARAMS from enhancement plan: markov_orders [1,2,3,5,7,10], exploitation_strength 0.9, randomness_factor 0.1
                'markov_order': 10,  # Highest order for ensemble
                'smoothing_factor': 0.3,  # Lower smoothing = aggressive exploitation
                'ensemble_weights': {1: 0.05, 2: 0.15, 3: 0.25, 5: 0.25, 7: 0.15, 10: 0.15},  # Long-term patterns [1,2,3,5,7,10]
                'pattern_memory_limit': 50,  # Remember last 50 moves
                'pattern_detection_speed': 1.0,  # Fast adaptation ('fast' adaptation_speed)
                'epsilon': 0.10,   # 10% randomness (randomness_factor 0.1)
                'gamma': 0.90,     # High exploitation (exploitation_strength 0.9)
                'personality_influence': 0.02,  # Minimal personality influence for consistency
                'expected_win_rate': 0.75,  # Target 70-80% AI win rate
                'computational_complexity': 'high'
            },
            
            Difficulty.GRANDMASTER: {
                # GRANDMASTER: pairs adaptive Markovism with Enhanced ML ensemble guidance
                'markov_order': 14,
                'smoothing_factor': 0.18,
                'ensemble_weights': {1: 0.04, 2: 0.1, 3: 0.18, 5: 0.2, 7: 0.16, 10: 0.16, 12: 0.08, 14: 0.08},
                'pattern_memory_limit': 110,
                'pattern_detection_speed': 1.35,
                'epsilon': 0.05,
                'gamma': 0.97,
                'personality_influence': 0.01,
                'expected_win_rate': 0.86,
                'computational_complexity': 'very_high'
            }
        }
        
        # Strategy adjustments - ONLY affect decision-making logic, not probabilities
        self.strategy_adjustments = {
            Strategy.TO_WIN: {
                'alpha': 0.95,  # Strongly favor aggressive win-seeking behavior
                'epsilon_multiplier': 1.2,  # Slightly more exploration when seeking wins
                'gamma_multiplier': 1.1,    # Slightly more exploitation when confident
                'description_suffix': 'Aggressive play, seeks wins'
            },
            Strategy.NOT_TO_LOSE: {
                'alpha': 0.05,  # Strongly favor conservative behavior
                'epsilon_multiplier': 0.8,  # Less exploration (more predictable)
                'gamma_multiplier': 0.9,    # Less exploitation (more cautious)
                'description_suffix': 'Conservative play, avoids losses'
            }
        }
        
        # Personality adjustments - ONLY marginal probability adjustments
        # No longer affects strategy parameters (alpha, epsilon, gamma)
        self.personality_adjustments = {
            Personality.NEUTRAL: {
                'influence_multiplier': 1.0,
                'description': 'Balanced, no specific bias'
            },
            Personality.AGGRESSIVE: {
                'influence_multiplier': 1.2,  # Slightly more personality influence
                'description': 'High-risk, high-reward approach'
            },
            Personality.DEFENSIVE: {
                'influence_multiplier': 0.8,  # Less personality influence (more methodical)
                'description': 'Risk-averse, conservative play'
            },
            Personality.UNPREDICTABLE: {
                'influence_multiplier': 1.5,  # More personality influence (more erratic)
                'description': 'Erratic, hard to predict'
            },
            Personality.CAUTIOUS: {
                'influence_multiplier': 0.6,  # Minimal personality influence
                'description': 'Careful, methodical approach'
            },
            Personality.CONFIDENT: {
                'influence_multiplier': 1.1,  # Slightly more influence
                'description': 'Bold, assertive play style'
            },
            Personality.CHAMELEON: {
                'influence_multiplier': 0.5,  # Minimal influence (adaptive)
                'description': 'Adaptive, mirrors human patterns'
            }
        }
        
    def generate_all_opponents(self) -> Dict[str, OpponentParameters]:
        """Generate all 42 opponents (3 × 2 × 7)."""
        
        opponents = {}
        
        for difficulty in Difficulty:
            for strategy in Strategy:
                for personality in Personality:
                    opponent = self._create_opponent(difficulty, strategy, personality)
                    opponents[opponent.opponent_id] = opponent
                    
        self.opponents = opponents
        return opponents
        
    def _create_opponent(self, 
                        difficulty: Difficulty, 
                        strategy: Strategy, 
                        personality: Personality) -> OpponentParameters:
        """Create a single opponent with specified characteristics."""
        
        # Get base difficulty parameters
        base_params = self.difficulty_presets[difficulty]
        
        # Calculate pattern detection speed from difficulty
        pattern_detection_speed = base_params['pattern_detection_speed']
        
        # Apply strategy adjustments
        strategy_adj = self.strategy_adjustments[strategy]
        alpha = strategy_adj['alpha']
        epsilon = base_params['epsilon'] * strategy_adj['epsilon_multiplier']
        gamma = base_params['gamma'] * strategy_adj['gamma_multiplier']
        
        # Get personality influence
        personality_adj = self.personality_adjustments[personality]
        personality_influence = personality_adj['influence_multiplier']
        
        # Clamp values to valid ranges
        alpha = np.clip(alpha, 0.1, 0.9)
        epsilon = np.clip(epsilon, 0.05, 0.4)
        gamma = np.clip(gamma, 0.5, 0.95)
        
        # Create opponent ID
        opponent_id = f"{difficulty.value}_{strategy.value}_{personality.value}"
        
        # Create description
        description = f"{difficulty.value.title()} {strategy_adj['description_suffix']} - {personality_adj['description']}"
        
        # Adjust expected win rate based on strategy and personality
        base_win_rate = base_params['expected_win_rate']
        if strategy == Strategy.TO_WIN:
            expected_win_rate = base_win_rate + 0.02
        else:  # NOT_TO_LOSE
            expected_win_rate = base_win_rate - 0.02
            
        if personality == Personality.AGGRESSIVE:
            expected_win_rate += 0.01
        elif personality == Personality.DEFENSIVE:
            expected_win_rate -= 0.01
            
        expected_win_rate = np.clip(expected_win_rate, 0.35, 0.75)
        
        return OpponentParameters(
            difficulty=difficulty,
            strategy=strategy,
            personality=personality,
            opponent_id=opponent_id,
            markov_order=base_params['markov_order'],
            smoothing_factor=base_params['smoothing_factor'],
            ensemble_weights=base_params['ensemble_weights'].copy(),
            pattern_memory_limit=base_params['pattern_memory_limit'],
            pattern_detection_speed=pattern_detection_speed,
            personality_influence=personality_influence,
            alpha=alpha,
            epsilon=epsilon,
            gamma=gamma,
            description=description,
            expected_win_rate=expected_win_rate,
            computational_complexity=base_params['computational_complexity']
        )
        
    def get_opponent(self, difficulty: str, strategy: str, personality: str) -> Optional[OpponentParameters]:
        """Get opponent by characteristics."""
        opponent_id = f"{difficulty.lower()}_{strategy.lower()}_{personality.lower()}"
        return self.opponents.get(opponent_id)
        
    def get_opponent_by_id(self, opponent_id: str) -> Optional[OpponentParameters]:
        """Get opponent by ID."""
        return self.opponents.get(opponent_id)
        
    def list_opponents(self, 
                      difficulty: Optional[str] = None,
                      strategy: Optional[str] = None,
                      personality: Optional[str] = None) -> List[OpponentParameters]:
        """List opponents matching criteria."""
        
        filtered = []
        for opponent in self.opponents.values():
            if difficulty and opponent.difficulty.value != difficulty.lower():
                continue
            if strategy and opponent.strategy.value != strategy.lower():
                continue
            if personality and opponent.personality.value != personality.lower():
                continue
            filtered.append(opponent)
            
        return filtered
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of all opponents."""
        
        if not self.opponents:
            self.generate_all_opponents()
            
        total_opponents = len(self.opponents)
        
        # Group by characteristics
        by_difficulty = {}
        by_strategy = {}
        by_personality = {}
        
        for opponent in self.opponents.values():
            # By difficulty
            diff = opponent.difficulty.value
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(opponent)
            
            # By strategy
            strat = opponent.strategy.value
            if strat not in by_strategy:
                by_strategy[strat] = []
            by_strategy[strat].append(opponent)
            
            # By personality
            pers = opponent.personality.value
            if pers not in by_personality:
                by_personality[pers] = []
            by_personality[pers].append(opponent)
            
        # Calculate statistics
        win_rates = [opp.expected_win_rate for opp in self.opponents.values()]
        pattern_speeds = [opp.pattern_detection_speed for opp in self.opponents.values()]
        personality_influences = [opp.personality_influence for opp in self.opponents.values()]
        
        return {
            'total_opponents': total_opponents,
            'expected_combinations': 3 * 2 * 7,
            'by_difficulty': {diff: len(opps) for diff, opps in by_difficulty.items()},
            'by_strategy': {strat: len(opps) for strat, opps in by_strategy.items()},
            'by_personality': {pers: len(opps) for pers, opps in by_personality.items()},
            'win_rate_range': (min(win_rates), max(win_rates)),
            'pattern_speed_range': (min(pattern_speeds), max(pattern_speeds)),
            'personality_influence_range': (min(personality_influences), max(personality_influences)),
            'avg_win_rate': np.mean(win_rates),
            'avg_pattern_speed': np.mean(pattern_speeds),
            'avg_personality_influence': np.mean(personality_influences)
        }
        
    def save_opponents(self, filepath: str) -> None:
        """Save all opponents to file."""
        data = {
            'meta': {
                'version': '1.0',
                'total_opponents': len(self.opponents),
                'generation_date': '2025-10-03'
            },
            'opponents': {
                opponent_id: opponent.to_dict() 
                for opponent_id, opponent in self.opponents.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_opponents(self, filepath: str) -> None:
        """Load opponents from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.opponents = {}
        for opponent_id, opponent_data in data['opponents'].items():
            self.opponents[opponent_id] = OpponentParameters.from_dict(opponent_data)


def create_parameter_synthesis_engine() -> ParameterSynthesisEngine:
    """Factory function to create and initialize PSE."""
    pse = ParameterSynthesisEngine()
    pse.generate_all_opponents()
    return pse


# Test function
def test_pse():
    """Test the Parameter-Synthesis Engine."""
    print("Testing Parameter-Synthesis Engine...")
    
    # Create PSE
    pse = create_parameter_synthesis_engine()
    
    # Get summary
    stats = pse.get_summary_stats()
    print(f"Generated {stats['total_opponents']} opponents")
    print(f"By difficulty: {stats['by_difficulty']}")
    print(f"By strategy: {stats['by_strategy']}")
    print(f"By personality: {stats['by_personality']}")
    print(f"Win rate range: {stats['win_rate_range']}")
    print(f"Pattern speed range: {stats['pattern_speed_range']}")
    print(f"Personality influence range: {stats['personality_influence_range']}")
    
    # Test specific opponent
    opponent = pse.get_opponent('challenger', 'to_win', 'aggressive')
    if opponent:
        print(f"\nSample opponent: {opponent.opponent_id}")
        print(f"Description: {opponent.description}")
        print(f"Alpha: {opponent.alpha}")
        print(f"Pattern Detection Speed: {opponent.pattern_detection_speed}")
        print(f"Personality Influence: {opponent.personality_influence}")
        print(f"Expected win rate: {opponent.expected_win_rate}")
    
    # Test filtering
    aggressive_opponents = pse.list_opponents(personality='aggressive')
    print(f"\nFound {len(aggressive_opponents)} aggressive opponents")
    
    # Save to file
    pse.save_opponents('test_opponents.json')
    print("Saved opponents to test_opponents.json")


if __name__ == "__main__":
    test_pse()
