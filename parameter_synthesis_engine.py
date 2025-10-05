"""
Parameter-Synthesis Engine (PSE) for RPS AI
===========================================

Generates 42 distinct AI opponents using systematic parameter combinations:
- 3 Difficulties: Rookie, Challenger, Master
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
    """Complete parameter set for an AI opponent."""
    
    # Identification
    difficulty: Difficulty
    strategy: Strategy
    personality: Personality
    opponent_id: str
    
    # Markov Predictor Parameters
    markov_order: int
    smoothing_factor: float
    ensemble_weights: Dict[int, float]  # {order: weight}
    
    # HLBM Parameters
    lambda_influence: float
    bias_weights: Dict[str, float]  # {RA, WSLS, ALT, CYC, META}
    bias_params: Dict[str, Dict[str, float]]
    
    # Strategy Parameters
    alpha: float  # To-win vs not-to-lose balance
    epsilon: float  # Exploration factor
    gamma: float  # Exploitation factor
    
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
            'lambda_influence': self.lambda_influence,
            'bias_weights': self.bias_weights,
            'bias_params': self.bias_params,
            'alpha': self.alpha,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
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
            lambda_influence=data['lambda_influence'],
            bias_weights=data['bias_weights'],
            bias_params=data['bias_params'],
            alpha=data['alpha'],
            epsilon=data['epsilon'],
            gamma=data['gamma'],
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
        
    def _init_base_parameters(self) -> None:
        """Initialize base parameter templates for each difficulty."""
        
        # Difficulty-based base parameters
        self.difficulty_presets = {
            Difficulty.ROOKIE: {
                'markov_order': 1,
                'smoothing_factor': 3,
                'ensemble_weights': {1: 0.65, 2: 0.3, 3: 0.05},
                'lambda_influence': 0.5,
                'bias_weights': {'RA': 0.32, 'WSLS': 0.28, 'ALT': 0.18, 'CYC': 0.12, 'META': 0.10},
                'bias_params': {
                    'RA': {'rho': 0.08},
                    'WSLS': {'delta_WS': 0.08, 'delta_LS': 0.06, 'delta_T': 0.03},
                    'ALT': {'delta_ALT': 0.08},
                    'CYC': {'delta_CYC': 0.08},
                    'META': {'delta_META': 0.02}
                },
                'epsilon': 0.2,
                'gamma': 0.4,
                'expected_win_rate': 0.45,
                'computational_complexity': 'low'
            },
            
            Difficulty.CHALLENGER: {
                'markov_order': 2,
                'smoothing_factor': 1.0,
                'ensemble_weights': {1: 0.3, 2: 0.5, 3: 0.2},
                'lambda_influence': 0.25,
                'bias_weights': {'RA': 0.28, 'WSLS': 0.30, 'ALT': 0.18, 'CYC': 0.16, 'META': 0.08},
                'bias_params': {
                    'RA': {'rho': 0.06},
                    'WSLS': {'delta_WS': 0.04, 'delta_LS': 0.05, 'delta_T': 0.02},
                    'ALT': {'delta_ALT': 0.04},
                    'CYC': {'delta_CYC': 0.04},
                    'META': {'delta_META': 0.015}
                },
                'epsilon': 0.15,
                'gamma': 0.75,
                'expected_win_rate': 0.55,
                'computational_complexity': 'medium'
            },
            
            Difficulty.MASTER: {
                'markov_order': 3,
                'smoothing_factor': 0.5,
                'ensemble_weights': {1: 0.1, 2: 0.3, 3: 0.6},
                'lambda_influence': 0.15,
                'bias_weights': {'RA': 0.24, 'WSLS': 0.30, 'ALT': 0.16, 'CYC': 0.24, 'META': 0.06},
                'bias_params': {
                    'RA': {'rho': 0.02},
                    'WSLS': {'delta_WS': 0.03, 'delta_LS': 0.02, 'delta_T': 0.01},
                    'ALT': {'delta_ALT': 0.02},
                    'CYC': {'delta_CYC': 0.02},
                    'META': {'delta_META': 0.01}
                },
                'epsilon': 0.08,
                'gamma': 0.90,
                'expected_win_rate': 0.65,
                'computational_complexity': 'high'
            }
        }
        
        # Strategy adjustments
        self.strategy_adjustments = {
            Strategy.TO_WIN: {
                'alpha': 0.95,
                'epsilon_multiplier': 1.2,  # More exploration
                'gamma_multiplier': 1.1,    # Slightly more exploitation
                'lambda_multiplier': 0.9,   # Less psychological bias
                'description_suffix': 'Aggressive play, seeks wins'
            },
            Strategy.NOT_TO_LOSE: {
                'alpha': 0.05,
                'epsilon_multiplier': 0.8,  # Less exploration
                'gamma_multiplier': 0.9,    # Less exploitation
                'lambda_multiplier': 1.1,   # More psychological bias
                'description_suffix': 'Conservative play, avoids losses'
            }
        }
        
        # Personality adjustments
        self.personality_adjustments = {
            Personality.NEUTRAL: {
                'delta_alpha': 0.0,
                'delta_epsilon': 0.0,
                'delta_gamma': 0.0,
                'delta_lambda': 0.0,
                'description': 'Balanced, no specific bias'
            },
            Personality.AGGRESSIVE: {
                'delta_alpha': 0.05,
                'delta_epsilon': 0.05,
                'delta_gamma': 0.1,
                'delta_lambda': -0.05,
                'description': 'High-risk, high-reward approach'
            },
            Personality.DEFENSIVE: {
                'delta_alpha': -0.1,
                'delta_epsilon': -0.05,
                'delta_gamma': -0.05,
                'delta_lambda': 0.05,
                'description': 'Risk-averse, conservative play'
            },
            Personality.UNPREDICTABLE: {
                'delta_alpha': 0.0,
                'delta_epsilon': 0.15,
                'delta_gamma': -0.1,
                'delta_lambda': 0.1,
                'description': 'Erratic, hard to predict'
            },
            Personality.CAUTIOUS: {
                'delta_alpha': -0.05,
                'delta_epsilon': -0.1,
                'delta_gamma': 0.05,
                'delta_lambda': 0.03,
                'description': 'Careful, methodical approach'
            },
            Personality.CONFIDENT: {
                'delta_alpha': 0.03,
                'delta_epsilon': 0.02,
                'delta_gamma': 0.08,
                'delta_lambda': -0.03,
                'description': 'Bold, assertive play style'
            },
            Personality.CHAMELEON: {
                'delta_alpha': 0.0,
                'delta_epsilon': 0.03,
                'delta_gamma': 0.02,
                'delta_lambda': 0.08,
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
        
        # Start with difficulty preset
        base_params = self.difficulty_presets[difficulty].copy()
        
        # Apply strategy adjustments
        strategy_adj = self.strategy_adjustments[strategy]
        alpha = strategy_adj['alpha']
        epsilon = base_params['epsilon'] * strategy_adj['epsilon_multiplier']
        gamma = base_params['gamma'] * strategy_adj['gamma_multiplier']
        lambda_influence = base_params['lambda_influence'] * strategy_adj['lambda_multiplier']
        
        # Apply personality adjustments
        personality_adj = self.personality_adjustments[personality]
        alpha += personality_adj['delta_alpha']
        epsilon += personality_adj['delta_epsilon']
        gamma += personality_adj['delta_gamma']
        lambda_influence += personality_adj['delta_lambda']
        
        # Apply personality adjustments to bias parameters
        adjusted_bias_params = {}
        for bias_type, params in base_params['bias_params'].items():
            adjusted_bias_params[bias_type] = {}
            for param_name, param_value in params.items():
                # Apply personality scaling to bias parameters
                scale_factor = 1.0
                
                if personality == Personality.AGGRESSIVE:
                    # Aggressive personalities have stronger biases
                    scale_factor = 1.2
                elif personality == Personality.DEFENSIVE:
                    # Defensive personalities have weaker biases
                    scale_factor = 0.8
                elif personality == Personality.UNPREDICTABLE:
                    # Unpredictable personalities have random bias scaling
                    scale_factor = np.random.uniform(0.7, 1.5)
                elif personality == Personality.CAUTIOUS:
                    # Cautious personalities have slightly weaker biases
                    scale_factor = 0.9
                elif personality == Personality.CONFIDENT:
                    # Confident personalities have slightly stronger biases
                    scale_factor = 1.1
                elif personality == Personality.CHAMELEON:
                    # Chameleon personalities adapt - stronger meta and wsls biases
                    if bias_type in ['META', 'WSLS']:
                        scale_factor = 1.3
                    else:
                        scale_factor = 0.9
                        
                adjusted_bias_params[bias_type][param_name] = param_value * scale_factor
        
        # Clamp values to valid ranges
        alpha = np.clip(alpha, 0.1, 0.9)
        epsilon = np.clip(epsilon, 0.05, 0.4)
        gamma = np.clip(gamma, 0.5, 0.95)
        lambda_influence = np.clip(lambda_influence, 0.10, 0.45)
        
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
            lambda_influence=lambda_influence,
            bias_weights=base_params['bias_weights'].copy(),
            bias_params=adjusted_bias_params,  # Use personality-adjusted bias parameters
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
        lambda_values = [opp.lambda_influence for opp in self.opponents.values()]
        
        return {
            'total_opponents': total_opponents,
            'expected_combinations': 3 * 2 * 7,
            'by_difficulty': {diff: len(opps) for diff, opps in by_difficulty.items()},
            'by_strategy': {strat: len(opps) for strat, opps in by_strategy.items()},
            'by_personality': {pers: len(opps) for pers, opps in by_personality.items()},
            'win_rate_range': (min(win_rates), max(win_rates)),
            'lambda_range': (min(lambda_values), max(lambda_values)),
            'avg_win_rate': np.mean(win_rates),
            'avg_lambda': np.mean(lambda_values)
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
    print(f"Lambda range: {stats['lambda_range']}")
    
    # Test specific opponent
    opponent = pse.get_opponent('challenger', 'to_win', 'aggressive')
    if opponent:
        print(f"\nSample opponent: {opponent.opponent_id}")
        print(f"Description: {opponent.description}")
        print(f"Alpha: {opponent.alpha}")
        print(f"Lambda: {opponent.lambda_influence}")
        print(f"Expected win rate: {opponent.expected_win_rate}")
    
    # Test filtering
    aggressive_opponents = pse.list_opponents(personality='aggressive')
    print(f"\nFound {len(aggressive_opponents)} aggressive opponents")
    
    # Save to file
    pse.save_opponents('test_opponents.json')
    print("Saved opponents to test_opponents.json")


if __name__ == "__main__":
    test_pse()