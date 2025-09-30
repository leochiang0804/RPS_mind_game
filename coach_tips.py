"""
Coach Tips Generator for Rock-Paper-Scissors ML Game
Analyzes player patterns and provides intelligent coaching advice
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter, deque
import random

class CoachTipsGenerator:
    def __init__(self):
        """Initialize the coaching system"""
        self.tip_templates = {
            'predictability': [
                "You're being too predictable! Try mixing up your moves more randomly.",
                "The AI is catching onto your pattern. Switch to a more unpredictable strategy.",
                "Your moves are easy to predict. Try avoiding obvious sequences."
            ],
            'repeater': [
                "You're repeating the same move too often. Mix it up to confuse the AI!",
                "Try breaking your repetition habit - the AI expects you to repeat '{move}'.",
                "Avoid playing the same move more than twice in a row."
            ],
            'cycler': [
                "Your cycling pattern is too regular. Try breaking the Rock→Paper→Scissors sequence.",
                "The AI has learned your cycle. Skip a move or reverse the order!",
                "Mix in some random moves to break your predictable cycle."
            ],
            'biased': [
                "You favor '{move}' too much. Try playing it less often to balance your strategy.",
                "Your {move} bias is showing. The AI is countering it effectively.",
                "Reduce your reliance on '{move}' - aim for more balanced play."
            ],
            'anti_strategy': [
                "Try the opposite of what you just played to confuse the AI.",
                "Counter your own last move - if you played Rock, try Paper next.",
                "Use anti-patterns: do the opposite of what feels natural."
            ],
            'randomness': [
                "Embrace randomness! Don't think too hard about your next move.",
                "Try flipping a mental coin for your next few moves.",
                "Random play is your best defense against pattern recognition."
            ],
            'adaptation': [
                "The AI adapted to your strategy around round {round}. Time to switch it up!",
                "You changed strategies recently - keep evolving to stay ahead.",
                "Good strategy shift! Keep the AI guessing with more changes."
            ],
            'performance': [
                "You're winning {win_rate}% - keep up the unpredictable play!",
                "Your win rate dropped to {win_rate}%. Try a completely different approach.",
                "Tie rate is high at {tie_rate}% - you're playing defensively well."
            ]
        }
        
        self.experiments = [
            {
                'name': 'Random Play',
                'description': 'Play completely randomly for 5-10 rounds',
                'strategy': 'Ignore patterns and just pick moves randomly'
            },
            {
                'name': 'Counter-Cycle', 
                'description': 'Play Paper→Scissors→Rock (reverse cycle)',
                'strategy': 'Use the opposite of the traditional Rock→Paper→Scissors'
            },
            {
                'name': 'Mirror Strategy',
                'description': 'Copy what the AI just played',
                'strategy': 'If AI played Rock, you play Rock next'
            },
            {
                'name': 'Anti-Mirror',
                'description': 'Counter what the AI just played',
                'strategy': 'If AI played Rock, you play Paper to beat it'
            },
            {
                'name': 'Frequency Counter',
                'description': 'Counter the AI\'s most common move',
                'strategy': 'Track what the AI plays most and counter it'
            },
            {
                'name': 'Double Repeat',
                'description': 'Play the same move twice, then switch',
                'strategy': 'Rock, Rock, then change to something else'
            },
            {
                'name': 'Emotional Play',
                'description': 'Let your mood pick the move',
                'strategy': 'Aggressive? Rock. Calm? Paper. Sharp? Scissors.'
            }
        ]
        
    def analyze_player_patterns(self, human_history: List[str], robot_history: List[str], 
                              result_history: List[str], change_points: List[Dict]) -> Dict:
        """Analyze player patterns and generate insights"""
        if len(human_history) < 3:
            return {'insights': [], 'pattern_type': 'insufficient_data'}
        
        insights = {}
        
        # Basic statistics
        move_counts = Counter(human_history)
        total_moves = len(human_history)
        insights['move_distribution'] = {move: count/total_moves for move, count in move_counts.items()}
        
        # Predictability analysis
        insights['predictability'] = self._calculate_predictability(human_history)
        
        # Pattern detection
        insights['pattern_type'] = self._detect_pattern_type(human_history)
        
        # Recent performance
        insights['recent_performance'] = self._analyze_recent_performance(result_history)
        
        # Adaptation analysis
        insights['adaptation_info'] = self._analyze_adaptation(change_points, total_moves)
        
        # Weaknesses
        insights['weaknesses'] = self._identify_weaknesses(human_history, robot_history, result_history)
        
        return insights
    
    def generate_tips(self, human_history: List[str], robot_history: List[str], 
                     result_history: List[str], change_points: List[Dict], 
                     current_strategy: str = 'unknown') -> Dict:
        """Generate 3-5 actionable coaching tips deterministically based on game state"""
        if len(human_history) < 5:
            return {
                'tips': [
                    "Keep playing! I need at least 5 moves to analyze your patterns.",
                    "Try different moves to see how the AI responds.",
                    "Experiment with different strategies early on."
                ],
                'experiments': self.experiments[:2],  # First 2 experiments consistently
                'insights': {}
            }
        
        # Create deterministic seed based on recent game state
        state_components = (
            tuple(human_history[-10:]),  # Last 10 moves
            tuple(result_history[-10:]),  # Last 10 results
            current_strategy,
            len(change_points),
            len(human_history)
        )
        state_hash = hash(state_components)
        
        # Use deterministic pseudo-random selection based on state
        import random
        random.seed(state_hash)
        
        # Analyze patterns
        insights = self.analyze_player_patterns(human_history, robot_history, result_history, change_points)
        
        # Generate specific tips based on analysis
        tips = []
        
        # Predictability tips
        if insights['predictability'] > 0.7:
            tips.extend(random.sample(self.tip_templates['predictability'], 1))
        
        # Pattern-specific tips
        pattern_type = insights['pattern_type']
        if pattern_type in self.tip_templates:
            pattern_tips = self.tip_templates[pattern_type].copy()
            if pattern_type == 'biased':
                # Find most common move for bias tips
                most_common = max(insights['move_distribution'], key=insights['move_distribution'].get)
                pattern_tips = [tip.format(move=most_common) for tip in pattern_tips]
            tips.extend(random.sample(pattern_tips, 1))
        
        # Performance tips
        perf = insights['recent_performance']
        if len(tips) < 3:
            perf_tips = self.tip_templates['performance'].copy()
            perf_tips = [tip.format(
                win_rate=round(perf['win_rate'] * 100, 1),
                tie_rate=round(perf['tie_rate'] * 100, 1)
            ) for tip in perf_tips]
            tips.extend(random.sample(perf_tips, 1))
        
        # Adaptation tips
        if insights['adaptation_info']['recent_changes'] > 0 and len(tips) < 4:
            adapt_tips = self.tip_templates['adaptation'].copy()
            last_change = insights['adaptation_info']['last_change_round']
            adapt_tips = [tip.format(round=last_change) for tip in adapt_tips]
            tips.extend(random.sample(adapt_tips, 1))
        
        # General strategy tips
        if len(tips) < 3:
            if insights['predictability'] < 0.4:
                tips.extend(random.sample(self.tip_templates['randomness'], 1))
            else:
                tips.extend(random.sample(self.tip_templates['anti_strategy'], 1))
        
        # Ensure we have 3-5 tips
        while len(tips) < 3:
            available_tips = self.tip_templates['randomness'] + self.tip_templates['anti_strategy']
            tips.append(random.choice(available_tips))
        
        tips = tips[:5]  # Cap at 5 tips
        
        # Select appropriate experiments deterministically
        experiments = self._select_experiments_deterministic(insights, current_strategy, state_hash)
        
        # Reset random seed to avoid affecting other parts of the system
        random.seed()
        
        return {
            'tips': tips,
            'experiments': experiments,
            'insights': insights
        }
    
    def _calculate_predictability(self, moves: List[str]) -> float:
        """Calculate how predictable the move sequence is (0-1 scale)"""
        if len(moves) < 3:
            return 0.0
        
        # Check for repeating patterns
        patterns = 0
        total_checks = 0
        
        # Check for immediate repetitions
        for i in range(1, len(moves)):
            total_checks += 1
            if moves[i] == moves[i-1]:
                patterns += 1
        
        # Check for 2-move cycles
        for i in range(2, len(moves)):
            total_checks += 1
            if moves[i] == moves[i-2]:
                patterns += 1
        
        # Check for 3-move cycles  
        for i in range(3, len(moves)):
            total_checks += 1
            if moves[i] == moves[i-3]:
                patterns += 1
        
        return patterns / total_checks if total_checks > 0 else 0.0
    
    def _detect_pattern_type(self, moves: List[str]) -> str:
        """Detect the primary pattern type in recent moves"""
        if len(moves) < 5:
            return 'insufficient_data'
        
        recent = moves[-10:]  # Look at last 10 moves
        
        # Check for repetition
        if len(set(recent[-3:])) == 1:
            return 'repeater'
        
        # Check for cycling
        cycle_patterns = [
            ['stone', 'paper', 'scissor'],
            ['paper', 'scissor', 'stone'], 
            ['scissor', 'stone', 'paper'],
            ['stone', 'scissor', 'paper'],  # reverse
            ['paper', 'stone', 'scissor'],  # reverse
            ['scissor', 'paper', 'stone']   # reverse
        ]
        
        for pattern in cycle_patterns:
            if len(recent) >= 6:
                if (recent[-3:] == pattern and recent[-6:-3] == pattern):
                    return 'cycler'
        
        # Check for bias
        move_counts = Counter(recent)
        if move_counts and max(move_counts.values()) / len(recent) > 0.6:
            return 'biased'
        
        # Check for balanced play
        if len(set(recent)) == 3 and all(count >= 2 for count in move_counts.values()):
            return 'balanced'
        
        return 'mixed'
    
    def _analyze_recent_performance(self, results: List[str]) -> Dict:
        """Analyze recent win/loss performance"""
        if not results:
            return {'win_rate': 0.0, 'tie_rate': 0.0, 'loss_rate': 0.0}
        
        recent = results[-10:]  # Last 10 games
        total = len(recent)
        
        wins = recent.count('human')
        ties = recent.count('tie')
        losses = recent.count('robot')
        
        return {
            'win_rate': wins / total,
            'tie_rate': ties / total,
            'loss_rate': losses / total,
            'total_recent': total
        }
    
    def _analyze_adaptation(self, change_points: List[Dict], total_moves: int) -> Dict:
        """Analyze strategy adaptation patterns"""
        recent_changes = len([cp for cp in change_points 
                            if cp.get('round', 0) > total_moves - 10])
        
        last_change = 0
        if change_points:
            last_change = max([cp.get('round', 0) for cp in change_points])
        
        return {
            'total_changes': len(change_points),
            'recent_changes': recent_changes,
            'last_change_round': last_change,
            'adaptation_rate': len(change_points) / max(total_moves, 1)
        }
    
    def _identify_weaknesses(self, human_moves: List[str], robot_moves: List[str], 
                           results: List[str]) -> List[str]:
        """Identify specific weaknesses in play"""
        weaknesses = []
        
        if len(human_moves) < 5:
            return weaknesses
        
        recent_human = human_moves[-10:]
        recent_robot = robot_moves[-10:]
        recent_results = results[-10:]
        
        # Check if AI is successfully countering
        ai_wins = recent_results.count('robot')
        if ai_wins > len(recent_results) * 0.6:
            weaknesses.append('AI is successfully predicting your moves')
        
        # Check for exploitable patterns
        if len(set(recent_human[-3:])) == 1:
            weaknesses.append('Repeating the same move too often')
        
        # Check if AI is adapting its strategy
        ai_moves = Counter(recent_robot)
        if ai_moves and max(ai_moves.values()) > len(recent_robot) * 0.7:
            weaknesses.append('AI has found a counter-strategy')
        
        return weaknesses
    
    def _select_experiments(self, insights: Dict, current_strategy: str) -> List[Dict]:
        """Select 2-3 appropriate experiments based on current patterns"""
        pattern_type = insights['pattern_type']
        predictability = insights['predictability']
        
        # Filter experiments based on current situation
        suitable_experiments = []
        
        if pattern_type == 'repeater':
            suitable_experiments.extend([
                exp for exp in self.experiments 
                if exp['name'] in ['Random Play', 'Counter-Cycle', 'Double Repeat']
            ])
        elif pattern_type == 'cycler':
            suitable_experiments.extend([
                exp for exp in self.experiments
                if exp['name'] in ['Anti-Mirror', 'Random Play', 'Frequency Counter']
            ])
        elif predictability > 0.6:
            suitable_experiments.extend([
                exp for exp in self.experiments
                if exp['name'] in ['Random Play', 'Emotional Play', 'Mirror Strategy']
            ])
        else:
            # Good unpredictable play - suggest advanced techniques
            suitable_experiments.extend([
                exp for exp in self.experiments
                if exp['name'] in ['Frequency Counter', 'Anti-Mirror', 'Emotional Play']
            ])
        
        # Ensure we have enough experiments
        if len(suitable_experiments) < 3:
            remaining = [exp for exp in self.experiments if exp not in suitable_experiments]
            suitable_experiments.extend(random.sample(remaining, 3 - len(suitable_experiments)))
        
        return random.sample(suitable_experiments, min(3, len(suitable_experiments)))
    
    def _select_experiments_deterministic(self, insights: Dict, current_strategy: str, state_hash: int) -> List[Dict]:
        """Select 2-3 appropriate experiments deterministically based on current patterns"""
        import random
        random.seed(state_hash)
        
        pattern_type = insights['pattern_type']
        predictability = insights['predictability']
        
        # Filter experiments based on current situation
        suitable_experiments = []
        
        if pattern_type == 'repeater':
            suitable_experiments.extend([
                exp for exp in self.experiments 
                if exp['name'] in ['Random Play', 'Counter-Cycle', 'Double Repeat']
            ])
        elif pattern_type == 'cycler':
            suitable_experiments.extend([
                exp for exp in self.experiments
                if exp['name'] in ['Anti-Mirror', 'Random Play', 'Frequency Counter']
            ])
        elif predictability > 0.6:
            suitable_experiments.extend([
                exp for exp in self.experiments
                if exp['name'] in ['Random Play', 'Emotional Play', 'Mirror Strategy']
            ])
        else:
            # Good unpredictable play - suggest advanced techniques
            suitable_experiments.extend([
                exp for exp in self.experiments
                if exp['name'] in ['Frequency Counter', 'Anti-Mirror', 'Emotional Play']
            ])
        
        # Ensure we have enough experiments
        if len(suitable_experiments) < 3:
            remaining = [exp for exp in self.experiments if exp not in suitable_experiments]
            suitable_experiments.extend(random.sample(remaining, 3 - len(suitable_experiments)))
        
        result = random.sample(suitable_experiments, min(3, len(suitable_experiments)))
        random.seed()  # Reset seed
        return result