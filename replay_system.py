"""
Game Replay & Analysis System for Rock Paper Scissors
Provides comprehensive replay functionality with move-by-move analysis
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter


class GameReplay:
    """Individual game session replay data"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now().isoformat()
        self.moves: List[Dict[str, Any]] = []
        self.metadata = {
            'difficulty': 'enhanced',
            'strategy': 'balanced', 
            'personality': 'neutral',
            'total_rounds': 0,
            'human_wins': 0,
            'robot_wins': 0,
            'ties': 0,
            'final_score': {'human': 0, 'robot': 0, 'ties': 0}
        }
        self.annotations: Dict[int, str] = {}  # round_number -> annotation
        self.analysis_data: Dict[str, Any] = {}
    
    def add_move(self, round_number: int, human_move: str, robot_move: str, 
                 result: str, confidence: float = 0.0, strategy_used: str = '',
                 analysis: Optional[Dict[str, Any]] = None):
        """Add a move to the replay with analysis data"""
        move_data = {
            'round': round_number,
            'timestamp': datetime.now().isoformat(),
            'human_move': human_move,
            'robot_move': robot_move,
            'result': result,  # 'human', 'robot', 'tie'
            'robot_confidence': confidence,
            'strategy_used': strategy_used,
            'analysis': analysis or {}
        }
        
        self.moves.append(move_data)
        self.metadata['total_rounds'] = len(self.moves)
        
        # Update score
        if result == 'human':
            self.metadata['human_wins'] += 1
        elif result == 'robot':
            self.metadata['robot_wins'] += 1
        else:
            self.metadata['ties'] += 1
        
        # Update final score
        self.metadata['final_score'] = {
            'human': self.metadata['human_wins'],
            'robot': self.metadata['robot_wins'],
            'ties': self.metadata['ties']
        }
    
    def add_annotation(self, round_number: int, annotation: str):
        """Add an annotation to a specific round"""
        self.annotations[round_number] = annotation
    
    def get_move_at_round(self, round_number: int) -> Optional[Dict[str, Any]]:
        """Get move data for a specific round"""
        for move in self.moves:
            if move['round'] == round_number:
                return move
        return None
    
    def get_moves_range(self, start_round: int, end_round: int) -> List[Dict[str, Any]]:
        """Get moves within a range of rounds"""
        return [move for move in self.moves 
                if start_round <= move['round'] <= end_round]
    
    def get_strategy_changes(self) -> List[Dict[str, Any]]:
        """Identify rounds where strategy changed"""
        changes = []
        prev_strategy = None
        
        for move in self.moves:
            current_strategy = move.get('strategy_used', '')
            if prev_strategy and current_strategy != prev_strategy:
                changes.append({
                    'round': move['round'],
                    'from_strategy': prev_strategy,
                    'to_strategy': current_strategy,
                    'timestamp': move['timestamp']
                })
            prev_strategy = current_strategy
        
        return changes
    
    def get_performance_trends(self) -> Dict[str, List[float]]:
        """Calculate performance trends over time"""
        window_size = 5
        trends = {
            'human_win_rate': [],
            'robot_confidence': [],
            'rounds': []
        }
        
        for i in range(len(self.moves)):
            start_idx = max(0, i - window_size + 1)
            window_moves = self.moves[start_idx:i+1]
            
            # Calculate win rate in window
            human_wins = sum(1 for m in window_moves if m['result'] == 'human')
            win_rate = human_wins / len(window_moves) if window_moves else 0
            
            # Average confidence in window
            avg_confidence = sum(m['robot_confidence'] for m in window_moves) / len(window_moves) if window_moves else 0
            
            trends['human_win_rate'].append(win_rate)
            trends['robot_confidence'].append(avg_confidence)
            trends['rounds'].append(i + 1)
        
        return trends
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert replay to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'metadata': self.metadata,
            'moves': self.moves,
            'annotations': self.annotations,
            'analysis_data': self.analysis_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameReplay':
        """Create replay from dictionary"""
        replay = cls(data['session_id'])
        replay.created_at = data['created_at']
        replay.metadata = data['metadata']
        replay.moves = data['moves']
        replay.annotations = data.get('annotations', {})
        replay.analysis_data = data.get('analysis_data', {})
        return replay


class ReplayManager:
    """Manages game replay storage and retrieval"""
    
    def __init__(self, storage_dir: str = "replays"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_replay(self, replay: GameReplay) -> str:
        """Save a replay to storage"""
        filename = f"replay_{replay.session_id}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(replay.to_dict(), f, indent=2)
        
        return filepath
    
    def load_replay(self, session_id: str) -> Optional[GameReplay]:
        """Load a replay from storage"""
        filename = f"replay_{session_id}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return GameReplay.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading replay {session_id}: {e}")
            return None
    
    def list_replays(self) -> List[Dict[str, Any]]:
        """List all available replays with metadata"""
        replays = []
        
        for filename in os.listdir(self.storage_dir):
            if filename.startswith('replay_') and filename.endswith('.json'):
                session_id = filename[7:-5]  # Remove 'replay_' and '.json'
                replay = self.load_replay(session_id)
                
                if replay:
                    replays.append({
                        'session_id': replay.session_id,
                        'created_at': replay.created_at,
                        'total_rounds': replay.metadata['total_rounds'],
                        'final_score': replay.metadata['final_score'],
                        'difficulty': replay.metadata.get('difficulty', 'unknown'),
                        'strategy': replay.metadata.get('strategy', 'unknown'),
                        'personality': replay.metadata.get('personality', 'unknown')
                    })
        
        # Sort by creation date (newest first)
        replays.sort(key=lambda x: x['created_at'], reverse=True)
        return replays
    
    def delete_replay(self, session_id: str) -> bool:
        """Delete a replay from storage"""
        filename = f"replay_{session_id}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
    
    def export_replay_csv(self, session_id: str) -> Optional[str]:
        """Export replay to CSV format"""
        replay = self.load_replay(session_id)
        if not replay:
            return None
        
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'Round', 'Timestamp', 'Human Move', 'Robot Move', 'Result', 
            'Robot Confidence', 'Strategy Used', 'Annotation'
        ])
        
        # Data rows
        for move in replay.moves:
            annotation = replay.annotations.get(move['round'], '')
            writer.writerow([
                move['round'],
                move['timestamp'],
                move['human_move'],
                move['robot_move'],
                move['result'],
                move['robot_confidence'],
                move.get('strategy_used', ''),
                annotation
            ])
        
        return output.getvalue()


class ReplayAnalyzer:
    """Analyzes replay data to provide insights"""
    
    def __init__(self):
        self.patterns = {
            'repetitive': 'Playing the same move repeatedly',
            'cyclical': 'Following a cyclical pattern (rock->paper->scissors)',
            'reactive': 'Reacting to opponent\'s previous move',
            'counter_reactive': 'Trying to counter opponent\'s reactions',
            'random': 'Playing randomly with no clear pattern',
            'frequency_based': 'Favoring certain moves over others'
        }
    
    def analyze_replay(self, replay: GameReplay) -> Dict[str, Any]:
        """Comprehensive analysis of a replay"""
        moves = replay.moves
        if len(moves) < 3:
            return {'error': 'Not enough moves for analysis'}
        
        analysis = {
            'session_info': {
                'total_rounds': len(moves),
                'duration': self._calculate_duration(moves),
                'final_score': replay.metadata['final_score']
            },
            'human_patterns': self._analyze_human_patterns(moves),
            'robot_performance': self._analyze_robot_performance(moves),
            'strategy_effectiveness': self._analyze_strategy_effectiveness(moves),
            'key_moments': self._identify_key_moments(moves),
            'improvement_suggestions': self._generate_suggestions(moves),
            'statistical_summary': self._generate_statistics(moves)
        }
        
        return analysis
    
    def _calculate_duration(self, moves: List[Dict[str, Any]]) -> str:
        """Calculate game duration"""
        if len(moves) < 2:
            return "N/A"
        
        start_time = datetime.fromisoformat(moves[0]['timestamp'])
        end_time = datetime.fromisoformat(moves[-1]['timestamp'])
        duration = end_time - start_time
        
        total_seconds = int(duration.total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        
        return f"{minutes}m {seconds}s"
    
    def _analyze_human_patterns(self, moves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze human playing patterns"""
        human_moves = [move['human_move'] for move in moves]
        
        # Move frequency
        from collections import Counter
        move_counts = Counter(human_moves)
        
        # Pattern detection
        patterns = []
        
        # Check for repetition
        if len(set(human_moves[-3:])) == 1 and len(human_moves) >= 3:
            patterns.append('repetitive')
        
        # Check for cycles
        if self._is_cyclical(human_moves):
            patterns.append('cyclical')
        
        # Check for reactivity to robot
        reactive_count = 0
        for i in range(1, len(moves)):
            prev_robot = moves[i-1]['robot_move']
            current_human = moves[i]['human_move']
            if self._beats(current_human, prev_robot):
                reactive_count += 1
        
        if reactive_count / len(moves) > 0.6:
            patterns.append('reactive')
        
        return {
            'move_distribution': dict(move_counts),
            'most_common_move': move_counts.most_common(1)[0][0],
            'detected_patterns': patterns,
            'predictability_score': self._calculate_predictability(human_moves),
            'pattern_descriptions': [self.patterns.get(p, p) for p in patterns]
        }
    
    def _analyze_robot_performance(self, moves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze robot performance metrics"""
        robot_wins = sum(1 for move in moves if move['result'] == 'robot')
        total_moves = len(moves)
        
        confidences = [move['robot_confidence'] for move in moves]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Strategy usage
        strategies = [move.get('strategy_used', 'unknown') for move in moves]
        from collections import Counter
        strategy_counts = Counter(strategies)
        
        return {
            'win_rate': robot_wins / total_moves,
            'average_confidence': avg_confidence,
            'confidence_trend': 'increasing' if confidences[-1] > confidences[0] else 'decreasing',
            'strategy_distribution': dict(strategy_counts),
            'peak_confidence': max(confidences),
            'low_confidence_rounds': [i+1 for i, c in enumerate(confidences) if c < 0.4]
        }
    
    def _analyze_strategy_effectiveness(self, moves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze effectiveness of different strategies"""
        strategy_performance = {}
        
        for move in moves:
            strategy = move.get('strategy_used', 'unknown')
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {'wins': 0, 'total': 0}
            
            strategy_performance[strategy]['total'] += 1
            if move['result'] == 'robot':
                strategy_performance[strategy]['wins'] += 1
        
        # Calculate win rates
        for strategy in strategy_performance:
            total = strategy_performance[strategy]['total']
            wins = strategy_performance[strategy]['wins']
            strategy_performance[strategy]['win_rate'] = wins / total if total > 0 else 0
        
        return strategy_performance
    
    def _identify_key_moments(self, moves: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key moments in the game"""
        key_moments = []
        
        # Winning streaks
        current_streak = 0
        streak_player = None
        
        for i, move in enumerate(moves):
            if move['result'] in ['human', 'robot']:
                if move['result'] == streak_player:
                    current_streak += 1
                else:
                    if current_streak >= 3 and streak_player:
                        key_moments.append({
                            'type': 'streak_end',
                            'round': i,
                            'description': f"{streak_player.title()} ended {current_streak}-game winning streak",
                            'significance': 'high' if current_streak >= 5 else 'medium'
                        })
                    
                    streak_player = move['result']
                    current_streak = 1
        
        # High confidence predictions that failed
        for i, move in enumerate(moves):
            if move['robot_confidence'] > 0.8 and move['result'] == 'human':
                key_moments.append({
                    'type': 'confident_failure',
                    'round': i + 1,
                    'description': f"Robot was very confident ({move['robot_confidence']:.1%}) but lost",
                    'significance': 'medium'
                })
        
        return key_moments
    
    def _generate_suggestions(self, moves: List[Dict[str, Any]]) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        human_moves = [move['human_move'] for move in moves]
        move_counts = Counter(human_moves)
        
        # Check for over-reliance on one move
        most_common_count = move_counts.most_common(1)[0][1]
        if most_common_count / len(human_moves) > 0.5:
            suggestions.append("Try varying your moves more - you're being too predictable")
        
        # Check for reactive play
        reactive_count = 0
        for i in range(1, len(moves)):
            prev_robot = moves[i-1]['robot_move']
            current_human = moves[i]['human_move']
            if self._beats(current_human, prev_robot):
                reactive_count += 1
        
        if reactive_count / len(moves) > 0.6:
            suggestions.append("You're being too reactive - try ignoring the robot's previous move")
        
        # Check recent performance
        recent_moves = moves[-5:] if len(moves) >= 5 else moves
        recent_wins = sum(1 for move in recent_moves if move['result'] == 'human')
        
        if recent_wins / len(recent_moves) < 0.3:
            suggestions.append("Try a completely different strategy - mix up your timing and choices")
        
        return suggestions
    
    def _generate_statistics(self, moves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed statistics"""
        total_moves = len(moves)
        human_wins = sum(1 for move in moves if move['result'] == 'human')
        robot_wins = sum(1 for move in moves if move['result'] == 'robot')
        ties = sum(1 for move in moves if move['result'] == 'tie')
        
        return {
            'total_rounds': total_moves,
            'human_win_rate': human_wins / total_moves,
            'robot_win_rate': robot_wins / total_moves,
            'tie_rate': ties / total_moves,
            'longest_human_streak': self._longest_streak(moves, 'human'),
            'longest_robot_streak': self._longest_streak(moves, 'robot'),
            'average_robot_confidence': sum(move['robot_confidence'] for move in moves) / total_moves
        }
    
    def _is_cyclical(self, moves: List[str]) -> bool:
        """Check if moves follow a cyclical pattern"""
        if len(moves) < 6:
            return False
        
        cycle = ['paper', 'stone', 'scissor']
        cycle_positions = {move: i for i, move in enumerate(cycle)}
        
        # Check last 6 moves for cycle
        recent = moves[-6:]
        for i in range(len(recent) - 2):
            if recent[i] in cycle_positions and recent[i+1] in cycle_positions:
                expected_next = cycle[(cycle_positions[recent[i]] + 1) % 3]
                if recent[i+1] != expected_next:
                    return False
        
        return True
    
    def _beats(self, move1: str, move2: str) -> bool:
        """Check if move1 beats move2"""
        winning_moves = {
            'paper': 'stone',
            'stone': 'scissor', 
            'scissor': 'paper'
        }
        return winning_moves.get(move1) == move2
    
    def _calculate_predictability(self, moves: List[str]) -> float:
        """Calculate how predictable the move sequence is"""
        if len(moves) < 3:
            return 0.0
        
        # Simple predictability based on repetition and patterns
        from collections import Counter
        move_counts = Counter(moves)
        entropy = -sum((count/len(moves)) * (count/len(moves)) for count in move_counts.values())
        max_entropy = -(1/3) * (1/3) * 3  # Maximum entropy for 3 equally likely moves
        
        return 1 - (entropy / max_entropy) if max_entropy != 0 else 0
    
    def _longest_streak(self, moves: List[Dict[str, Any]], player: str) -> int:
        """Find longest winning streak for a player"""
        longest = 0
        current = 0
        
        for move in moves:
            if move['result'] == player:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        
        return longest


# Global instance
replay_manager = ReplayManager()
replay_analyzer = ReplayAnalyzer()

def get_replay_manager():
    """Get the global replay manager instance"""
    return replay_manager

def get_replay_analyzer():
    """Get the global replay analyzer instance"""
    return replay_analyzer