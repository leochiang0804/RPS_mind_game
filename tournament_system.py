"""
Tournament System for Rock Paper Scissors
Handles player rankings, match history, and tournament brackets
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid

class Player:
    def __init__(self, name: str, player_id: Optional[str] = None):
        self.id = player_id or str(uuid.uuid4())
        self.name = name
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.total_games = 0
        self.elo_rating = 1200.0  # Float for ELO calculations
        self.created_at = datetime.now().isoformat()
        
    def get_win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return (self.wins / self.total_games) * 100
    
    def update_stats(self, result: str):
        """Update player stats after a match"""
        if result == 'win':
            self.wins += 1
        elif result == 'loss':
            self.losses += 1
        else:  # tie
            self.ties += 1
        self.total_games += 1
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'wins': self.wins,
            'losses': self.losses,
            'ties': self.ties,
            'total_games': self.total_games,
            'win_rate': self.get_win_rate(),
            'elo_rating': self.elo_rating,
            'created_at': self.created_at
        }

class Match:
    def __init__(self, player1_id: str, player2_id: str, match_id: Optional[str] = None):
        self.id = match_id or str(uuid.uuid4())
        self.player1_id = player1_id
        self.player2_id = player2_id
        self.player1_moves = []
        self.player2_moves = []
        self.results = []
        self.winner_id = None
        self.status = 'pending'  # pending, in_progress, completed
        self.created_at = datetime.now().isoformat()
        self.completed_at = None
        
    def add_round(self, p1_move: str, p2_move: str) -> str:
        """Add a round to the match and return result"""
        self.player1_moves.append(p1_move)
        self.player2_moves.append(p2_move)
        
        # Determine round winner
        result = self._get_round_result(p1_move, p2_move)
        self.results.append(result)
        
        if self.status == 'pending':
            self.status = 'in_progress'
            
        return result
    
    def _get_round_result(self, move1: str, move2: str) -> str:
        """Determine winner of a single round"""
        if move1 == move2:
            return 'tie'
        elif (move1 == 'paper' and move2 == 'stone') or \
             (move1 == 'stone' and move2 == 'scissor') or \
             (move1 == 'scissor' and move2 == 'paper'):
            return 'player1'
        else:
            return 'player2'
    
    def complete_match(self, best_of: int = 5) -> str:
        """Complete the match and determine overall winner"""
        p1_wins = self.results.count('player1')
        p2_wins = self.results.count('player2')
        
        rounds_to_win = (best_of + 1) // 2
        
        if p1_wins >= rounds_to_win:
            self.winner_id = self.player1_id
            self.status = 'completed'
            self.completed_at = datetime.now().isoformat()
            return 'player1'
        elif p2_wins >= rounds_to_win:
            self.winner_id = self.player2_id
            self.status = 'completed'
            self.completed_at = datetime.now().isoformat()
            return 'player2'
        else:
            return 'ongoing'
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'player1_id': self.player1_id,
            'player2_id': self.player2_id,
            'player1_moves': self.player1_moves,
            'player2_moves': self.player2_moves,
            'results': self.results,
            'winner_id': self.winner_id,
            'status': self.status,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'score': f"{self.results.count('player1')}-{self.results.count('player2')}"
        }

class Tournament:
    def __init__(self, name: str, tournament_id: Optional[str] = None):
        self.id = tournament_id or str(uuid.uuid4())
        self.name = name
        self.players = []
        self.matches = []
        self.bracket = {}
        self.status = 'registration'  # registration, in_progress, completed
        self.created_at = datetime.now().isoformat()
        self.winner_id = None
        
    def add_player(self, player: Player):
        """Add player to tournament"""
        if self.status == 'registration':
            self.players.append(player)
            return True
        return False
    
    def start_tournament(self):
        """Start the tournament and generate bracket"""
        if len(self.players) < 2:
            return False
            
        self.status = 'in_progress'
        self._generate_bracket()
        return True
    
    def _generate_bracket(self):
        """Generate tournament bracket"""
        import random
        players_copy = self.players.copy()
        random.shuffle(players_copy)
        
        round_num = 1
        current_round = []
        
        # Create first round matches
        for i in range(0, len(players_copy), 2):
            if i + 1 < len(players_copy):
                match = Match(players_copy[i].id, players_copy[i + 1].id)
                current_round.append(match)
                self.matches.append(match)
        
        self.bracket[f'round_{round_num}'] = [m.id for m in current_round]
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'players': [p.to_dict() for p in self.players],
            'matches': [m.to_dict() for m in self.matches],
            'bracket': self.bracket,
            'status': self.status,
            'created_at': self.created_at,
            'winner_id': self.winner_id
        }

class TournamentSystem:
    def __init__(self, data_file: str = 'tournament_data.json'):
        self.data_file = data_file
        self.players = {}
        self.matches = {}
        self.tournaments = {}
        self.leaderboard = []
        self.load_data()
    
    def load_data(self):
        """Load tournament data from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    
                # Reconstruct players
                for player_data in data.get('players', []):
                    player = Player(player_data['name'], player_data['id'])
                    player.wins = player_data['wins']
                    player.losses = player_data['losses']
                    player.ties = player_data['ties']
                    player.total_games = player_data['total_games']
                    player.elo_rating = player_data['elo_rating']
                    player.created_at = player_data['created_at']
                    self.players[player.id] = player
                
                # Reconstruct matches
                for match_data in data.get('matches', []):
                    match = Match(match_data['player1_id'], match_data['player2_id'], match_data['id'])
                    match.player1_moves = match_data['player1_moves']
                    match.player2_moves = match_data['player2_moves']
                    match.results = match_data['results']
                    match.winner_id = match_data['winner_id']
                    match.status = match_data['status']
                    match.created_at = match_data['created_at']
                    match.completed_at = match_data['completed_at']
                    self.matches[match.id] = match
                    
            except Exception as e:
                print(f"Error loading tournament data: {e}")
    
    def save_data(self):
        """Save tournament data to file"""
        data = {
            'players': [p.to_dict() for p in self.players.values()],
            'matches': [m.to_dict() for m in self.matches.values()],
            'tournaments': [t.to_dict() for t in self.tournaments.values()]
        }
        
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving tournament data: {e}")
    
    def create_player(self, name: str) -> Player:
        """Create a new player"""
        player = Player(name)
        self.players[player.id] = player
        self.save_data()
        return player
    
    def get_player_by_name(self, name: str) -> Optional[Player]:
        """Get player by name"""
        for player in self.players.values():
            if player.name.lower() == name.lower():
                return player
        return None
    
    def create_match(self, player1_id: str, player2_id: str) -> Match:
        """Create a new match between two players"""
        match = Match(player1_id, player2_id)
        self.matches[match.id] = match
        self.save_data()
        return match
    
    def update_elo_ratings(self, winner: Player, loser: Player, k_factor: int = 32):
        """Update ELO ratings after a match"""
        expected_win_prob = 1 / (1 + 10**((loser.elo_rating - winner.elo_rating) / 400))
        
        winner.elo_rating += k_factor * (1 - expected_win_prob)
        loser.elo_rating += k_factor * (0 - (1 - expected_win_prob))
        
        # Ensure ratings don't go below 100
        winner.elo_rating = max(100.0, winner.elo_rating)
        loser.elo_rating = max(100.0, loser.elo_rating)
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get top players by ELO rating"""
        sorted_players = sorted(self.players.values(), 
                              key=lambda p: p.elo_rating, reverse=True)
        return [p.to_dict() for p in sorted_players[:limit]]
    
    def get_player_stats(self, player_id: str) -> Dict:
        """Get detailed stats for a player"""
        player = self.players.get(player_id)
        if not player:
            return {}
            
        # Get recent matches
        recent_matches = []
        for match in self.matches.values():
            if player_id in [match.player1_id, match.player2_id] and match.status == 'completed':
                recent_matches.append(match.to_dict())
        
        recent_matches.sort(key=lambda m: m['completed_at'], reverse=True)
        
        return {
            'player': player.to_dict(),
            'recent_matches': recent_matches[:10],
            'rank': self._get_player_rank(player_id)
        }
    
    def _get_player_rank(self, player_id: str) -> int:
        """Get player's current rank"""
        leaderboard = self.get_leaderboard(len(self.players))
        for i, player_data in enumerate(leaderboard):
            if player_data['id'] == player_id:
                return i + 1
        return len(self.players)