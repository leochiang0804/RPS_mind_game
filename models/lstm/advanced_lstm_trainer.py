"""
Advanced LSTM Training System for Rock-Paper-Scissors
Based on lstm_rps_setup.md recommendations for optimal human prediction

Features:
- Rich input features (moves, outcomes, streaks, pattern detection)
- Pretrain + online adaptation strategy
- Change-point detection with adaptive learning
- Ensemble with n-gram models
- Proper hyperparameters per game length
"""

import os
import sys
import json
import time
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from move_mapping import MOVES, normalize_move, get_counter_move

@dataclass
class GameConfig:
    """Configuration for different game lengths"""
    match_length: int
    hidden_size: int
    buffer_size: int
    sequence_length: int
    updates_per_turn: int
    update_frequency: int  # every N turns
    exploration_rate: float = 0.05

# Hyperparameters from the setup guide
GAME_CONFIGS = {
    25: GameConfig(25, 32, 20, 12, 1, 1, 0.05),
    50: GameConfig(50, 32, 20, 12, 2, 1, 0.05),
    100: GameConfig(100, 32, 40, 16, 2, 2, 0.05),
    250: GameConfig(250, 48, 40, 16, 3, 2, 0.05),
    500: GameConfig(500, 48, 60, 20, 3, 3, 0.05),
    1000: GameConfig(1000, 48, 60, 20, 3, 3, 0.05),
}

class AdvancedLSTM(nn.Module):
    """
    Enhanced LSTM with rich input features following setup guide
    Input features: human_move(3) + robot_move(3) + outcome(3) + streaks(2) + time_since_switch(1) = 12 dims
    """
    
    def __init__(self, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.1, 
                 embed_dim: Optional[int] = None):
        super().__init__()
        
        # Input dimensions: 12 (as per setup guide)
        self.input_dim = 12
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Optional embedding layer
        if embed_dim:
            self.embedding = nn.Linear(self.input_dim, embed_dim)
            lstm_input_dim = embed_dim
        else:
            self.embedding = None
            lstm_input_dim = self.input_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Dropout on LSTM output
        self.dropout = nn.Dropout(dropout)
        
        # Output head with softmax
        self.head = nn.Linear(hidden_size, 3)
        
        # Move mappings
        self.move_to_idx = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.idx_to_move = {0: 'rock', 1: 'paper', 2: 'scissors'}
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """Forward pass"""
        batch_size, seq_len, _ = x.shape
        
        # Optional embedding
        if self.embedding:
            x = self.embedding(x)
        
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output head (only last timestep)
        logits = self.head(lstm_out[:, -1, :])  # Shape: (batch_size, 3)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden state"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)

class FeatureExtractor:
    """Extract rich features from game history following setup guide"""
    
    def __init__(self):
        self.move_to_idx = {'rock': 0, 'paper': 1, 'scissors': 2}
        
    def extract_features(self, human_moves: List[str], robot_moves: List[str], 
                        outcomes: List[str]) -> torch.Tensor:
        """
        Extract features for each timestep:
        - One-hot human move (3)
        - One-hot robot move (3) 
        - One-hot outcome (3): win/lose/draw
        - Streak features (2): win_streak, lose_streak (clipped to ±3)
        - Time since pattern switch (1): clipped
        """
        if len(human_moves) == 0:
            return torch.zeros(1, 12)
            
        features = []
        win_streak = 0
        lose_streak = 0
        last_pattern_change = 0
        
        for i in range(len(human_moves)):
            # One-hot human move
            human_onehot = [0, 0, 0]
            human_onehot[self.move_to_idx[human_moves[i]]] = 1
            
            # One-hot robot move
            robot_onehot = [0, 0, 0]
            robot_onehot[self.move_to_idx[robot_moves[i]]] = 1
            
            # One-hot outcome
            outcome_onehot = [0, 0, 0]
            if outcomes[i] == 'win':
                outcome_onehot[0] = 1
                win_streak += 1
                lose_streak = 0
            elif outcomes[i] == 'lose':
                outcome_onehot[1] = 1
                lose_streak += 1
                win_streak = 0
            else:  # draw
                outcome_onehot[2] = 1
                win_streak = 0
                lose_streak = 0
            
            # Clip streaks to ±3
            win_streak_clipped = min(max(win_streak, -3), 3) / 3.0
            lose_streak_clipped = min(max(lose_streak, -3), 3) / 3.0
            
            # Detect pattern changes (simple heuristic)
            if i >= 3:
                recent_moves = human_moves[max(0, i-3):i+1]
                if len(set(recent_moves)) > len(set(human_moves[max(0, i-6):i-3])):
                    last_pattern_change = 0
                else:
                    last_pattern_change += 1
            
            # Time since switch (clipped and normalized)
            time_since_switch = min(last_pattern_change, 10) / 10.0
            
            # Combine all features
            timestep_features = (
                human_onehot + robot_onehot + outcome_onehot + 
                [win_streak_clipped, lose_streak_clipped, time_since_switch]
            )
            features.append(timestep_features)
        
        return torch.tensor(features, dtype=torch.float32)

class NGramPredictor:
    """Lightweight n-gram frequency model for ensemble"""
    
    def __init__(self, decay_rate: float = 0.9):
        self.decay_rate = decay_rate
        self.unigram_counts = defaultdict(float)
        self.bigram_counts = defaultdict(lambda: defaultdict(float))
        self.trigram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.total_count = 0
        
    def update(self, moves: List[str]):
        """Update n-gram counts with exponential decay"""
        # Apply decay
        self.total_count *= self.decay_rate
        for move in self.unigram_counts:
            self.unigram_counts[move] *= self.decay_rate
        
        for prev_move in self.bigram_counts:
            for move in self.bigram_counts[prev_move]:
                self.bigram_counts[prev_move][move] *= self.decay_rate
                
        for prev_prev in self.trigram_counts:
            for prev_move in self.trigram_counts[prev_prev]:
                for move in self.trigram_counts[prev_prev][prev_move]:
                    self.trigram_counts[prev_prev][prev_move][move] *= self.decay_rate
        
        # Add new observations
        for i, move in enumerate(moves):
            self.unigram_counts[move] += 1
            self.total_count += 1
            
            if i > 0:
                prev_move = moves[i-1]
                self.bigram_counts[prev_move][move] += 1
                
            if i > 1:
                prev_prev = moves[i-2]
                prev_move = moves[i-1]
                self.trigram_counts[prev_prev][prev_move][move] += 1
    
    def predict(self, recent_moves: List[str]) -> Dict[str, float]:
        """Predict next move probabilities"""
        probs = {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
        
        if len(recent_moves) == 0:
            return probs
            
        # Try trigram first
        if len(recent_moves) >= 2:
            prev_prev = recent_moves[-2]
            prev_move = recent_moves[-1]
            if prev_prev in self.trigram_counts and prev_move in self.trigram_counts[prev_prev]:
                counts = self.trigram_counts[prev_prev][prev_move]
                total = sum(counts.values())
                if total > 0:
                    return {move: counts[move]/total for move in ['rock', 'paper', 'scissors']}
        
        # Fall back to bigram
        if len(recent_moves) >= 1:
            prev_move = recent_moves[-1]
            if prev_move in self.bigram_counts:
                counts = self.bigram_counts[prev_move]
                total = sum(counts.values())
                if total > 0:
                    return {move: counts[move]/total for move in ['rock', 'paper', 'scissors']}
        
        # Fall back to unigram
        if self.total_count > 0:
            return {move: self.unigram_counts[move]/self.total_count 
                   for move in ['rock', 'paper', 'scissors']}
        
        return probs

class ChangePointDetector:
    """Detect when human changes strategy"""
    
    def __init__(self, window_size: int = 15, threshold: float = 0.25):
        self.window_size = window_size
        self.threshold = threshold
        self.recent_accuracies = deque(maxlen=window_size)
        self.previous_window_acc = None
        
    def update(self, accuracy: float) -> bool:
        """Update with new accuracy, return True if change point detected"""
        self.recent_accuracies.append(accuracy)
        
        if len(self.recent_accuracies) < self.window_size:
            return False
            
        current_acc = np.mean(self.recent_accuracies)
        
        if self.previous_window_acc is not None:
            drop = self.previous_window_acc - current_acc
            if drop > self.threshold:
                self.previous_window_acc = current_acc
                return True
        
        self.previous_window_acc = current_acc
        return False

class SyntheticDataGenerator:
    """Generate synthetic data for pretraining"""
    
    def __init__(self):
        self.moves = ['rock', 'paper', 'scissors']
        
    def generate_patterns(self, num_sequences: int = 1000, seq_length: int = 30) -> List[List[str]]:
        """Generate various human-like patterns"""
        sequences = []
        
        for _ in range(num_sequences):
            pattern_type = random.choice([
                'repeat', 'alternate', 'win_stay_lose_shift', 'biased', 
                'burst', 'periodic', 'anti_frequency', 'random'
            ])
            
            sequence = self._generate_pattern(pattern_type, seq_length)
            sequences.append(sequence)
            
        return sequences
    
    def _generate_pattern(self, pattern_type: str, length: int) -> List[str]:
        """Generate specific pattern type"""
        if pattern_type == 'repeat':
            move = random.choice(self.moves)
            return [move] * length
            
        elif pattern_type == 'alternate':
            moves = random.sample(self.moves, 2)
            return [moves[i % 2] for i in range(length)]
            
        elif pattern_type == 'win_stay_lose_shift':
            sequence = [random.choice(self.moves)]
            for i in range(1, length):
                # Simulate win/lose based on random robot move
                robot_move = random.choice(self.moves)
                if self._get_winner(sequence[-1], robot_move) == 'human':
                    sequence.append(sequence[-1])  # Stay
                else:
                    # Shift to different move
                    other_moves = [m for m in self.moves if m != sequence[-1]]
                    sequence.append(random.choice(other_moves))
            return sequence
            
        elif pattern_type == 'biased':
            weights = [0.45, 0.35, 0.2]  # Biased towards rock
            return random.choices(self.moves, weights=weights, k=length)
            
        elif pattern_type == 'burst':
            sequence = []
            while len(sequence) < length:
                move = random.choice(self.moves)
                burst_length = random.randint(2, 5)
                sequence.extend([move] * min(burst_length, length - len(sequence)))
            return sequence[:length]
            
        elif pattern_type == 'periodic':
            period = random.randint(3, 7)
            pattern = random.choices(self.moves, k=period)
            return [pattern[i % period] for i in range(length)]
            
        elif pattern_type == 'anti_frequency':
            # Start random, then anti-frequency
            sequence = random.choices(self.moves, k=min(10, length))
            for i in range(10, length):
                # Count recent frequencies
                recent = sequence[-10:]
                counts = {move: recent.count(move) for move in self.moves}
                # Choose least frequent
                min_move = min(counts.keys(), key=lambda k: counts[k])
                sequence.append(min_move)
            return sequence[:length]
            
        else:  # random
            return random.choices(self.moves, k=length)
    
    def _get_winner(self, human_move: str, robot_move: str) -> str:
        """Determine winner"""
        if human_move == robot_move:
            return 'draw'
        elif ((human_move == 'rock' and robot_move == 'scissors') or
              (human_move == 'paper' and robot_move == 'rock') or
              (human_move == 'scissors' and robot_move == 'paper')):
            return 'human'
        else:
            return 'robot'

class AdvancedLSTMTrainer:
    """Main training class that orchestrates everything"""
    
    def __init__(self, game_length: int):
        self.config = GAME_CONFIGS.get(game_length, GAME_CONFIGS[100])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.model = AdvancedLSTM(
            hidden_size=self.config.hidden_size,
            embed_dim=8  # As suggested in setup
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=5e-3, 
            betas=(0.9, 0.999)
        )
        
        # Components
        self.feature_extractor = FeatureExtractor()
        self.ngram_predictor = NGramPredictor()
        self.change_detector = ChangePointDetector()
        
        # Training state
        self.replay_buffer = deque(maxlen=self.config.buffer_size)
        self.hidden_state = None
        self.turn_counter = 0
        self.ensemble_weight = 0.8  # LSTM weight (0.2 for n-gram)
        
        # Label smoothing loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        print(f"🏗️ Advanced LSTM Trainer initialized for {game_length}-move games")
        print(f"   Hidden size: {self.config.hidden_size}")
        print(f"   Buffer size: {self.config.buffer_size}")
        print(f"   Sequence length: {self.config.sequence_length}")
    
    def pretrain(self, num_epochs: int = 2, batch_size: int = 32):
        """Pretrain on synthetic data"""
        print("🔄 Starting pretraining on synthetic data...")
        
        generator = SyntheticDataGenerator()
        sequences = generator.generate_patterns(num_sequences=500, seq_length=40)
        
        self.model.train()
        pretrain_optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i:i+batch_size]
                
                # Convert to training data
                batch_features = []
                batch_targets = []
                
                for seq in batch_sequences:
                    if len(seq) < self.config.sequence_length + 1:
                        continue
                        
                    # Simulate robot moves and outcomes for feature extraction
                    robot_moves = [random.choice(['rock', 'paper', 'scissors']) for _ in seq]
                    outcomes = []
                    for h, r in zip(seq, robot_moves):
                        if h == r:
                            outcomes.append('draw')
                        elif get_counter_move(h) == r:
                            outcomes.append('lose')
                        else:
                            outcomes.append('win')
                    
                    # Extract features
                    features = self.feature_extractor.extract_features(seq, robot_moves, outcomes)
                    
                    # Create sequences
                    for j in range(len(features) - self.config.sequence_length):
                        input_seq = features[j:j+self.config.sequence_length]
                        target_move = seq[j + self.config.sequence_length]
                        target_idx = self.model.move_to_idx[target_move]
                        
                        batch_features.append(input_seq)
                        batch_targets.append(target_idx)
                
                if len(batch_features) == 0:
                    continue
                
                # Convert to tensors
                X = torch.stack(batch_features).to(self.device)
                y = torch.tensor(batch_targets, dtype=torch.long).to(self.device)
                
                # Forward pass
                pretrain_optimizer.zero_grad()
                logits, _ = self.model(X)
                loss = self.criterion(logits, y)
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                pretrain_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        print("✅ Pretraining completed")
    
    def predict(self, human_moves: List[str], robot_moves: List[str], 
                outcomes: List[str]) -> Dict[str, float]:
        """Predict next human move with ensemble"""
        if len(human_moves) == 0:
            return {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
        
        # Get LSTM prediction
        lstm_probs = self._lstm_predict(human_moves, robot_moves, outcomes)
        
        # Get n-gram prediction
        ngram_probs = self.ngram_predictor.predict(human_moves)
        
        # Ensemble
        final_probs = {}
        for move in ['rock', 'paper', 'scissors']:
            final_probs[move] = (
                self.ensemble_weight * lstm_probs.get(move, 0.33) +
                (1 - self.ensemble_weight) * ngram_probs.get(move, 0.33)
            )
        
        # Normalize
        total = sum(final_probs.values())
        if total > 0:
            final_probs = {move: prob/total for move, prob in final_probs.items()}
        
        return final_probs
    
    def _lstm_predict(self, human_moves: List[str], robot_moves: List[str],
                     outcomes: List[str]) -> Dict[str, float]:
        """Get LSTM prediction"""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(human_moves, robot_moves, outcomes)
            
            if len(features) == 0:
                return {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
            
            # Use last sequence_length steps
            if len(features) >= self.config.sequence_length:
                input_features = features[-self.config.sequence_length:].unsqueeze(0)
            else:
                # Pad with zeros if needed
                pad_length = self.config.sequence_length - len(features)
                padding = torch.zeros(pad_length, 12)
                input_features = torch.cat([padding, features]).unsqueeze(0)
            
            # Forward pass
            self.model.eval()
            with torch.no_grad():
                input_features = input_features.to(self.device)
                logits, self.hidden_state = self.model(input_features, self.hidden_state)
                probs = F.softmax(logits, dim=1)[0]
            
            return {
                'rock': probs[0].item(),
                'paper': probs[1].item(), 
                'scissors': probs[2].item()
            }
            
        except Exception as e:
            print(f"⚠️ LSTM prediction error: {e}")
            return {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
    
    def update(self, human_moves: List[str], robot_moves: List[str], outcomes: List[str]):
        """Online update after each turn"""
        self.turn_counter += 1
        
        # Update n-gram model
        self.ngram_predictor.update(human_moves)
        
        # Add to replay buffer
        if len(human_moves) >= 2:
            self.replay_buffer.append({
                'human_moves': human_moves.copy(),
                'robot_moves': robot_moves.copy(),
                'outcomes': outcomes.copy()
            })
        
        # Check if we should do online updates
        if (self.turn_counter % self.config.update_frequency == 0 and 
            len(self.replay_buffer) >= self.config.sequence_length):
            
            self._online_update()
        
        # Change point detection
        if len(human_moves) >= 10:
            recent_predictions = []
            for i in range(max(0, len(human_moves)-10), len(human_moves)):
                if i == 0:
                    continue
                pred_probs = self._lstm_predict(
                    human_moves[:i], robot_moves[:i], outcomes[:i]
                )
                predicted_move = max(pred_probs.keys(), key=lambda k: pred_probs[k])
                actual_move = human_moves[i]
                recent_predictions.append(1 if predicted_move == actual_move else 0)
            
            if recent_predictions:
                accuracy = float(np.mean(recent_predictions))
                if self.change_detector.update(accuracy):
                    self._handle_change_point()
    
    def _online_update(self):
        """Perform online learning updates"""
        if len(self.replay_buffer) < self.config.sequence_length:
            return
        
        self.model.train()
        
        for _ in range(self.config.updates_per_turn):
            # Sample random subsequence from buffer
            sample = random.choice(self.replay_buffer)
            
            human_moves = sample['human_moves']
            robot_moves = sample['robot_moves'] 
            outcomes = sample['outcomes']
            
            if len(human_moves) < self.config.sequence_length + 1:
                continue
            
            # Extract features
            features = self.feature_extractor.extract_features(human_moves, robot_moves, outcomes)
            
            # Create training example
            start_idx = random.randint(0, len(features) - self.config.sequence_length - 1)
            input_seq = features[start_idx:start_idx + self.config.sequence_length]
            target_move = human_moves[start_idx + self.config.sequence_length]
            target_idx = self.model.move_to_idx[target_move]
            
            # Convert to tensors
            X = input_seq.unsqueeze(0).to(self.device)
            y = torch.tensor([target_idx], dtype=torch.long).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(X)
            loss = self.criterion(logits, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
    
    def _handle_change_point(self):
        """Handle detected strategy change"""
        print("🔄 Change point detected - adapting...")
        
        # Reset optimizer state
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=7e-3,  # Increased LR
            betas=(0.9, 0.999)
        )
        
        # Shrink buffer temporarily
        temp_buffer_size = self.config.buffer_size // 2
        while len(self.replay_buffer) > temp_buffer_size:
            self.replay_buffer.popleft()
        
        # Adjust ensemble weights temporarily
        self.ensemble_weight = 0.6  # Reduce LSTM confidence
    
    def reset_episode(self):
        """Reset for new episode"""
        self.hidden_state = None
        self.turn_counter = 0
        self.ensemble_weight = 0.8  # Reset to default
        
    def save_model(self, path: str):
        """Save the trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'ensemble_weight': self.ensemble_weight
        }
        
        torch.save(state, path)
        print(f"💾 Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        if not os.path.exists(path):
            return False
            
        try:
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state['model_state_dict'])
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.ensemble_weight = state.get('ensemble_weight', 0.8)
            print(f"✅ Model loaded from {path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

def main():
    """Main function for testing"""
    print("🚀 Advanced LSTM Training System")
    print("Based on lstm_rps_setup.md recommendations")
    
    # Test with 100-move game configuration
    trainer = AdvancedLSTMTrainer(game_length=100)
    
    # Pretrain
    trainer.pretrain(num_epochs=2)
    
    # Test prediction
    human_moves = ['rock', 'paper', 'scissors', 'rock']
    robot_moves = ['paper', 'scissors', 'rock', 'paper']
    outcomes = ['lose', 'lose', 'lose', 'lose']
    
    probs = trainer.predict(human_moves, robot_moves, outcomes)
    print(f"✅ Test prediction: {probs}")
    
    # Save model
    model_path = "models/lstm/advanced_lstm_100.pth"
    trainer.save_model(model_path)
    
    print("✅ Advanced LSTM training system ready!")

if __name__ == "__main__":
    main()