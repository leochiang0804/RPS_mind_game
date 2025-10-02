#!/usr/bin/env python3
"""
LSTM Model for Rock Paper Scissors Prediction
Implements a tiny LSTM with PyTorch for predicting human moves
Based on the development plan requirements
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import random

class TinyLSTM(nn.Module):
    """
    Tiny LSTM model for RPS prediction
    Architecture: embed=8, hidden=24, 1-2 layers, dropout≤0.1
    """
    
    def __init__(self, vocab_size=3, embed_dim=8, hidden_dim=24, num_layers=2, dropout=0.1):
        super(TinyLSTM, self).__init__()
        self.vocab_size = vocab_size  # 3 for R, P, S
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer: convert move indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
        # Move mapping - using unified system: 0=rock, 1=paper, 2=scissors
        from move_mapping import MOVE_TO_NUMBER, NUMBER_TO_MOVE, normalize_move
        self.move_to_idx = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.idx_to_move = {0: 'rock', 1: 'paper', 2: 'scissors'}
        self.normalize_move = normalize_move
        
    def forward(self, x, hidden=None):
        """Forward pass through the network"""
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch_size, seq_len, hidden_dim)
        
        # Take the last output for prediction
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Project to vocab size
        logits = self.output_proj(last_output)  # (batch_size, vocab_size)
        probabilities = self.softmax(logits)
        
        return probabilities, hidden
    
    def predict_next_move(self, move_history: List[str], temperature=1.0) -> Dict[str, float]:
        """
        Predict the next move given history
        Returns probabilities for each move
        """
        if len(move_history) == 0:
            return {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
        
        # Convert moves to indices using normalized moves
        try:
            normalized_history = [self.normalize_move(move.lower()) for move in move_history]
            move_indices = [self.move_to_idx[move] for move in normalized_history]
        except KeyError as e:
            print(f"Unknown move in history: {e}")
            return {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
        
        # Convert to tensor
        x = torch.tensor([move_indices], dtype=torch.long)  # (1, seq_len)
        
        self.eval()
        with torch.no_grad():
            probabilities, _ = self.forward(x)
            probs = probabilities[0].numpy()  # Extract single prediction
            
            # Apply temperature
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs = probs / np.sum(probs)
        
        return {
            'rock': float(probs[0]),
            'paper': float(probs[1]),
            'scissors': float(probs[2])
        }

class LSTMTrainer:
    """Training pipeline for the LSTM model"""
    
    def __init__(self, model: TinyLSTM, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.training_history = []
        
    def create_sequences(self, moves: List[str], seq_length: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create training sequences from move history
        Returns input sequences and target moves
        """
        if len(moves) < seq_length + 1:
            return None, None
        
        move_indices = [self.model.move_to_idx[move.lower()] for move in moves]
        
        sequences = []
        targets = []
        
        for i in range(len(move_indices) - seq_length):
            seq = move_indices[i:i + seq_length]
            target = move_indices[i + seq_length]
            sequences.append(seq)
            targets.append(target)
        
        return torch.tensor(sequences, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    def train_on_history(self, move_history: List[str], seq_length: int = 10, epochs: int = 50):
        """
        Train the model on a sequence of moves
        """
        if len(move_history) < seq_length + 1:
            print(f"Not enough history for training. Need at least {seq_length + 1} moves, got {len(move_history)}")
            return
        
        # Create training data
        X, y = self.create_sequences(move_history, seq_length)
        if X is None:
            return
        
        self.model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            probabilities, _ = self.model(X)
            
            # Calculate loss
            loss = self.criterion(probabilities, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / epochs
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'data_size': len(move_history),
            'sequences': len(X),
            'epochs': epochs,
            'final_loss': avg_loss
        })
        
        print(f"Training completed: {epochs} epochs, final loss: {avg_loss:.4f}")
    
    def fine_tune(self, recent_moves: List[str], epochs: int = 10, seq_length: int = 5):
        """
        Fine-tune on recent moves (every 5 rounds as per plan)
        """
        if len(recent_moves) < seq_length + 1:
            return
        
        print(f"Fine-tuning on last {len(recent_moves)} moves...")
        self.train_on_history(recent_moves, seq_length, epochs)

class SyntheticPlayerGenerator:
    """Generate synthetic player data for training and testing"""
    
    @staticmethod
    def repeater_player(length: int, repeat_prob: float = 0.8) -> List[str]:
        """Player that tends to repeat their last move"""
        moves = ['rock', 'paper', 'scissors']
        history = [random.choice(moves)]
        
        for _ in range(length - 1):
            if random.random() < repeat_prob and len(history) > 0:
                history.append(history[-1])
            else:
                history.append(random.choice(moves))
        
        return history
    
    @staticmethod
    def cycler_player(length: int, cycle_prob: float = 0.7) -> List[str]:
        """Player that tends to cycle through R→P→S"""
        cycle = ['rock', 'paper', 'scissors']
        history = []
        position = 0
        
        for _ in range(length):
            if random.random() < cycle_prob:
                history.append(cycle[position % 3])
                position += 1
            else:
                history.append(random.choice(cycle))
        
        return history
    
    @staticmethod
    def mirror_player(length: int, opponent_history: List[str], mirror_prob: float = 0.6) -> List[str]:
        """Player that tends to mirror opponent's previous move"""
        moves = ['rock', 'paper', 'scissors']
        history = [random.choice(moves)]
        
        for i in range(1, length):
            if i < len(opponent_history) and random.random() < mirror_prob:
                history.append(opponent_history[i-1])
            else:
                history.append(random.choice(moves))
        
        return history
    
    @staticmethod
    def shifter_player(length: int, shift_interval: int = 20) -> List[str]:
        """Player that changes strategy every N rounds"""
        history = []
        strategies = [
            lambda: SyntheticPlayerGenerator.repeater_player(shift_interval),
            lambda: SyntheticPlayerGenerator.cycler_player(shift_interval),
            lambda: [random.choice(['rock', 'paper', 'scissors']) for _ in range(shift_interval)]
        ]
        
        remaining = length
        while remaining > 0:
            chunk_size = min(shift_interval, remaining)
            strategy = random.choice(strategies)
            chunk = strategy()[:chunk_size]
            history.extend(chunk)
            remaining -= chunk_size
        
        return history[:length]

def create_and_export_model(export_path: str = "models/lstm/lstm_rps.onnx"):
    """
    Create, train, and export LSTM model
    """
    print("🧠 Creating and training LSTM model...")
    
    # Create model
    model = TinyLSTM()
    trainer = LSTMTrainer(model)
    
    # Generate synthetic training data
    print("📊 Generating synthetic training data...")
    training_data = []
    
    # Different player types
    training_data.extend(SyntheticPlayerGenerator.repeater_player(200))
    training_data.extend(SyntheticPlayerGenerator.cycler_player(200))
    training_data.extend(SyntheticPlayerGenerator.shifter_player(200))
    
    # Add some random data
    for _ in range(100):
        training_data.append(random.choice(['rock', 'paper', 'scissors']))
    
    print(f"Generated {len(training_data)} training moves")
    
    # Train the model
    print("🎯 Training model...")
    trainer.train_on_history(training_data, epochs=100)
    
    # Create export directory
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    # Export to ONNX (requires onnx package)
    try:
        import onnx
        
        # Create dummy input for tracing
        dummy_input = torch.randint(0, 3, (1, 10))  # batch_size=1, seq_length=10
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['move_sequence'],
            output_names=['move_probabilities'],
            dynamic_axes={
                'move_sequence': {1: 'sequence_length'},
                'move_probabilities': {0: 'batch_size'}
            }
        )
        
        print(f"✅ Model exported to ONNX: {export_path}")
        
    except ImportError:
        print("⚠️ ONNX not available, saving PyTorch model instead")
        torch_path = export_path.replace('.onnx', '.pth')
        torch.save(model.state_dict(), torch_path)
        print(f"✅ PyTorch model saved: {torch_path}")
    
    # Save model metadata
    metadata = {
        'model_type': 'TinyLSTM',
        'vocab_size': model.vocab_size,
        'embed_dim': model.embed_dim,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
        'move_mapping': model.move_to_idx,
        'training_data_size': len(training_data),
        'created_at': datetime.now().isoformat(),
        'training_history': trainer.training_history
    }
    
    metadata_path = export_path.replace('.onnx', '_metadata.json').replace('.pth', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved: {metadata_path}")
    
    # Test the model
    print("🧪 Testing model...")
    test_history = ['rock', 'rock', 'paper', 'paper', 'scissors']
    predictions = model.predict_next_move(test_history)
    print(f"Test prediction for {test_history}: {predictions}")
    
    return model, trainer

if __name__ == "__main__":
    print("🚀 LSTM Model Training Pipeline")
    print("=" * 50)
    
    model, trainer = create_and_export_model()
    
    print("\n🎉 LSTM training pipeline completed!")
    print("📂 Check the 'models/lstm/' directory for exported files")