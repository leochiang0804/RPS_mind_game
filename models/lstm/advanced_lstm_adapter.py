"""
Advanced LSTM Model Adapter
Converts advanced LSTM models to work with the existing lstm_unified.py system
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_lstm_trainer import AdvancedLSTM, AdvancedLSTMTrainer
from lstm_unified import LSTMManager, TinyLSTM

class AdvancedLSTMAdapter:
    """Adapter to make advanced LSTM work with existing test platform"""
    
    def __init__(self, game_length: int = 100):
        self.game_length = game_length
        self.advanced_trainer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_advanced_model(self, model_path: str) -> bool:
        """Load an advanced LSTM model"""
        try:
            self.advanced_trainer = AdvancedLSTMTrainer(self.game_length)
            success = self.advanced_trainer.load_model(model_path)
            if success:
                print(f"✅ Advanced LSTM model loaded from {model_path}")
                return True
            else:
                print(f"❌ Failed to load advanced model from {model_path}")
                return False
        except Exception as e:
            print(f"❌ Error loading advanced model: {e}")
            return False
    
    def convert_to_simple_lstm(self, output_path: str) -> bool:
        """Convert advanced LSTM to simple format for existing system"""
        if not self.advanced_trainer:
            print("❌ No advanced model loaded")
            return False
        
        try:
            # Create a simple LSTM with similar architecture
            simple_lstm = TinyLSTM(
                vocab_size=3,
                embed_dim=16,
                hidden_dim=self.advanced_trainer.config.hidden_size,
                num_layers=1
            )
            
            # We can't directly convert the weights due to different architectures
            # So we'll save the advanced model in a way that can be loaded by a wrapper
            
            # Save the advanced model state
            model_data = {
                'model_state_dict': self.advanced_trainer.model.state_dict(),
                'model_type': 'advanced_lstm',
                'game_length': self.game_length,
                'config': self.advanced_trainer.config,
                'architecture': {
                    'hidden_size': self.advanced_trainer.config.hidden_size,
                    'input_dim': 12,  # Advanced LSTM uses 12-dim features
                    'embed_dim': 8
                }
            }
            
            # Save metadata for the existing system
            metadata = {
                'vocab_size': 3,
                'embed_dim': 16,
                'hidden_dim': self.advanced_trainer.config.hidden_size,
                'num_layers': 1,
                'model_type': 'advanced_lstm_converted',
                'game_length': self.game_length,
                'win_rate': 'trained_on_realistic_data',
                'accuracy': 'advanced_features'
            }
            
            # Save both
            torch.save(model_data, output_path)
            
            metadata_path = output_path.replace('.pth', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Advanced model converted and saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error converting model: {e}")
            return False

class AdvancedLSTMWrapper:
    """Wrapper to make advanced LSTM work like the simple LSTM in predictions"""
    
    def __init__(self, game_length: int = 100):
        self.game_length = game_length
        self.trainer = None
        self.move_history = []
        self.robot_history = []
        self.outcome_history = []
        
    def load_model(self, model_path: str) -> bool:
        """Load the advanced model"""
        try:
            self.trainer = AdvancedLSTMTrainer(self.game_length)
            success = self.trainer.load_model(model_path)
            if success:
                self.trainer.reset_episode()
                return True
            return False
        except Exception as e:
            print(f"❌ Error loading wrapper model: {e}")
            return False
    
    def predict(self, moves: List[str]) -> Dict[str, float]:
        """Predict next move using advanced LSTM"""
        if not self.trainer:
            return {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
        
        try:
            # For the wrapper, we need to simulate robot moves and outcomes
            # since the test platform only gives us human moves
            
            # Update our internal history
            self.move_history = moves.copy()
            
            # Generate synthetic robot moves and outcomes for missing data
            while len(self.robot_history) < len(self.move_history):
                self.robot_history.append('rock')  # Default robot move
            
            while len(self.outcome_history) < len(self.move_history):
                # Generate random outcomes for training
                self.outcome_history.append('draw')  # Default outcome
            
            # Use the advanced trainer to predict
            probs = self.trainer.predict(
                self.move_history, 
                self.robot_history, 
                self.outcome_history
            )
            
            return probs
            
        except Exception as e:
            print(f"⚠️ Advanced LSTM wrapper prediction error: {e}")
            return {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
    
    def update_with_outcome(self, robot_move: str, outcome: str):
        """Update the model with the actual robot move and outcome"""
        if len(self.robot_history) > 0:
            self.robot_history[-1] = robot_move
        else:
            self.robot_history.append(robot_move)
            
        if len(self.outcome_history) > 0:
            self.outcome_history[-1] = outcome
        else:
            self.outcome_history.append(outcome)
        
        # Update the trainer with real data
        if self.trainer and len(self.move_history) > 0:
            try:
                self.trainer.update(self.move_history, self.robot_history, self.outcome_history)
            except Exception as e:
                print(f"⚠️ Error updating advanced LSTM: {e}")

def setup_advanced_lstm_for_testing(game_lengths: Optional[List[int]] = None):
    """Set up advanced LSTM models for testing with the existing platform"""
    if game_lengths is None:
        game_lengths = [25, 50, 100, 250, 500, 1000]
    
    print("🔧 Setting up Advanced LSTM models for testing...")
    
    for length in game_lengths:
        print(f"📏 Setting up model for {length}-move games...")
        
        # Check if we have a trained advanced model
        advanced_model_path = f"models/lstm/advanced_lstm_trained_{length}.pth"
        
        if not os.path.exists(advanced_model_path):
            print(f"⚠️ No trained model found for {length}, creating default...")
            # Create and train a quick model
            trainer = AdvancedLSTMTrainer(length)
            trainer.pretrain(num_epochs=2)
            trainer.save_model(advanced_model_path)
        
        # Convert for the existing system
        adapter = AdvancedLSTMAdapter(length)
        if adapter.load_advanced_model(advanced_model_path):
            output_path = f"models/lstm/advanced_converted_{length}.pth"
            adapter.convert_to_simple_lstm(output_path)
    
    print("✅ Advanced LSTM models ready for testing!")

def main():
    """Test the adapter system"""
    print("🧪 Testing Advanced LSTM Adapter")
    
    # Setup models
    setup_advanced_lstm_for_testing([100])
    
    # Test the wrapper
    wrapper = AdvancedLSTMWrapper(100)
    model_path = "models/lstm/advanced_lstm_trained_100.pth"
    
    if os.path.exists(model_path):
        if wrapper.load_model(model_path):
            # Test prediction
            test_moves = ['rock', 'paper', 'scissors', 'rock']
            result = wrapper.predict(test_moves)
            print(f"✅ Test prediction: {result}")
        else:
            print("❌ Failed to load model")
    else:
        print(f"⚠️ Model not found: {model_path}")

if __name__ == "__main__":
    main()