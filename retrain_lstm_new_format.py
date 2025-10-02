#!/usr/bin/env python3
"""
Retrain LSTM Model with New Format
Retrains the LSTM model using the new rock/paper/scissors format instead of stone/paper/scissor
"""

import os
import json
import random
from datetime import datetime
from lstm_model import TinyLSTM, LSTMTrainer, SyntheticPlayerGenerator

def generate_training_data(size: int = 700) -> list:
    """Generate training data using new rock/paper/scissors format"""
    print(f"🎯 Generating {size} training moves with new format...")
    
    # Generate various player patterns using new format
    training_data = []
    
    # Random player
    training_data.extend([random.choice(['rock', 'paper', 'scissors']) for _ in range(100)])
    
    # Repeater patterns (30% of data)
    training_data.extend(SyntheticPlayerGenerator.repeater_player(210, repeat_prob=0.8))
    
    # Cycler patterns (30% of data)  
    training_data.extend(SyntheticPlayerGenerator.cycler_player(210, cycle_prob=0.7))
    
    # Mirror and adaptive patterns (40% of data)
    opponent_moves = [random.choice(['rock', 'paper', 'scissors']) for _ in range(180)]
    training_data.extend(SyntheticPlayerGenerator.mirror_player(180, opponent_moves, mirror_prob=0.6))
    
    print(f"✅ Generated {len(training_data)} training moves")
    return training_data

def retrain_model():
    """Retrain the LSTM model with new format"""
    print("🚀 Starting LSTM retraining with rock/paper/scissors format...")
    
    # Create model with new mapping
    model = TinyLSTM(vocab_size=3, embed_dim=8, hidden_dim=24, num_layers=2)
    
    # Generate training data
    training_data = generate_training_data(700)
    
    # Create trainer
    trainer = LSTMTrainer(model, learning_rate=0.001)
    
    # Train the model
    print("🏋️ Training model...")
    trainer.train_on_history(training_data, seq_length=10, epochs=100)
    
    # Test the model
    print("🧪 Testing model...")
    test_history = ['rock', 'rock', 'paper', 'paper', 'scissors']
    predictions = model.predict_next_move(test_history)
    print(f"Test prediction for {test_history}: {predictions}")
    
    # Save the model
    model_dir = "models/lstm"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "lstm_rps.pth")
    metadata_path = os.path.join(model_dir, "lstm_rps_metadata.json")
    
    # Save model weights
    import torch
    torch.save(model.state_dict(), model_path)
    
    # Create new metadata with correct format
    metadata = {
        "model_type": "TinyLSTM",
        "vocab_size": 3,
        "embed_dim": 8,
        "hidden_dim": 24,
        "num_layers": 2,
        "move_mapping": {
            "rock": 0,
            "paper": 1,
            "scissors": 2
        },
        "training_data_size": len(training_data),
        "created_at": datetime.now().isoformat(),
        "training_history": [
            {
                "timestamp": datetime.now().isoformat(),
                "data_size": len(training_data),
                "sequences": 1,
                "epochs": 100,
                "format": "rock/paper/scissors"
            }
        ]
    }
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model saved to {model_path}")
    print(f"✅ Metadata saved to {metadata_path}")
    print("🎉 LSTM retraining complete!")
    
    return model

if __name__ == "__main__":
    retrain_model()