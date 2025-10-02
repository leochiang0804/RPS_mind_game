#!/usr/bin/env python3
"""
Quick start script for enhanced student model training
Uses teacher model toggle and 1-hour training duration
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def main():
    """Main training entry point"""
    print("🚀 Starting Enhanced Student Model Training")
    print("   Duration: ~1 hour")
    print("   Teacher: LLM (existing coaching system)")
    print("   Architecture: 114M parameters with LoRA")
    
    try:
        # Import training components
        from model_training.real_training_pipeline import RealTrainingConfig, RealTrainingPipeline
        
        # Create configuration for 1-hour training
        config = RealTrainingConfig()
        config.teacher_type = "llm"  # Use LLM teacher
        config.total_training_steps = 2500  # ~1 hour
        config.total_steps = 2500  # Alias for compatibility
        config.batch_size = 16
        config.num_training_examples = 4000
        config.use_lora = True
        config.use_mixed_precision = True
        
        print(f"\n📋 Training Configuration:")
        print(f"   Model: {config.hidden_size}D, {config.num_layers} layers")
        print(f"   Steps: {config.total_training_steps:,}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Examples: {config.num_training_examples:,}")
        print(f"   Teacher: {config.teacher_type}")
        print(f"   LoRA: {config.use_lora}")
        
        # Initialize training pipeline
        print(f"\n🏗️ Initializing training pipeline...")
        pipeline = RealTrainingPipeline(config)
        
        # Create and setup model
        print(f"🔧 Creating enhanced model...")
        model = pipeline.create_model()
        pipeline.model = model
        
        # Setup training components
        print(f"⚙️ Setting up training components...")
        pipeline.setup_training_components()
        
        # Start training
        print(f"\n🎯 Starting training...")
        print(f"   Estimated duration: ~1 hour")
        print(f"   Using teacher: LLM (existing coaching system)")
        
        # Run training
        results = pipeline.train()
        
        print(f"\n✅ Training completed!")
        print(f"   Final model size: {results.get('final_model_size_mb', 'Unknown')} MB")
        print(f"   Total parameters: {results.get('total_parameters', 'Unknown'):,}")
        print(f"   Training loss: {results.get('final_loss', 'Unknown'):.4f}")
        print(f"   Model saved to: {results.get('model_path', 'Unknown')}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())