#!/usr/bin/env python3
"""
Validation test for enhanced ML model
Compares original vs enhanced model performance
"""

import sys
import random
from collections import defaultdict

# Import both models
from ml_model import MLModel as OriginalMLModel
from ml_model_enhanced import EnhancedMLModel

def generate_test_patterns(length=200):
    """Generate test data with known patterns"""
    patterns = []
    
    # Pattern 1: Repeater (repeats last move 70% of time)
    repeater_data = []
    last = random.choice(['paper', 'scissor', 'stone'])
    for _ in range(length // 4):
        if random.random() < 0.7 and repeater_data:
            last = repeater_data[-1]
        else:
            last = random.choice(['paper', 'scissor', 'stone'])
        repeater_data.append(last)
    patterns.append(("Repeater", repeater_data))
    
    # Pattern 2: Cycler (follows rock->paper->scissors cycle)
    cycle = ['stone', 'paper', 'scissor']
    cycler_data = []
    for i in range(length // 4):
        if random.random() < 0.8:  # 80% cycle adherence
            cycler_data.append(cycle[i % 3])
        else:
            cycler_data.append(random.choice(['paper', 'scissor', 'stone']))
    patterns.append(("Cycler", cycler_data))
    
    # Pattern 3: Anti-repeater (never repeats last move)
    anti_repeater_data = []
    last = random.choice(['paper', 'scissor', 'stone'])
    anti_repeater_data.append(last)
    for _ in range(length // 4 - 1):
        choices = [m for m in ['paper', 'scissor', 'stone'] if m != last]
        last = random.choice(choices)
        anti_repeater_data.append(last)
    patterns.append(("Anti-repeater", anti_repeater_data))
    
    # Pattern 4: Strategy shifter (changes pattern mid-game)
    shifter_data = []
    # First half: repeat pattern
    last = random.choice(['paper', 'scissor', 'stone'])
    for _ in range(length // 8):
        if random.random() < 0.7:
            shifter_data.append(last)
        else:
            last = random.choice(['paper', 'scissor', 'stone'])
            shifter_data.append(last)
    
    # Second half: cycle pattern
    for i in range(length // 8):
        if random.random() < 0.8:
            shifter_data.append(cycle[i % 3])
        else:
            shifter_data.append(random.choice(['paper', 'scissor', 'stone']))
    
    patterns.append(("Shifter", shifter_data))
    
    return patterns

def test_model_accuracy(model, data, model_name, pattern_name):
    """Test model accuracy on given data"""
    correct_predictions = 0
    total_predictions = 0
    confidence_scores = []
    
    # Train on first 70% of data, test on remaining 30%
    split_point = int(len(data) * 0.7)
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    model.train(train_data)
    
    for i in range(len(test_data) - 1):
        current_history = train_data + test_data[:i+1]
        actual_next = test_data[i+1]
        
        # Get prediction
        if hasattr(model, 'predict') and len(model.predict.__code__.co_varnames) > 1:
            # Enhanced model returns (move, confidence)
            try:
                predicted_counter, confidence = model.predict(current_history)
                confidence_scores.append(confidence)
            except ValueError:
                # Fallback for models that return only move
                predicted_counter = model.predict(current_history)
                confidence = 0.5
                confidence_scores.append(confidence)
        else:
            predicted_counter = model.predict(current_history)
            confidence = 0.5
            confidence_scores.append(confidence)
        
        # Convert counter move back to predicted human move
        reverse_counter = {'scissor': 'paper', 'stone': 'scissor', 'paper': 'stone'}
        predicted_human = reverse_counter.get(predicted_counter, predicted_counter)
        
        if predicted_human == actual_next:
            correct_predictions += 1
        total_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct_predictions,
        'total': total_predictions,
        'avg_confidence': avg_confidence
    }

def main():
    print("=== Enhanced ML Model Validation Test ===\n")
    
    # Generate test patterns
    patterns = generate_test_patterns(200)
    
    # Initialize models
    original_model = OriginalMLModel()
    enhanced_models = [
        EnhancedMLModel(order=1, recency_weight=0.9),
        EnhancedMLModel(order=2, recency_weight=0.8),
        EnhancedMLModel(order=3, recency_weight=0.7),
    ]
    
    results = defaultdict(list)
    
    # Test each pattern
    for pattern_name, data in patterns:
        print(f"Testing pattern: {pattern_name}")
        print("-" * 40)
        
        # Test original model
        orig_result = test_model_accuracy(original_model, data, "Original", pattern_name)
        results[pattern_name].append(("Original", orig_result))
        print(f"Original Model: {orig_result['accuracy']:.3f} accuracy ({orig_result['correct']}/{orig_result['total']})")
        
        # Test enhanced models
        for i, enhanced_model in enumerate(enhanced_models):
            model_name = f"Enhanced(order={enhanced_model.order}, decay={enhanced_model.recency_weight})"
            enh_result = test_model_accuracy(enhanced_model, data, model_name, pattern_name)
            results[pattern_name].append((model_name, enh_result))
            print(f"{model_name}: {enh_result['accuracy']:.3f} accuracy ({enh_result['correct']}/{enh_result['total']}) avg_conf={enh_result['avg_confidence']:.3f}")
        
        print()
    
    # Summary
    print("=== SUMMARY ===")
    overall_improvement = False
    
    for pattern_name in results:
        original_acc = results[pattern_name][0][1]['accuracy']
        best_enhanced = max(results[pattern_name][1:], key=lambda x: x[1]['accuracy'])
        best_acc = best_enhanced[1]['accuracy']
        
        improvement = best_acc - original_acc
        if improvement > 0.05:  # 5% improvement threshold
            overall_improvement = True
            print(f"✓ {pattern_name}: +{improvement:.3f} improvement ({best_enhanced[0]})")
        elif improvement > 0:
            print(f"~ {pattern_name}: +{improvement:.3f} slight improvement ({best_enhanced[0]})")
        else:
            print(f"✗ {pattern_name}: {improvement:.3f} no improvement")
    
    print(f"\nOverall result: {'PASS' if overall_improvement else 'FAIL'}")
    print("Enhanced model shows significant improvement on pattern recognition tasks." if overall_improvement 
          else "Enhanced model needs further tuning.")
    
    return overall_improvement

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)