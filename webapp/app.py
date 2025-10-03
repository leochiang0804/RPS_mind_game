
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy import FrequencyStrategy, MarkovStrategy
from change_point_detector import ChangePointDetector
from optimized_strategies import ToWinStrategy, NotToLoseStrategy
from personality_engine import get_personality_engine
from move_mapping import normalize_move, get_counter_move, MOVES

# Centralized Data Management - All AI coach endpoints use this for consistent data building
from game_context import build_game_context

# Import Developer Console
try:
    from developer_console import console, track_move, track_inference, get_developer_report, get_chart
    DEVELOPER_CONSOLE_AVAILABLE = True
    print("✅ Developer Console available")
except ImportError as e:
    DEVELOPER_CONSOLE_AVAILABLE = False
    print(f"⚠️ Developer Console not available: {e}")

# Import Performance Optimizer
try:
    from performance_optimizer import optimizer, time_model_inference, get_performance_report, start_performance_monitoring
    PERFORMANCE_OPTIMIZER_AVAILABLE = True
    print("✅ Performance Optimizer available")
    # Start performance monitoring
    start_performance_monitoring()
    print("📊 Performance monitoring started")
except ImportError as e:
    PERFORMANCE_OPTIMIZER_AVAILABLE = False
    print(f"⚠️ Performance Optimizer not available: {e}")

# Try to import LSTM functionality
try:
    from lstm_web_integration import get_lstm_predictor, init_lstm_model
    LSTM_AVAILABLE = True
    print("✅ LSTM integration available")
except ImportError as e:
    LSTM_AVAILABLE = False
    print(f"⚠️ LSTM not available: {e}")

# AI Coach functionality - Currently under development
# Coach View frontend is available, but backend AI Coach features are being redesigned
AI_COACH_AVAILABLE = False
print("🚧 AI Coach under development - Coach View available in frontend")

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-for-ai-coach-sessions-2024'

# Initialize strategy instances
frequency_strategy = FrequencyStrategy()
markov_strategy = MarkovStrategy()
to_win_strategy = ToWinStrategy()
not_to_lose_strategy = NotToLoseStrategy()

# Initialize LSTM if available
lstm_predictor = None
if LSTM_AVAILABLE:
    try:
        lstm_predictor = get_lstm_predictor()
        if init_lstm_model():
            print("✅ LSTM model initialized for webapp")
        else:
            print("⚠️ LSTM model failed to initialize")
            LSTM_AVAILABLE = False
    except Exception as e:
        print(f"⚠️ LSTM initialization error: {e}")
        LSTM_AVAILABLE = False

# Initialize advanced personality engine
personality_engine = get_personality_engine()

# Initialize strategies and change detector with balanced settings for web gameplay
change_detector = ChangePointDetector(window_size=6, chi2_threshold=3.5, min_segment_length=4)

game_state = {
    'human_history': [],
    'robot_history': [],
    'result_history': [],
    'round_history': [],  # Each entry: {'round': n, 'human': move, 'robot': move}
    'round': 0,
    'stats': {'human_win': 0, 'robot_win': 0, 'tie': 0},
    'difficulty': 'markov',  # Default to markov mode
    'ai_difficulty': 'markov',  # Explicit AI difficulty label
    'strategy_preference': 'to_win',  # Default strategy preference
    'personality': 'neutral',  # Default personality
    'multiplayer': False,
    'change_points': [],  # Store detected strategy changes
    'current_strategy': 'warming up',  # Human strategy label (legacy field)
    'human_strategy_label': 'warming up',  # Explicit human strategy label for analytics
    'accuracy': {
        'random': None,
        'frequency': None,
        'markov': None,
        'lstm': None,
    },
    'correct_predictions': {
        'random': 0,
        'frequency': 0,
        'markov': 0,
        'lstm': 0,
    },
    'total_predictions': {
        'random': 0,
        'frequency': 0,
        'markov': 0,
        'lstm': 0,
    },
    'model_predictions_history': {
        'random': [],
        'frequency': [],
        'markov': [],
        'lstm': [],
    },
    'model_confidence_history': {
        'random': [],
        'frequency': [],
        'markov': [],
        'lstm': [],
        'to_win': [],
        'not_to_lose': [],
    },
    'confidence_score_history': [],  # Track confidence for each round based on active strategy
    'confidence_score_modified_by_personality_history': [],  # Track personality-modified confidence
}

# Game constants - using standard rock/paper/scissors terminology
MOVES = ['rock', 'paper', 'scissors']
HOTKEYS = {'a': 'paper', 'w': 'scissors', 'd': 'rock'}

@app.route('/')
def index():
    # Serve all game state for single-page UI - using main index.html template
    session['game_recording'] = False
    return render_template('index.html', moves=MOVES, hotkeys=HOTKEYS,
        stats=game_state['stats'],
        human_history=game_state['human_history'],
        robot_history=game_state['robot_history'],
        result_history=game_state['result_history'],
        round=game_state['round'],
        default_difficulty=game_state['difficulty'])

@app.route('/test')
def test():
    # Simple test page
    return render_template('test_simple.html')

@app.route('/debug')
def debug():
    # Minimal debug page
    return render_template('minimal_debug.html')

def create_prediction_debug_object():
    """
    Creates a comprehensive debug object showing:
    1. All model predictions for each round
    2. Actual human moves for each round  
    3. Move-by-move comparison showing matches/misses
    4. Current accuracy calculation (wrong method)
    5. Correct accuracy calculation
    6. Manual verification data
    """
    debug_data = {}
    
    # Get current game data
    human_moves = game_state['human_history']
    predictions_history = game_state['model_predictions_history']
    
    # Models to analyze (exclude enhanced since it was removed)
    models = ['random', 'frequency', 'markov', 'lstm']
    
    for model in models:
        if model in predictions_history:
            predictions = predictions_history[model]
            
            # Create detailed comparison for this model
            debug_data[model] = {
                'prediction_history': predictions.copy(),
                'human_move_history': human_moves.copy(),
                'total_predictions': len(predictions),
                'total_human_moves': len(human_moves),
                
                # Current (wrong) accuracy calculation: predictions[i] vs human_history[i]
                'current_method': {
                    'description': 'Current backend method (WRONG): predictions[i] vs human_history[i]',
                    'comparisons': [],
                    'correct_count': 0,
                    'accuracy_percentage': 0
                },
                
                # Correct accuracy calculation: predictions[i] vs human_history[i+1]
                'correct_method': {
                    'description': 'Correct method (RIGHT): predictions[i] vs human_history[i+1]', 
                    'comparisons': [],
                    'correct_count': 0,
                    'accuracy_percentage': 0
                },
                
                # AI Strategy Accuracy from current system
                'ai_strategy_accuracy': game_state['accuracy'].get(model, 0)
            }
            
            # Calculate current (wrong) method - predictions[i] vs human_history[i] 
            current_correct = 0
            for i in range(min(len(predictions), len(human_moves))):
                predicted = predictions[i]
                actual = human_moves[i]  # Wrong: should be human_moves[i+1]
                is_correct = predicted == actual
                if is_correct:
                    current_correct += 1
                    
                debug_data[model]['current_method']['comparisons'].append({
                    'round': i + 1,
                    'predicted': predicted,
                    'actual': actual,
                    'correct': is_correct,
                    'note': f'Round {i+1}: Prediction #{i} vs Human move #{i}'
                })
            
            if len(predictions) > 0:
                debug_data[model]['current_method']['correct_count'] = current_correct
                debug_data[model]['current_method']['accuracy_percentage'] = round((current_correct / len(predictions)) * 100, 1)
            
            # Calculate correct method - predictions[i] vs human_history[i+1]
            correct_correct = 0
            for i in range(len(predictions)):
                if i + 1 < len(human_moves):  # Make sure we have the next human move
                    predicted = predictions[i] 
                    actual = human_moves[i + 1]  # Correct: prediction was for NEXT move
                    is_correct = predicted == actual
                    if is_correct:
                        correct_correct += 1
                        
                    debug_data[model]['correct_method']['comparisons'].append({
                        'round': i + 1,
                        'predicted': predicted, 
                        'actual': actual,
                        'correct': is_correct,
                        'note': f'Round {i+1}: Prediction #{i} vs Human move #{i+1}'
                    })
            
            # Calculate accuracy for correct method
            valid_predictions = len([p for i, p in enumerate(predictions) if i + 1 < len(human_moves)])
            if valid_predictions > 0:
                debug_data[model]['correct_method']['correct_count'] = correct_correct
                debug_data[model]['correct_method']['accuracy_percentage'] = round((correct_correct / valid_predictions) * 100, 1)
            
            # Add discrepancy analysis
            current_acc = debug_data[model]['current_method']['accuracy_percentage']
            correct_acc = debug_data[model]['correct_method']['accuracy_percentage'] 
            ai_strategy_acc = debug_data[model]['ai_strategy_accuracy'] or 0  # Handle None values
            
            debug_data[model]['discrepancy_analysis'] = {
                'current_vs_correct': round(abs(current_acc - correct_acc), 1),
                'ai_strategy_vs_current': round(abs(ai_strategy_acc - current_acc), 1),
                'ai_strategy_vs_correct': round(abs(ai_strategy_acc - correct_acc), 1),
                'explanation': f'AI Strategy shows {ai_strategy_acc}%, Current method shows {current_acc}%, Correct method shows {correct_acc}%'
            }
    
    return debug_data

def create_centralized_model_prediction_tracking():
    """
    Create Model Prediction Tracking object sourced from centralized data management.
    Returns prediction history for each model that can be used for charts and analysis.
    """
    # Get data from centralized system
    game_data = build_game_context(
        session=dict(session),
        overrides={},
        context_type='full'
    )
    
    # Extract model predictions from centralized data
    model_predictions = game_data.get('model_predictions_history', {})
    human_moves = game_data.get('human_moves', [])
    
    # Create tracking object with raw prediction data for charts
    tracking_data = {
        'models': ['random', 'frequency', 'markov', 'lstm'],
        'prediction_counts': {},
        'prediction_history': {},
        'total_predictions': len(human_moves) - 1 if len(human_moves) > 1 else 0,  # Predictions start from round 2
        'rounds_data': []
    }
    
    # For each model, get their prediction history
    for model in tracking_data['models']:
        predictions = model_predictions.get(model, [])
        tracking_data['prediction_history'][model] = predictions.copy()
        
        # Count predictions by move type
        tracking_data['prediction_counts'][model] = {
            'rock': predictions.count('rock'),
            'paper': predictions.count('paper'), 
            'scissors': predictions.count('scissors'),
            'total': len(predictions)
        }
    
    # Create round-by-round data for detailed tracking
    max_rounds = max(len(model_predictions.get(model, [])) for model in tracking_data['models']) if model_predictions else 0
    
    for round_num in range(max_rounds):
        round_data = {
            'round': round_num + 1,
            'predictions': {}
        }
        
        for model in tracking_data['models']:
            predictions = model_predictions.get(model, [])
            if round_num < len(predictions):
                round_data['predictions'][model] = predictions[round_num]
            else:
                round_data['predictions'][model] = None
                
        tracking_data['rounds_data'].append(round_data)
    
    return tracking_data

def create_centralized_ai_strategy_accuracy():
    """
    Create AI Strategy Accuracy object sourced from centralized data management.
    Calculates accuracy by comparing model predictions to actual human moves using raw data.
    """
    # Get data from centralized system  
    game_data = build_game_context(
        session=dict(session),
        overrides={},
        context_type='full'
    )
    
    # Extract raw data
    model_predictions = game_data.get('model_predictions_history', {})
    human_moves = game_data.get('human_moves', [])
    
    # Create accuracy object
    accuracy_data = {
        'models': ['random', 'frequency', 'markov', 'lstm'],
        'accuracy_percentages': {},
        'correct_predictions': {},
        'total_valid_predictions': {},
        'detailed_comparisons': {},
        'calculation_method': 'predictions[i] vs human_moves[i]'  # Correct method
    }
    
    # For each model, calculate accuracy correctly
    for model in accuracy_data['models']:
        predictions = model_predictions.get(model, [])
        
        # Initialize counters
        correct_count = 0
        total_valid = 0
        comparisons = []
        
        # Compare predictions[i] with human_moves[i+1] (correct method)
        for i in range(len(predictions)):
            if i + 1 < len(human_moves):  # Ensure we have the next human move
                predicted_move = predictions[i]
                actual_move = human_moves[i + 1]  # Prediction was for the NEXT move
                
                is_correct = predicted_move == actual_move
                if is_correct:
                    correct_count += 1
                total_valid += 1
                
                comparisons.append({
                    'prediction_round': i + 1,
                    'predicted': predicted_move,
                    'actual': actual_move,
                    'correct': is_correct,
                    'note': f'Prediction #{i} vs Human move #{i+1}'
                })
        
        # Calculate accuracy percentage
        accuracy_percentage = (correct_count / total_valid * 100) if total_valid > 0 else 0
        
        # Store results
        accuracy_data['accuracy_percentages'][model] = round(accuracy_percentage, 1)
        accuracy_data['correct_predictions'][model] = correct_count
        accuracy_data['total_valid_predictions'][model] = total_valid
        accuracy_data['detailed_comparisons'][model] = comparisons
    
    return accuracy_data

@app.route('/play', methods=['POST'])
def play():
    data = request.get_json(force=True)
    move = data.get('move')
    difficulty = data.get('difficulty', game_state['difficulty'])
    strategy_preference = data.get('strategy', 'to_win')
    personality = data.get('personality', 'neutral')
    multiplayer = bool(data.get('multiplayer', False))

    game_state['difficulty'] = difficulty
    game_state['ai_difficulty'] = difficulty
    game_state['multiplayer'] = multiplayer
    game_state['strategy_preference'] = strategy_preference
    game_state['personality'] = personality

    if move not in MOVES:
        return jsonify({'error': 'Invalid move'}), 400

    import random
    # Difficulty strategies
    def robot_strategy(history, difficulty, strategy_preference='to_win', personality='neutral'):
        # Performance tracking - start timing
        start_time = time.time()
        
        # Set the personality in the engine
        advanced_personalities = ['berserker', 'guardian', 'chameleon', 'professor', 'wildcard', 'mirror']
        
        if personality in advanced_personalities:
            personality_engine.set_personality(personality)
        
        # Predict next human move, then counter it
        predicted = None
        confidence = 0.33
        base_move = None  # Initialize base_move
        personality_modified_confidence = confidence  # Initialize as same as original
        
        # For random and frequency models, personality doesn't affect confidence
        use_personality_for_confidence = difficulty in ['markov', 'lstm']
        
        # Apply strategy preference to determine which approach to use
        if strategy_preference == 'to_win' and difficulty in ['markov', 'lstm']:
            # Force the AI to use the aggressive "to win" approach
            if len(history) >= 3:
                if difficulty == 'lstm' and LSTM_AVAILABLE and lstm_predictor:
                    # LSTM with TO_WIN strategy: confidence = abs(2 * highest_prob - 1)
                    try:
                        lstm_probs = lstm_predictor.predict(history)
                        highest_prob = max(lstm_probs.values())
                        confidence = abs(2 * highest_prob - 1)
                        base_move = lstm_predictor.get_counter_move(history)
                    except Exception as e:
                        print(f"LSTM prediction error in TO_WIN: {e}")
                        base_move = random.choice(MOVES)
                        confidence = 0.33
                else:
                    # Use ToWinStrategy for MARKOV or fallback
                    human_pred = to_win_strategy.predict(history)
                    confidence = to_win_strategy.get_confidence()
                    # Convert human prediction to robot counter move
                    counter = {'paper': 'scissors', 'scissors': 'rock', 'rock': 'paper'}
                    base_move = counter.get(human_pred) if human_pred else random.choice(MOVES)
            else:
                base_move = random.choice(MOVES)
                confidence = 0.33
            
        elif strategy_preference == 'not_to_lose' and difficulty in ['markov', 'lstm']:
            # Force the AI to use the defensive "not to lose" approach
            if len(history) >= 3:
                if difficulty == 'lstm' and LSTM_AVAILABLE and lstm_predictor:
                    # LSTM with NOT_TO_LOSE strategy: confidence = abs(2 * (sum of highest two probs) - 1)
                    try:
                        lstm_probs = lstm_predictor.predict(history)
                        sorted_probs = sorted(lstm_probs.values(), reverse=True)
                        highest_two_sum = sorted_probs[0] + sorted_probs[1]
                        confidence = abs(2 * highest_two_sum - 1)
                        base_move = lstm_predictor.get_counter_move(history)
                    except Exception as e:
                        print(f"LSTM prediction error in NOT_TO_LOSE: {e}")
                        base_move = random.choice(MOVES)
                        confidence = 0.33
                else:
                    # Use NotToLoseStrategy for MARKOV or fallback
                    human_pred = not_to_lose_strategy.predict(history)
                    confidence = not_to_lose_strategy.get_confidence()
                    # Convert human prediction to robot counter move
                    counter = {'paper': 'scissors', 'scissors': 'rock', 'rock': 'paper'}
                    base_move = counter.get(human_pred) if human_pred else random.choice(MOVES)
            else:
                base_move = random.choice(MOVES)
                confidence = 0.33
            
        else:
            # Regular difficulty-based strategy
            if difficulty == 'random':
                predicted = random.choice(MOVES)
                confidence = 0.0  # Random strategy has no confidence
            elif difficulty == 'frequency':
                predicted_counter = frequency_strategy.predict(history)
                # Convert back to predicted human move
                reverse_counter = {'scissors': 'paper', 'rock': 'scissors', 'paper': 'rock'}
                predicted = reverse_counter.get(predicted_counter, random.choice(MOVES))
                confidence = 0.0  # Frequency strategy has no confidence
            elif difficulty == 'markov':
                markov_strategy.train(history)
                predicted_counter = markov_strategy.predict(history)
                reverse_counter = {'scissors': 'paper', 'rock': 'scissors', 'paper': 'rock'}
                predicted = reverse_counter.get(predicted_counter, random.choice(MOVES))
                confidence = 0.6
            elif difficulty == 'lstm' and LSTM_AVAILABLE and lstm_predictor:
                # LSTM prediction with strategy-specific confidence calculation
                try:
                    # Get LSTM probabilities for human moves
                    lstm_probs = lstm_predictor.predict(history)
                    
                    # Calculate strategy-specific confidence
                    if strategy_preference == 'to_win':
                        # ToWin strategy: confidence = abs(2 * highest_prob - 1)
                        highest_prob = max(lstm_probs.values())
                        confidence = abs(2 * highest_prob - 1)
                    elif strategy_preference == 'not_to_lose':
                        # NotToLose strategy: confidence = abs(2 * (sum of highest two probs) - 1)
                        sorted_probs = sorted(lstm_probs.values(), reverse=True)
                        highest_two_sum = sorted_probs[0] + sorted_probs[1]
                        confidence = abs(2 * highest_two_sum - 1)
                    else:
                        # Default to original LSTM confidence
                        confidence = max(lstm_probs.values())
                    
                    # Get the counter move
                    base_move = lstm_predictor.get_counter_move(history)
                    
                    # Track LSTM inference time
                    inference_duration = time.time() - start_time
                    if DEVELOPER_CONSOLE_AVAILABLE:
                        track_inference('lstm', inference_duration)
                    
                    # Don't return early - continue to personality application
                    predicted = None  # Set to None since we have base_move
                except Exception as e:
                    print(f"LSTM prediction error: {e}")
                    predicted = random.choice(MOVES)
                    confidence = 0.33
            else:
                predicted = random.choice(MOVES)
                confidence = 0.0  # Default to no confidence
            
            # Convert prediction to robot move (skip for LSTM as it already has base_move)
            if difficulty != 'lstm' or not base_move:
                counter = {'paper': 'scissors', 'scissors': 'rock', 'rock': 'paper'}
                base_move = counter.get(predicted) if predicted else random.choice(MOVES)
        
        # Track inference time for non-LSTM models
        inference_duration = time.time() - start_time
        if DEVELOPER_CONSOLE_AVAILABLE:
            track_inference(difficulty, inference_duration)
        
        # Apply personality modifications using the advanced engine
        if personality in advanced_personalities and use_personality_for_confidence:
            # Create game history in the format expected by personality engine
            game_history = []
            if len(game_state['human_history']) == len(game_state['robot_history']):
                game_history = list(zip(game_state['human_history'], game_state['robot_history']))
            
            final_move, personality_modified_confidence = personality_engine.apply_personality_to_move(
                base_move or random.choice(MOVES), confidence, history, game_history
            )
            return final_move, confidence, personality_modified_confidence
        elif personality in advanced_personalities and not use_personality_for_confidence:
            # For random/frequency models, apply personality to move but not confidence
            game_history = []
            if len(game_state['human_history']) == len(game_state['robot_history']):
                game_history = list(zip(game_state['human_history'], game_state['robot_history']))
            
            final_move, _ = personality_engine.apply_personality_to_move(
                base_move or random.choice(MOVES), confidence, history, game_history
            )
            # For random/frequency, personality-modified confidence equals original confidence
            return final_move, confidence, confidence
        
        # Legacy personality modifiers for backwards compatibility
        elif personality == 'aggressive':
            confidence = min(1.0, confidence * 1.2)  # More confident
            return base_move, confidence, confidence  # Legacy personalities don't modify confidence differently
        elif personality == 'defensive':
            confidence = max(0.1, confidence * 0.8)  # Less confident
            return base_move, confidence, confidence
        elif personality == 'chaotic':
            if random.random() < 0.3:  # 30% chance to be completely random
                return random.choice(MOVES), 0.33, 0.33
            return base_move, confidence, confidence
        elif personality == 'adaptive':
            # Simple adaptive logic for legacy support
            if len(game_state['result_history']) >= 5:
                recent_results = game_state['result_history'][-5:]
                robot_wins = recent_results.count('robot')
                if robot_wins <= 1:  # Losing, be more aggressive
                    return base_move, min(1.0, confidence * 1.3), min(1.0, confidence * 1.3)
            return base_move, confidence, confidence
        
        return base_move, confidence, confidence  # No personality modification

    results = []
    robot_move, confidence, personality_modified_confidence = robot_strategy(
        game_state['human_history'], 
        difficulty, 
        strategy_preference, 
        personality
    )
    
    # Track predictions from all models for comparison chart
    if len(game_state['human_history']) > 0:  # Only track after first move
        history = game_state['human_history']
        
        # Get predictions from all models
        import random
        
        # Random prediction
        random_pred = random.choice(MOVES)
        game_state['model_predictions_history']['random'].append(random_pred)
        
        # Frequency prediction
        freq_robot_move = frequency_strategy.predict(history)  # This returns robot's move
        # Convert robot move back to what human move it was expecting
        reverse_counter = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
        freq_pred = reverse_counter.get(freq_robot_move, random.choice(MOVES))
        game_state['model_predictions_history']['frequency'].append(freq_pred)
        
        # Markov prediction
        markov_strategy.train(history)
        markov_result = markov_strategy.predict(history)  # This returns (robot_move, confidence)
        markov_robot_move = markov_result[0] if isinstance(markov_result, tuple) else markov_result
        # Convert robot move back to what human move it was expecting
        reverse_counter = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
        markov_pred = reverse_counter.get(markov_robot_move, random.choice(MOVES))
        game_state['model_predictions_history']['markov'].append(markov_pred)
        
        # LSTM prediction (if available)
        if LSTM_AVAILABLE and lstm_predictor:
            try:
                lstm_probs = lstm_predictor.predict(history)
                lstm_pred = max(lstm_probs.items(), key=lambda x: x[1])[0]  # Most likely human move
                game_state['model_predictions_history']['lstm'].append(lstm_pred)
            except Exception as e:
                game_state['model_predictions_history']['lstm'].append(random.choice(MOVES))
        else:
            game_state['model_predictions_history']['lstm'].append(random.choice(MOVES))
        
        # Track confidence values for each model - Fixed to use model-specific confidence
        # Random and Frequency should always be 0%
        game_state['model_confidence_history']['random'].append(0.0)
        game_state['model_confidence_history']['frequency'].append(0.0)
        
        # Get actual model predictions and apply strategy-specific confidence formulas
        strategy_preference = game_state.get('strategy_preference', 'to_win')
        
        # Markov model - get its actual probability distribution
        if len(history) > 0:
            markov_strategy.train(history)
            markov_pred, markov_raw_confidence = markov_strategy.predict(history)
            
            # Apply strategy-specific confidence formula using model's actual probabilities
            # Note: markov_raw_confidence comes from the model's internal probability calculation
            if strategy_preference == 'to_win':
                # Formula: abs(2*highest_prob-1)
                markov_confidence = abs(2 * markov_raw_confidence - 1)
            elif strategy_preference == 'not_to_lose':
                # For "not to lose", we need to estimate the sum of top two probabilities
                # Since we only have the highest probability, we estimate the second highest
                # Assuming uniform distribution for remaining probability mass
                remaining_prob = 1 - markov_raw_confidence
                second_highest = remaining_prob / 2  # Rough estimation
                markov_confidence = abs(2 * (markov_raw_confidence + second_highest) - 1)
            else:
                markov_confidence = markov_raw_confidence
        else:
            markov_confidence = 0.33  # Default for no history
        
        game_state['model_confidence_history']['markov'].append(markov_confidence)
        
        # LSTM confidence - use its own confidence calculation with strategy formulas
        if LSTM_AVAILABLE and lstm_predictor:
            try:
                lstm_raw_confidence = lstm_predictor.get_confidence(history)
                
                # Apply strategy-specific confidence formula
                if strategy_preference == 'to_win':
                    lstm_confidence = abs(2 * lstm_raw_confidence - 1)
                elif strategy_preference == 'not_to_lose':
                    # Estimate second highest probability
                    remaining_prob = 1 - lstm_raw_confidence
                    second_highest = remaining_prob / 2
                    lstm_confidence = abs(2 * (lstm_raw_confidence + second_highest) - 1)
                else:
                    lstm_confidence = lstm_raw_confidence
                game_state['model_confidence_history']['lstm'].append(lstm_confidence)
            except:
                game_state['model_confidence_history']['lstm'].append(0.0)
        else:
            game_state['model_confidence_history']['lstm'].append(0.0)
        
        game_state['model_confidence_history']['to_win'].append(to_win_strategy.get_confidence())
        game_state['model_confidence_history']['not_to_lose'].append(not_to_lose_strategy.get_confidence())
    
    # Store confidence for response
    game_state['last_confidence'] = confidence
    if multiplayer:
        move2 = data.get('move2', move)
        result1 = get_result(robot_move, move)
        result2 = get_result(robot_move, move2)
        game_state['human_history'].append(move)
        game_state['human_history'].append(move2)
        game_state['robot_history'].append(robot_move)
        game_state['robot_history'].append(robot_move)
        game_state['result_history'].append(result1)
        game_state['result_history'].append(result2)
        game_state['confidence_score_history'].append(confidence)  # Same confidence for both moves
        game_state['confidence_score_history'].append(confidence)
        game_state['confidence_score_modified_by_personality_history'].append(personality_modified_confidence)  # Track personality-modified confidence for both moves
        game_state['confidence_score_modified_by_personality_history'].append(personality_modified_confidence)
        game_state['round_history'].append({'round': game_state['round']+1, 'human': move, 'robot': robot_move})
        game_state['round_history'].append({'round': game_state['round']+2, 'human': move2, 'robot': robot_move})
        game_state['round'] += 2
        if result1 == 'human':
            game_state['stats']['human_win'] += 1
        elif result1 == 'robot':
            game_state['stats']['robot_win'] += 1
        else:
            game_state['stats']['tie'] += 1
        if result2 == 'human':
            game_state['stats']['human_win'] += 1
        elif result2 == 'robot':
            game_state['stats']['robot_win'] += 1
        else:
            game_state['stats']['tie'] += 1
        results = [result1, result2]
    else:
        result = get_result(robot_move, move)
        game_state['human_history'].append(move)
        game_state['robot_history'].append(robot_move)
        game_state['result_history'].append(result)
        game_state['confidence_score_history'].append(confidence)  # Track confidence for each round
        game_state['confidence_score_modified_by_personality_history'].append(personality_modified_confidence)  # Track personality-modified confidence
        game_state['round_history'].append({'round': game_state['round']+1, 'human': move, 'robot': robot_move})
        game_state['round'] += 1
        
        # Update personality engine with game result
        personality_engine.update_game_state(move, robot_move or 'paper', result, confidence)
        
        # Add move to change detector for strategy analysis
        change_result = change_detector.add_move(move)
        
        change_points = []
        current_strategy = 'warming up'
        if len(game_state['human_history']) >= 5:  # Need minimum moves for analysis
            change_points = change_detector.get_all_change_points()
            current_strategy = change_detector.get_current_strategy_label()
            game_state['change_points'] = change_points
            game_state['human_strategy_label'] = current_strategy
            game_state['current_strategy'] = current_strategy  # Legacy support for existing UI
        
        if result == 'human':
            game_state['stats']['human_win'] += 1
        elif result == 'robot':
            game_state['stats']['robot_win'] += 1
        else:
            game_state['stats']['tie'] += 1
        
        results = [result]
    
    # Calculate model accuracy after each move (only if we have predictions to compare)
    if len(game_state['human_history']) > 1:  # Need at least 2 moves to have a prediction
        current_human_move = game_state['human_history'][-1]  # Just played move
        
        # Calculate accuracy for each model
        for model_name in game_state['model_predictions_history']:
            predictions = game_state['model_predictions_history'][model_name]
            if len(predictions) > 0:
                # Compare predictions to actual human moves
                correct_predictions = 0
                total_predictions = min(len(predictions), len(game_state['human_history']))
                
                for i in range(total_predictions):
                    if predictions[i] == game_state['human_history'][i]:  # Direct comparison of same indices
                        correct_predictions += 1
                
                # Calculate accuracy percentage
                accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
                game_state['accuracy'][model_name] = round(accuracy, 1)
    
    # Track game move in developer console
    if DEVELOPER_CONSOLE_AVAILABLE and len(game_state['human_history']) > 0:
        # Get all model predictions for this move
        model_predictions = {}
        model_confidences = {}
        
        for model_name in game_state['model_predictions_history']:
            if game_state['model_predictions_history'][model_name]:
                model_predictions[model_name] = game_state['model_predictions_history'][model_name][-1]
        
        for model_name in game_state['model_confidence_history']:
            if game_state['model_confidence_history'][model_name]:
                model_confidences[model_name] = game_state['model_confidence_history'][model_name][-1]
        
        # Track the complete move
        track_move(
            human_move=game_state['human_history'][-1],
            robot_move=game_state['robot_history'][-1] if game_state['robot_history'] else 'unknown',
            result=game_state['result_history'][-1] if game_state['result_history'] else 'unknown',
            model_predictions=model_predictions,
            model_confidences=model_confidences
        )
    
    # Store game data in session for AI coach access
    session['human_moves'] = game_state['human_history']
    session['robot_moves'] = game_state['robot_history']
    session['results'] = game_state['result_history']
    session['confidence_score_history'] = game_state['confidence_score_history']
    session['confidence_score_modified_by_personality_history'] = game_state['confidence_score_modified_by_personality_history']
    session['difficulty'] = game_state['difficulty']
    session['ai_difficulty'] = game_state.get('ai_difficulty', game_state['difficulty'])
    session['current_strategy'] = game_state.get('current_strategy', 'unknown')
    session['human_strategy_label'] = game_state.get('human_strategy_label', game_state.get('current_strategy', 'unknown'))
    session['round_count'] = game_state['round']
    session['strategy_preference'] = game_state.get('strategy_preference', 'to_win')
    session['personality'] = game_state.get('personality', 'neutral')
    session['multiplayer'] = game_state.get('multiplayer', False)
    session['accuracy'] = game_state['accuracy']
    session['confidence'] = confidence
    session['change_points'] = game_state.get('change_points', [])
    session['model_predictions_history'] = game_state['model_predictions_history']
    session['model_confidence_history'] = game_state['model_confidence_history']
    session['correct_predictions'] = game_state['correct_predictions']
    session['total_predictions'] = game_state['total_predictions']
    
    # Build unified game context with metrics (CRITICAL for frontend real-time updates)
    game_data = build_game_context(dict(session))  # Match download_game_context pattern
    
    # Return updated state for AJAX with unified metrics
    return jsonify({
        'stats': game_state['stats'],
        'human_history': game_state['human_history'],
        'robot_history': game_state['robot_history'],
        'result_history': game_state['result_history'],
        'round_history': game_state['round_history'],
        'round': game_state['round'],
        'robot_move': robot_move,
        'result': results,
        'difficulty': game_state['difficulty'],
        'ai_difficulty': game_state.get('ai_difficulty', game_state['difficulty']),
        'strategy_preference': game_state.get('strategy_preference', 'to_win'),
        'personality': game_state.get('personality', 'neutral'),
        'multiplayer': game_state['multiplayer'],
        'accuracy': game_state['accuracy'],
        'confidence': confidence,
        'change_points': game_state.get('change_points', []),
        'current_strategy': game_state.get('current_strategy', 'unknown'),
        'human_strategy_label': game_state.get('human_strategy_label', game_state.get('current_strategy', 'unknown')),
        'model_predictions_history': game_state['model_predictions_history'],
        'model_confidence_history': game_state['model_confidence_history'],
        'correct_predictions': game_state['correct_predictions'],
        'total_predictions': game_state['total_predictions'],
        'centralized_prediction_tracking': create_centralized_model_prediction_tracking(),  # New centralized object
        'centralized_ai_strategy_accuracy': create_centralized_ai_strategy_accuracy(),  # New centralized object
        # *** CRITICAL: Include unified metrics for frontend consumption ***
        'metrics': game_data.get('game_status', {}).get('metrics', {}),  # Extract from game_status
        'game_status': game_data.get('game_status', {}),
        'full_game_snapshot': game_data.get('full_game_snapshot', {})
    })

@app.route('/coaching', methods=['GET', 'POST'])
def get_coaching_tips():
    """Get intelligent coaching tips - Under Development"""
    return jsonify({
        'status': 'under_development',
        'coaching_tips': ['AI Coach is being redesigned with better specifications.'],
        'experiments': [],
        'insights': {'message': 'Enhanced coaching features coming soon.'},
        'round': session.get('round', 0),
        'current_strategy': 'under_development',
        'llm_type': 'placeholder',
        'enhanced_analysis': {},
        'behavioral_insights': {},
        'confidence': 0.0
    })


@app.route('/stats', methods=['GET'])
def stats():
    """Return stats as JSON for AJAX using centralized data management"""
    
    # 🎯 CENTRALIZED DATA CONSTRUCTION: Use the centralized game context builder
    game_data = build_game_context(
        session=dict(session),  # Convert Flask session to dict
        overrides={},  # No overrides for stats endpoint
        context_type='full'  # Get comprehensive context with all available data
    )
    
    # Return stats using centralized data
    return jsonify({
        'stats': game_state['stats'],  # Stats still from game_state as they're not in context yet
        'human_history': game_data['human_moves'],  # Centralized field name
        'robot_history': game_data['robot_moves'],   # Centralized field name
        'result_history': game_data['results'],      # Centralized field name
        'round': game_data['round'],
        'accuracy': game_data.get('accuracy', {}),
        'ai_difficulty': game_data['ai_difficulty'],
        'human_strategy_label': game_data['human_strategy_label']
    })

@app.route('/history', methods=['GET'])
def history():
    """Return full game history including round details using centralized data management"""
    
    # 🎯 CENTRALIZED DATA CONSTRUCTION: Use the centralized game context builder
    game_data = build_game_context(
        session=dict(session),  # Convert Flask session to dict
        overrides={},  # No overrides for history endpoint
        context_type='full'  # Get comprehensive context with all available data
    )
    
    # The centralized context already contains most of what we need
    # Add any additional fields that might be missing
    response_data = {
        'stats': game_state['stats'],  # Stats still from game_state as they're not in context yet
        'human_history': game_data['human_moves'],  # Centralized field name
        'robot_history': game_data['robot_moves'],   # Centralized field name
        'result_history': game_data['results'],      # Centralized field name
        'round_history': game_state.get('round_history', []),  # This might not be in context yet
        'round': game_data['round'],
        'accuracy': game_data.get('accuracy', {}),
        'change_points': game_data.get('change_points', []),
        'current_strategy': game_data['human_strategy_label'],
        'human_strategy_label': game_data['human_strategy_label'],
        'ai_difficulty': game_data['ai_difficulty'],
        'centralized_prediction_tracking': create_centralized_model_prediction_tracking(),  # New centralized object
        'centralized_ai_strategy_accuracy': create_centralized_ai_strategy_accuracy()  # New centralized object
    }
    
    return jsonify(response_data)

@app.route('/reset', methods=['POST', 'GET'])
def reset():
    game_state['human_history'].clear()
    game_state['robot_history'].clear()
    game_state['result_history'].clear()
    game_state['round'] = 0
    game_state['stats'] = {'human_win': 0, 'robot_win': 0, 'tie': 0}
    game_state['difficulty'] = 'markov'  # Default to markov
    game_state['ai_difficulty'] = 'markov'
    game_state['multiplayer'] = False
    game_state['change_points'] = []
    game_state['current_strategy'] = 'unknown'
    game_state['human_strategy_label'] = 'unknown'
    
    # Reset change detector
    change_detector.reset()
    
    if request.method == 'POST':
        return jsonify({'success': True})
    return redirect(url_for('index'))

def get_result(robot_move, human_move):
    if robot_move == human_move:
        return 'tie'
    elif (robot_move == 'paper' and human_move == 'rock') or \
         (robot_move == 'rock' and human_move == 'scissors') or \
         (robot_move == 'scissor' and human_move == 'paper'):
        return 'robot'
    else:
        return 'human'

@app.route('/personality/info/<personality_name>')
def get_personality_info(personality_name):
    """Get detailed information about a personality"""
    try:
        personality_info = personality_engine.get_personality_info(personality_name)
        if not personality_info:
            return jsonify({'error': 'Personality not found'}), 404
        return jsonify(personality_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/personality/stats')
def get_personality_stats():
    """Get current personality performance stats"""
    try:
        stats = personality_engine.get_personality_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/personality/list')
def list_personalities():
    """Get list of all available personalities"""
    try:
        personalities = personality_engine.get_all_personalities()
        personality_details = {}
        for name in personalities:
            personality_details[name] = personality_engine.get_personality_info(name)
        return jsonify(personality_details)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_game():
    # Reset game state
    game_state['human_history'] = []
    game_state['robot_history'] = []
    game_state['result_history'] = []
    game_state['round_history'] = []
    game_state['round'] = 0
    game_state['stats'] = {'human_win': 0, 'robot_win': 0, 'tie': 0}
    game_state['change_points'] = []
    game_state['confidence_score_history'] = []
    game_state['confidence_score_modified_by_personality_history'] = []

    # Reset session game context and recording flag
    session['human_moves'] = []
    session['robot_moves'] = []
    session['results'] = []
    session['confidence_score_history'] = []
    session['confidence_score_modified_by_personality_history'] = []
    session['round_count'] = 0
    session['stats'] = {'human_win': 0, 'robot_win': 0, 'tie': 0}
    session['change_points'] = []
    session['model_predictions_history'] = {}
    session['model_confidence_history'] = {}
    session['correct_predictions'] = {}
    session['total_predictions'] = {}
    session['strategy_preference'] = 'to_win'
    session['personality'] = 'neutral'
    session['ai_difficulty'] = 'unknown'
    session['game_length'] = None
    session['game_recording'] = False
    session['analytics_metrics'] = {}
    session['endgame_metrics'] = {}

    return jsonify({'message': 'Game reset successfully'})

@app.route('/developer', methods=['GET'])
def developer_console():
    """Developer console interface with comprehensive debugging and monitoring."""
    if not DEVELOPER_CONSOLE_AVAILABLE:
        return jsonify({'error': 'Developer console not available'}), 503
    
    report = get_developer_report()
    chart = get_chart()
    
    return render_template('developer_console.html', 
                         report=report, 
                         chart=chart,
                         session_active=True)

@app.route('/developer/api/report', methods=['GET'])
def developer_api_report():
    """API endpoint for developer report data."""
    if not DEVELOPER_CONSOLE_AVAILABLE:
        return jsonify({'error': 'Developer console not available'}), 503
    
    return jsonify(get_developer_report())

@app.route('/developer/api/chart', methods=['GET'])
def developer_api_chart():
    """API endpoint for model comparison chart."""
    if not DEVELOPER_CONSOLE_AVAILABLE:
        return jsonify({'error': 'Developer console not available'}), 503
    
    chart_data = get_chart()
    return jsonify({'chart': chart_data})

@app.route('/developer/api/export', methods=['POST'])
def developer_api_export():
    """Export session data for analysis."""
    if not DEVELOPER_CONSOLE_AVAILABLE:
        return jsonify({'error': 'Developer console not available'}), 503
    
    try:
        timestamp = int(time.time())
        filename = f'session_export_{timestamp}.json'
        filepath = os.path.join(os.path.dirname(__file__), '..', filename)
        
        console.export_session_data(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'message': f'Session data exported to {filename}'
        })
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/performance', methods=['GET'])
def performance_dashboard():
    """Performance optimization dashboard."""
    if not PERFORMANCE_OPTIMIZER_AVAILABLE:
        return jsonify({'error': 'Performance optimizer not available'}), 503
    
    report = get_performance_report()
    return render_template('performance_dashboard.html', report=report)

@app.route('/performance/api/report', methods=['GET'])
def performance_api_report():
    """API endpoint for performance report."""
    if not PERFORMANCE_OPTIMIZER_AVAILABLE:
        return jsonify({'error': 'Performance optimizer not available'}), 503
    
    return jsonify(get_performance_report())

@app.route('/performance/api/bundle-analysis', methods=['GET'])
def performance_api_bundle():
    """API endpoint for bundle size analysis."""
    if not PERFORMANCE_OPTIMIZER_AVAILABLE:
        return jsonify({'error': 'Performance optimizer not available'}), 503
    
    return jsonify(optimizer.bundle_analyzer.analyze_file_sizes())

@app.route('/performance/api/timing/<model_name>', methods=['GET'])
def performance_api_timing(model_name):
    """API endpoint for model timing analysis."""
    if not PERFORMANCE_OPTIMIZER_AVAILABLE:
        return jsonify({'error': 'Performance optimizer not available'}), 503
    
    return jsonify(optimizer.timing_validator.get_timing_analysis(model_name))


# AI Coach API Endpoints - Simplified for Development
@app.route('/ai_coach/status')
def ai_coach_status():
    """Get AI coach availability status"""
    return jsonify({
        'available': False,
        'features': {
            'metrics_aggregation': False,
            'enhanced_coaching': False,
            'langchain_integration': False
        },
        'status': 'under_development',
        'message': 'AI Coach is being redesigned. Coach View frontend remains available.'
    })


@app.route('/ai_coach/realtime', methods=['POST'])
def ai_coach_realtime():
    """Get real-time AI coaching advice - Under Development"""
    return jsonify({
        'status': 'under_development',
        'message': 'Real-time AI coaching is being redesigned with improved specifications.',
        'coaching_advice': {
            'primary_tip': 'AI Coach functionality is temporarily unavailable.',
            'reasoning': 'The system is being rebuilt with better architecture.',
            'confidence': 0.0,
            'risk_assessment': 'Feature under development'
        }
    })


@app.route('/ai_coach/comprehensive', methods=['POST'])
def ai_coach_comprehensive():
    """Get comprehensive AI coaching analysis - Under Development"""
    return jsonify({
        'status': 'under_development',
        'message': 'Comprehensive AI coaching analysis is being redesigned.',
        'analysis': {
            'pattern_insights': 'Feature under development',
            'strategic_recommendations': 'AI Coach is being rebuilt',
            'performance_analysis': 'Enhanced specifications coming soon',
            'confidence': 0.0
        }
    })


@app.route('/ai_coach/metrics', methods=['GET'])
def ai_coach_metrics():
    """Get AI coach metrics - Under Development"""
    return jsonify({
        'status': 'under_development',
        'message': 'AI Coach metrics are being redesigned.',
        'metrics': {
            'availability': False,
            'features': 'Under development'
        }
    })


@app.route('/ai_coach_demo')
def ai_coach_demo():
    """AI Coach demonstration page - Placeholder"""
    return jsonify({
        'status': 'under_development',
        'message': 'AI Coach demo is being redesigned. Use Coach View in main interface.',
        'redirect': '/'
    })


@app.route('/ai_coach/toggle_mode', methods=['POST'])
def ai_coach_toggle_mode():
    """Toggle AI coach mode - Under Development"""
    return jsonify({
        'status': 'under_development',
        'message': 'AI Coach mode toggle is being redesigned.'
    })


@app.route('/ai_coach/set_style', methods=['POST'])
def ai_coach_set_style():
    """Set coaching style - Under Development"""
    return jsonify({
        'status': 'under_development',
        'message': 'Coaching style settings are being redesigned.'
    })


@app.route('/ai_coach/get_style', methods=['GET'])
def ai_coach_get_style():
    """Get coaching style - Under Development"""
    return jsonify({
        'status': 'under_development',
        'message': 'Coaching style retrieval is being redesigned.'
    })


@app.route('/ai_coach/enhanced_analysis', methods=['POST'])
def ai_coach_enhanced_analysis():
    """Enhanced analysis - Under Development"""
    return jsonify({
        'status': 'under_development',
        'message': 'Enhanced analysis features are being redesigned.'
    })


@app.route('/ai_coach/llm_comparison', methods=['POST'])
def ai_coach_llm_comparison():
    """LLM comparison - Under Development"""
    return jsonify({
        'status': 'under_development',
        'message': 'LLM comparison features are being redesigned.'
    })


# Game Context Download endpoint (used by Coach View)
@app.route('/download_game_context')
def download_game_context():
    from game_context import build_game_context
    context = build_game_context(dict(session))
    return jsonify(context)

import threading
import webbrowser
import time

def open_browser():
    time.sleep(1)  # Wait a moment for the server to start
    webbrowser.open('http://192.168.0.229:5050')

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    threading.Thread(target=open_browser).start()
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=5050)
