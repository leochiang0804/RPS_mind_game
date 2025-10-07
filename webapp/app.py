
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from change_point_detector import ChangePointDetector
from optimized_strategies import ToWinStrategy, NotToLoseStrategy
from personality_engine import get_personality_engine
from move_mapping import MOVES
from ml_model_enhanced import EnhancedMLModel

# Centralized Data Management - All AI coach endpoints use this for consistent data building
from game_context import build_game_context, set_opponent_parameters, get_ai_prediction, update_ai_with_result, reset_ai_system

# Import SBC Backend for enhanced banter and coaching
try:
    from sbc_backend import get_sbc_backend
    SBC_BACKEND_AVAILABLE = True
    print("✅ SBC Backend available")
except ImportError as e:
    SBC_BACKEND_AVAILABLE = False
    print(f"⚠️ SBC Backend not available: {e}")

# Import the adaptive RPS AI system
try:
    from rps_ai_system import get_ai_system, initialize_ai_system
    RPS_AI_SYSTEM_AVAILABLE = True
    print("✅ Adaptive RPS AI system available")
except ImportError as e:
    RPS_AI_SYSTEM_AVAILABLE = False
    print(f"⚠️ Adaptive RPS AI system not available: {e}")

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

# LSTM functionality replaced by adaptive system
LSTM_AVAILABLE = False
print("ℹ️ LSTM replaced by adaptive system")

# AI Coach functionality - Currently under development
# Coach View frontend is available, but backend AI Coach features are being redesigned
AI_COACH_AVAILABLE = False
print("🚧 AI Coach under development - Coach View available in frontend")

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-for-ai-coach-sessions-2024'

# Initialize enhanced predictor used for analytics and fallback displays
def _create_enhanced_predictor() -> EnhancedMLModel:
    return EnhancedMLModel(order=3, recency_weight=0.82, max_history=220)

enhanced_predictor = _create_enhanced_predictor()
to_win_strategy = ToWinStrategy()
not_to_lose_strategy = NotToLoseStrategy()

# LSTM predictor is replaced by adaptive system
lstm_predictor = None

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
    'difficulty': 'challenger',  # Default to challenger difficulty
    'ai_difficulty': 'challenger',  # Explicit AI difficulty label
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
    # Adaptive AI System
    def robot_strategy(history, difficulty, strategy_preference='to_win', personality='neutral'):
        """
        Updated robot strategy using the adaptive opponent engine.
        """
        start_time = time.time()
        
        # Check if new AI system is available
        if RPS_AI_SYSTEM_AVAILABLE:
            try:
                # Set opponent parameters in the AI system
                success = set_opponent_parameters(difficulty, strategy_preference, personality)
                
                if success:
                    # Get prediction from the AI system
                    session_data = {
                        'human_moves': history,
                        'results': game_state.get('result_history', []),
                        'ai_difficulty': difficulty,
                        'strategy_preference': strategy_preference,
                        'personality': personality
                    }
                    
                    prediction_data = get_ai_prediction(session_data)
                    
                    # Extract data
                    ai_move = prediction_data.get('ai_move', random.choice(MOVES))
                    confidence = prediction_data.get('confidence', 0.33)
                    
                    # Track inference time
                    inference_duration = time.time() - start_time
                    if DEVELOPER_CONSOLE_AVAILABLE:
                        track_inference(f'{difficulty}_{strategy_preference}_{personality}', inference_duration)
                    
                    # Return: base_move, confidence, personality_modified_confidence
                    # In the new system, confidence already includes personality modifications
                    return ai_move, confidence, confidence
                    
            except Exception as e:
                print(f"Error using adaptive system: {e}")
                # Fall through to legacy system
        
        # Fallback to a lightweight random policy if the adaptive engine is unavailable
        fallback_move = random.choice(MOVES)
        fallback_confidence = 0.2

        inference_duration = time.time() - start_time
        if DEVELOPER_CONSOLE_AVAILABLE:
            try:
                track_inference(f'legacy_{difficulty}', inference_duration)
            except Exception:
                pass

        return fallback_move, fallback_confidence, fallback_confidence


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
        reverse_counter = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
        
        # Random prediction
        random_pred = random.choice(MOVES)
        game_state['model_predictions_history']['random'].append(random_pred)
        
        # Frequency-style prediction using direct move counts
        freq_pred = random_pred
        if history:
            counts = Counter(history)
            most_common_human, _ = counts.most_common(1)[0]
            freq_pred = most_common_human
        game_state['model_predictions_history']['frequency'].append(freq_pred)
        
        # Enhanced (Markov-inspired) prediction using the upgraded ML model
        enhanced_predictor.train(history)
        markov_robot_move, markov_raw_confidence = enhanced_predictor.predict(history)
        markov_raw_confidence = float(markov_raw_confidence or 0.0)
        markov_pred = reverse_counter.get(markov_robot_move, random.choice(MOVES))
        game_state['model_predictions_history']['markov'].append(markov_pred)
        
        # LSTM prediction - replaced by adaptive system
        game_state['model_predictions_history']['lstm'].append(random.choice(MOVES))
        
        # Track confidence values for each model - Fixed to use model-specific confidence
        # Random and Frequency should always be 0%
        game_state['model_confidence_history']['random'].append(0.0)
        game_state['model_confidence_history']['frequency'].append(0.0)
        
        # Get actual model predictions and apply strategy-specific confidence formulas
        strategy_preference = game_state.get('strategy_preference', 'to_win')
        
        # Convert enhanced predictor confidence using strategy-specific formulas
        if markov_raw_confidence <= 0:
            markov_confidence = 0.33
        elif strategy_preference == 'to_win':
            # Formula: abs(2*highest_prob-1)
            markov_confidence = abs(2 * markov_raw_confidence - 1)
        elif strategy_preference == 'not_to_lose':
            remaining_prob = max(0.0, 1 - markov_raw_confidence)
            second_highest = remaining_prob / 2  # Rough estimation
            markov_confidence = abs(2 * (markov_raw_confidence + second_highest) - 1)
        else:
            markov_confidence = markov_raw_confidence
        
        game_state['model_confidence_history']['markov'].append(markov_confidence)
        
        # LSTM confidence - replaced by adaptive system
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
        
        # Update the adaptive AI system with the result
        if RPS_AI_SYSTEM_AVAILABLE:
            try:
                update_ai_with_result(move, robot_move)
            except Exception as e:
                print(f"Error updating AI system: {e}")
        
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

    # Ensure confidence histories do not exceed total rounds
    total_rounds_logged = len(game_state['result_history'])
    if total_rounds_logged:
        history_limit = slice(-total_rounds_logged, None)
        game_state['confidence_score_history'] = game_state['confidence_score_history'][history_limit]
        game_state['confidence_score_modified_by_personality_history'] = (
            game_state['confidence_score_modified_by_personality_history'][history_limit]
        )

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
    if DEVELOPER_CONSOLE_AVAILABLE:
        try:
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
        except:
            pass  # Silently fail if developer console functions are not available
    
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
    """Get intelligent coaching tips powered by SBC Backend"""
    if not SBC_BACKEND_AVAILABLE:
        return jsonify({
            'status': 'unavailable',
            'coaching_tips': ['SBC Backend not available. Coach is under development.'],
            'experiments': [],
            'insights': {'message': 'Enhanced coaching features require SBC Backend.'},
            'round': session.get('round', 0),
            'current_strategy': 'under_development',
            'llm_type': 'placeholder',
            'enhanced_analysis': {},
            'behavioral_insights': {},
            'confidence': 0.0
        })
    
    try:
        # Build current game context
        game_context = build_game_context(session)
        
        # Get SBC backend instance
        sbc = get_sbc_backend()
        
        # Generate 3 coaching tips using SBC backend
        coaching_seeds = sbc.rule_selector.select_coaching_seeds(game_context, 3)
        
        coaching_tips = []
        for seed_data in coaching_seeds:
            prompt = sbc._create_coaching_prompt(seed_data, game_context)
            tip_text = sbc.model_adapter.generate_response(prompt, 'professor')
            
            coaching_tips.append({
                'tip': tip_text,
                'reason': seed_data['reason_key'],
                'priority': seed_data['priority'],
                'specific_advice': seed_data.get('specific_advice', ''),
                'category': seed_data['tip_key']
            })
        
        # Sort by priority (highest first)
        coaching_tips.sort(key=lambda x: x['priority'], reverse=True)
        
        return jsonify({
            'status': 'active',
            'coaching_tips': [tip['tip'] for tip in coaching_tips],  # For backward compatibility
            'enhanced_tips': coaching_tips,  # New detailed format
            'total_tips': len(coaching_tips),
            'experiments': [],
            'insights': {
                'message': f'Generated {len(coaching_tips)} coaching tips using SBC Backend',
                'engine': sbc.model_adapter.engine.value,
                'personality': 'professor'
            },
            'round': session.get('round', 0),
            'current_strategy': 'sbc_powered',
            'llm_type': sbc.model_adapter.engine.value,
            'enhanced_analysis': game_context.get('game_status', {}).get('metrics', {}).get('sbc_metrics', {}),
            'behavioral_insights': {
                'frustration_level': game_context.get('game_status', {}).get('metrics', {}).get('sbc_metrics', {}).get('emotional_context', {}).get('frustration_level', 0),
                'momentum_state': game_context.get('game_status', {}).get('metrics', {}).get('sbc_metrics', {}).get('emotional_context', {}).get('momentum_state', 'neutral'),
                'learning_trend': game_context.get('game_status', {}).get('metrics', {}).get('sbc_metrics', {}).get('emotional_context', {}).get('learning_trend', 'stable')
            },
            'confidence': max([tip['priority'] for tip in coaching_tips]) if coaching_tips else 0.0
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'coaching_tips': [f'Error generating tips: {str(e)}'],
            'experiments': [],
            'insights': {'message': 'Error in SBC Backend'},
            'round': session.get('round', 0),
            'current_strategy': 'error',
            'llm_type': 'error',
            'enhanced_analysis': {},
            'behavioral_insights': {},
            'confidence': 0.0
        })


@app.route('/game_data', methods=['GET'])
def get_game_data():
    """Get current game session data for frontend analysis"""
    try:
        return jsonify({
            'human_moves': session.get('human_moves', []),
            'robot_moves': session.get('robot_moves', []),
            'results': session.get('results', []),
            'round': session.get('round', 0),
            'human_wins': session.get('human_wins', 0),
            'robot_wins': session.get('robot_wins', 0),
            'ties': session.get('ties', 0)
        })
    except Exception as e:
        return jsonify({
            'error': f'Failed to get game data: {str(e)}',
            'human_moves': [],
            'robot_moves': [],
            'results': [],
            'round': 0,
            'human_wins': 0,
            'robot_wins': 0,
            'ties': 0
        })


@app.route('/sbc/banter', methods=['POST'])
def sbc_banter():
    """Generate personality-aware banter using SBC Backend"""
    if not SBC_BACKEND_AVAILABLE:
        return jsonify({'error': 'SBC Backend not available'}), 503
    
    try:
        # Build current game context
        game_context = build_game_context(session)
        
        # Get SBC backend instance
        sbc = get_sbc_backend()
        
        # Generate banter
        seed_data = sbc.rule_selector.select_banter_seed(game_context)
        prompt = sbc._create_banter_prompt(seed_data, game_context)
        personality = seed_data['styling_hints'].get('personality', 'neutral')
        
        banter_text = sbc.model_adapter.generate_response(prompt, personality)
        
        return jsonify({
            'banter': banter_text,
            'personality': personality,
            'context': seed_data['context_reason'],
            'banter_key': seed_data['banter_key'],
            'source': {
                'engine': sbc.model_adapter.engine.value,
                'personality': personality,
                'analysis': 'comprehensive'
            },
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({'error': f'Error generating banter: {str(e)}'}), 500


@app.route('/sbc/coach', methods=['POST'])
def sbc_coach():
    """Generate coaching tips using SBC Backend (API endpoint)"""
    if not SBC_BACKEND_AVAILABLE:
        return jsonify({'error': 'SBC Backend not available'}), 503
    
    try:
        # Build current game context
        game_context = build_game_context(session)
        
        # Get tip count from request
        data = request.get_json() or {}
        tip_count = data.get('tip_count', 3)
        
        # Get SBC backend instance
        sbc = get_sbc_backend()
        
        # Generate coaching tips
        seed_data_list = sbc.rule_selector.select_coaching_seeds(game_context, tip_count)
        
        tips = []
        for seed_data in seed_data_list:
            prompt = sbc._create_coaching_prompt(seed_data, game_context)
            tip_text = sbc.model_adapter.generate_response(prompt, 'professor')
            
            tips.append({
                'tip': tip_text,
                'reason': seed_data['reason_key'],
                'priority': seed_data['priority'],
                'specific_advice': seed_data.get('specific_advice', ''),
                'tip_key': seed_data['tip_key']
            })
        
        return jsonify({
            'tips': tips,
            'total_tips': len(tips),
            'source': {
                'engine': sbc.model_adapter.engine.value,
                'personality': 'professor',
                'analysis_depth': 'comprehensive'
            },
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({'error': f'Error generating coaching: {str(e)}'}), 500


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
    game_state['difficulty'] = 'challenger'  # Default to challenger difficulty
    game_state['ai_difficulty'] = 'challenger'
    game_state['multiplayer'] = False
    game_state['change_points'] = []
    game_state['current_strategy'] = 'unknown'
    game_state['human_strategy_label'] = 'unknown'
    
    # Reset change detector
    change_detector.reset()
    global enhanced_predictor
    enhanced_predictor = _create_enhanced_predictor()
    
    # Reset the adaptive AI system
    if RPS_AI_SYSTEM_AVAILABLE:
        try:
            reset_ai_system()
        except Exception as e:
            print(f"Error resetting AI system: {e}")
    
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
    global enhanced_predictor
    enhanced_predictor = _create_enhanced_predictor()

    return jsonify({'message': 'Game reset successfully'})

@app.route('/developer', methods=['GET'])
def developer_console():
    """Developer console interface with comprehensive debugging and monitoring."""
    if not DEVELOPER_CONSOLE_AVAILABLE:
        return jsonify({'error': 'Developer console not available'}), 503
    
    try:
        report = get_developer_report()
        chart = get_chart()
        
        return render_template('developer_console.html', 
                             report=report, 
                             chart=chart,
                             session_active=True)
    except:
        return jsonify({'error': 'Developer console functions not available'}), 503

@app.route('/developer/api/report', methods=['GET'])
def developer_api_report():
    """API endpoint for developer report data."""
    if not DEVELOPER_CONSOLE_AVAILABLE:
        return jsonify({'error': 'Developer console not available'}), 503
    
    try:
        return jsonify(get_developer_report())
    except:
        return jsonify({'error': 'Developer report functions not available'}), 503

@app.route('/developer/api/chart', methods=['GET'])
def developer_api_chart():
    """API endpoint for model comparison chart."""
    if not DEVELOPER_CONSOLE_AVAILABLE:
        return jsonify({'error': 'Developer console not available'}), 503
    
    try:
        chart_data = get_chart()
        return jsonify({'chart': chart_data})
    except:
        return jsonify({'error': 'Chart functions not available'}), 503

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
    
    try:
        report = get_performance_report()
        return render_template('performance_dashboard.html', report=report)
    except:
        return jsonify({'error': 'Performance optimizer functions not available'}), 503

@app.route('/performance/api/report', methods=['GET'])
def performance_api_report():
    """API endpoint for performance report."""
    if not PERFORMANCE_OPTIMIZER_AVAILABLE:
        return jsonify({'error': 'Performance optimizer not available'}), 503
    
    try:
        return jsonify(get_performance_report())
    except:
        return jsonify({'error': 'Performance report functions not available'}), 503

@app.route('/performance/api/bundle-analysis', methods=['GET'])
def performance_api_bundle():
    """API endpoint for bundle size analysis."""
    if not PERFORMANCE_OPTIMIZER_AVAILABLE:
        return jsonify({'error': 'Performance optimizer not available'}), 503
    
    try:
        return jsonify(optimizer.bundle_analyzer.analyze_file_sizes())
    except:
        return jsonify({'error': 'Bundle analysis functions not available'}), 503

@app.route('/performance/api/timing/<model_name>', methods=['GET'])
def performance_api_timing(model_name):
    """API endpoint for model timing analysis."""
    if not PERFORMANCE_OPTIMIZER_AVAILABLE:
        return jsonify({'error': 'Performance optimizer not available'}), 503
    
    try:
        return jsonify(optimizer.timing_validator.get_timing_analysis(model_name))
    except:
        return jsonify({'error': 'Timing analysis functions not available'}), 503


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
