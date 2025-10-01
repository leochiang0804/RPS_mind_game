from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy import EnhancedStrategy, FrequencyStrategy, MarkovStrategy
from change_point_detector import ChangePointDetector
from coach_tips import CoachTipsGenerator
from tournament_system import TournamentSystem
from optimized_strategies import ToWinStrategy, NotToLoseStrategy
from personality_engine import get_personality_engine
from replay_system import GameReplay, get_replay_manager, get_replay_analyzer

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

# Try to import AI Coach functionality
try:
    from ai_coach_metrics import get_metrics_aggregator
    from enhanced_coach import get_enhanced_coach
    from ai_coach_langchain import get_langchain_coach
    AI_COACH_AVAILABLE = True
    print("✅ AI Coach integration available")
except ImportError as e:
    AI_COACH_AVAILABLE = False
    print(f"⚠️ AI Coach not available: {e}")

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-for-ai-coach-sessions-2024'

# Initialize strategy instances
enhanced_strategy = EnhancedStrategy(order=2, recency_weight=0.8)
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

# Initialize coach tips generator
coach = CoachTipsGenerator()

# Initialize tournament system
tournament_system = TournamentSystem('webapp/tournament_data.json')

# Initialize replay system
replay_manager = get_replay_manager()
replay_analyzer = get_replay_analyzer()

game_state = {
    'human_history': [],
    'robot_history': [],
    'result_history': [],
    'round_history': [],  # Each entry: {'round': n, 'human': move, 'robot': move}
    'round': 0,
    'stats': {'human_win': 0, 'robot_win': 0, 'tie': 0},
    'difficulty': 'enhanced',  # Default to enhanced mode (legacy)
    'ai_difficulty': 'enhanced',  # Explicit AI difficulty label
    'strategy_preference': 'balanced',  # Default strategy preference
    'personality': 'neutral',  # Default personality
    'multiplayer': False,
    'change_points': [],  # Store detected strategy changes
    'current_replay': None,  # Current game replay session
    'current_strategy': 'warming up',  # Human strategy label (legacy field)
    'human_strategy_label': 'warming up',  # Explicit human strategy label for analytics
    'accuracy': {
        'random': None,
        'frequency': None,
        'markov': None,
        'hybrid': None,
        'decision_tree': None,
        'enhanced': None
    },
    'correct_predictions': {
        'random': 0,
        'frequency': 0,
        'markov': 0,
        'hybrid': 0,
        'decision_tree': 0,
        'enhanced': 0
    },
    'total_predictions': {
        'random': 0,
        'frequency': 0,
        'markov': 0,
        'hybrid': 0,
        'decision_tree': 0,
        'enhanced': 0
    },
        'model_predictions_history': {
            'random': [],
            'frequency': [],
            'markov': [],
            'enhanced': [],
            'lstm': [],
            'to_win': [],
            'not_to_lose': []
        },
        'model_confidence_history': {
            'random': [],
            'frequency': [],
            'markov': [],
            'enhanced': [],
            'lstm': [],
            'to_win': [],
            'not_to_lose': []
        }
    }

MOVES = ['paper', 'scissor', 'stone']
HOTKEYS = {'a': 'paper', 'w': 'scissor', 'd': 'stone'}

@app.route('/')
def index():
    # Serve all game state for single-page UI - using main index.html template
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

@app.route('/play', methods=['POST'])
def play():
    data = request.get_json(force=True)
    move = data.get('move')
    difficulty = data.get('difficulty', game_state['difficulty'])
    strategy_preference = data.get('strategy', 'balanced')
    personality = data.get('personality', 'neutral')
    multiplayer = data.get('multiplayer', game_state['multiplayer'])
    
    game_state['difficulty'] = difficulty
    game_state['ai_difficulty'] = difficulty
    game_state['multiplayer'] = multiplayer
    game_state['strategy_preference'] = strategy_preference
    game_state['personality'] = personality
    
    # Initialize replay if starting new game
    if game_state['current_replay'] is None:
        game_state['current_replay'] = GameReplay()
        game_state['current_replay'].metadata.update({
            'difficulty': difficulty,
            'strategy': strategy_preference,
            'personality': personality
        })
    
    if move not in MOVES:
        return jsonify({'error': 'Invalid move'}), 400

    import random
    # Difficulty strategies
    def robot_strategy(history, difficulty, strategy_preference='balanced', personality='neutral'):
        # Performance tracking - start timing
        start_time = time.time()
        
        # Set the personality in the engine
        advanced_personalities = ['berserker', 'guardian', 'chameleon', 'professor', 'wildcard', 'mirror']
        
        if personality in advanced_personalities:
            personality_engine.set_personality(personality)
        
        # Predict next human move, then counter it
        predicted = None
        confidence = 0.33
        
        # Apply strategy preference to determine which approach to use
        if strategy_preference == 'to_win' and difficulty in ['enhanced', 'markov', 'frequency']:
            # Force the AI to use the aggressive "to win" approach
            predicted = to_win_strategy.predict(history)
            confidence = to_win_strategy.get_confidence()
            # Convert to robot counter move
            counter = {'paper': 'scissor', 'scissor': 'stone', 'stone': 'paper'}
            base_move = counter.get(predicted) if predicted else random.choice(MOVES)
            
        elif strategy_preference == 'not_to_lose' and difficulty in ['enhanced', 'markov', 'frequency']:
            # Force the AI to use the defensive "not to lose" approach
            predicted = not_to_lose_strategy.predict(history)
            confidence = not_to_lose_strategy.get_confidence()
            # Convert to robot counter move
            counter = {'paper': 'scissor', 'scissor': 'stone', 'stone': 'paper'}
            base_move = counter.get(predicted) if predicted else random.choice(MOVES)
            
        else:
            # Regular difficulty-based strategy
            if difficulty == 'random':
                predicted = random.choice(MOVES)
                confidence = 0.33
            elif difficulty == 'frequency':
                predicted_counter = frequency_strategy.predict(history)
                # Convert back to predicted human move
                reverse_counter = {'scissor': 'paper', 'stone': 'scissor', 'paper': 'stone'}
                predicted = reverse_counter.get(predicted_counter, random.choice(MOVES))
                confidence = 0.5
            elif difficulty == 'markov':
                markov_strategy.train(history)
                predicted_counter = markov_strategy.predict(history)
                reverse_counter = {'scissor': 'paper', 'stone': 'scissor', 'paper': 'stone'}
                predicted = reverse_counter.get(predicted_counter, random.choice(MOVES))
                confidence = 0.6
            elif difficulty == 'enhanced':
                if PERFORMANCE_OPTIMIZER_AVAILABLE:
                    # Use performance-timed inference
                    def enhanced_inference():
                        enhanced_strategy.train(history)
                        return enhanced_strategy.predict(history)
                    predicted_counter, inference_time = time_model_inference('enhanced', enhanced_inference)
                else:
                    enhanced_strategy.train(history)
                    predicted_counter = enhanced_strategy.predict(history)
                    
                confidence = enhanced_strategy.get_confidence()
                reverse_counter = {'scissor': 'paper', 'stone': 'scissor', 'paper': 'stone'}
                predicted = reverse_counter.get(predicted_counter, random.choice(MOVES))
            elif difficulty == 'lstm' and LSTM_AVAILABLE and lstm_predictor:
                # LSTM prediction
                try:
                    base_move = lstm_predictor.get_counter_move(history)
                    confidence = lstm_predictor.get_confidence(history)
                    
                    # Track LSTM inference time
                    inference_duration = time.time() - start_time
                    if DEVELOPER_CONSOLE_AVAILABLE:
                        track_inference('lstm', inference_duration)
                    
                    return base_move, confidence
                except Exception as e:
                    print(f"LSTM prediction error: {e}")
                    predicted = random.choice(MOVES)
                    confidence = 0.33
            else:
                predicted = random.choice(MOVES)
                confidence = 0.33
            
            # Convert prediction to robot move
            counter = {'paper': 'scissor', 'scissor': 'stone', 'stone': 'paper'}
            base_move = counter.get(predicted) if predicted else random.choice(MOVES)
        
        # Track inference time for non-LSTM models
        inference_duration = time.time() - start_time
        if DEVELOPER_CONSOLE_AVAILABLE:
            track_inference(difficulty, inference_duration)
        
        # Apply personality modifications using the advanced engine
        if personality in advanced_personalities:
            # Create game history in the format expected by personality engine
            game_history = []
            if len(game_state['human_history']) == len(game_state['robot_history']):
                game_history = list(zip(game_state['human_history'], game_state['robot_history']))
            
            final_move = personality_engine.apply_personality_to_move(
                base_move or random.choice(MOVES), confidence, history, game_history
            )
            return final_move, confidence
        
        # Legacy personality modifiers for backwards compatibility
        elif personality == 'aggressive':
            confidence = min(1.0, confidence * 1.2)  # More confident
            return base_move, confidence
        elif personality == 'defensive':
            confidence = max(0.1, confidence * 0.8)  # Less confident
            return base_move, confidence
        elif personality == 'chaotic':
            if random.random() < 0.3:  # 30% chance to be completely random
                return random.choice(MOVES), 0.33
            return base_move, confidence
        elif personality == 'adaptive':
            # Simple adaptive logic for legacy support
            if len(game_state['result_history']) >= 5:
                recent_results = game_state['result_history'][-5:]
                robot_wins = recent_results.count('robot')
                if robot_wins <= 1:  # Losing, be more aggressive
                    return base_move, min(1.0, confidence * 1.3)
            return base_move, confidence
        
        return base_move, confidence

    results = []
    robot_move, confidence = robot_strategy(
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
        freq_pred_counter = frequency_strategy.predict(history)
        reverse_counter = {'scissor': 'paper', 'stone': 'scissor', 'paper': 'stone'}
        freq_pred = reverse_counter.get(freq_pred_counter, random.choice(MOVES))
        game_state['model_predictions_history']['frequency'].append(freq_pred)
        
        # Markov prediction
        markov_strategy.train(history)
        markov_pred_counter = markov_strategy.predict(history)
        markov_pred = reverse_counter.get(markov_pred_counter, random.choice(MOVES))
        game_state['model_predictions_history']['markov'].append(markov_pred)
        
        # Enhanced prediction
        enhanced_strategy.train(history)
        enhanced_pred_counter = enhanced_strategy.predict(history)
        enhanced_pred = reverse_counter.get(enhanced_pred_counter, random.choice(MOVES))
        game_state['model_predictions_history']['enhanced'].append(enhanced_pred)
        
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
        
        # To Win prediction
        to_win_pred = to_win_strategy.predict(history)
        game_state['model_predictions_history']['to_win'].append(to_win_pred)
        
        # Not to Lose prediction
        not_to_lose_pred = not_to_lose_strategy.predict(history)
        game_state['model_predictions_history']['not_to_lose'].append(not_to_lose_pred)
        
        # Track confidence values for each model
        game_state['model_confidence_history']['random'].append(0.33)
        game_state['model_confidence_history']['frequency'].append(min(0.8, 0.3 + len(history) * 0.02))
        game_state['model_confidence_history']['markov'].append(min(0.85, 0.4 + len(history) * 0.015))
        game_state['model_confidence_history']['enhanced'].append(enhanced_strategy.get_confidence())
        
        # LSTM confidence
        if LSTM_AVAILABLE and lstm_predictor:
            try:
                lstm_confidence = lstm_predictor.get_confidence(history)
                game_state['model_confidence_history']['lstm'].append(lstm_confidence)
            except:
                game_state['model_confidence_history']['lstm'].append(0.33)
        else:
            game_state['model_confidence_history']['lstm'].append(0.33)
        
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
        
        # Add move to replay system
        if game_state['current_replay']:
            analysis_data = {
                'human_pattern': current_strategy,
                'change_points_detected': len(change_points),
                'robot_strategy': f"{difficulty}_{strategy_preference}_{personality}"
            }
            
            game_state['current_replay'].add_move(
                round_number=game_state['round'],
                human_move=move,
                robot_move=robot_move or 'paper',
                result=result,
                confidence=confidence,
                strategy_used=f"{difficulty}_{strategy_preference}",
                analysis=analysis_data
            )
        
        results = [result]
    
    # Calculate model accuracy after each move (only if we have predictions to compare)
    if len(game_state['human_history']) > 1:  # Need at least 2 moves to have a prediction
        current_human_move = game_state['human_history'][-1]  # Just played move
        
        # Calculate accuracy for each model
        for model_name in game_state['model_predictions_history']:
            predictions = game_state['model_predictions_history'][model_name]
            if len(predictions) > 0:
                # Compare predictions to actual human moves (excluding first move since no prediction exists)
                correct_predictions = 0
                total_predictions = min(len(predictions), len(game_state['human_history']) - 1)
                
                for i in range(total_predictions):
                    if predictions[i] == game_state['human_history'][i + 1]:  # i+1 because predictions start from move 2
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
    session['difficulty'] = game_state['difficulty']
    session['ai_difficulty'] = game_state.get('ai_difficulty', game_state['difficulty'])
    session['current_strategy'] = game_state.get('current_strategy', 'unknown')
    session['human_strategy_label'] = game_state.get('human_strategy_label', game_state.get('current_strategy', 'unknown'))
    session['round_count'] = game_state['round']
    session['strategy_preference'] = game_state.get('strategy_preference', 'balanced')
    session['personality'] = game_state.get('personality', 'neutral')
    session['multiplayer'] = game_state.get('multiplayer', False)
    session['accuracy'] = game_state['accuracy']
    session['confidence'] = confidence
    session['change_points'] = game_state.get('change_points', [])
    session['model_predictions_history'] = game_state['model_predictions_history']
    session['model_confidence_history'] = game_state['model_confidence_history']
    session['correct_predictions'] = game_state['correct_predictions']
    session['total_predictions'] = game_state['total_predictions']
    
    # Return updated state for AJAX
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
        'strategy_preference': game_state.get('strategy_preference', 'balanced'),
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
        'total_predictions': game_state['total_predictions']
    })

@app.route('/coaching', methods=['GET'])
def get_coaching_tips():
    """Get intelligent coaching tips based on current game state"""
    # Generate coaching tips
    tips_data = coach.generate_tips(
        human_history=game_state['human_history'],
        robot_history=game_state['robot_history'],
        result_history=game_state['result_history'],
        change_points=game_state.get('change_points', []),
        current_strategy=game_state.get('current_strategy', 'unknown')
    )
    
    return jsonify({
        'coaching_tips': tips_data['tips'],
        'experiments': tips_data['experiments'],
        'insights': tips_data['insights'],
        'round': game_state['round'],
        'current_strategy': game_state.get('current_strategy', 'unknown')
    })

@app.route('/analytics/export', methods=['GET'])
def export_analytics():
    """Export analytics data in JSON format"""
    import math
    from datetime import datetime
    
    try:
        format_type = request.args.get('format', 'json')
        
        # Calculate analytics data
        history = game_state['human_history']
        total_moves = len(history)
        
        if total_moves == 0:
            analytics_data = {
                'message': 'No data available for export',
                'total_games': 0
            }
        else:
            paper_count = history.count('paper')
            stone_count = history.count('stone')
            scissor_count = history.count('scissor')
            
            # Calculate predictability (highest percentage)
            max_count = max(paper_count, stone_count, scissor_count)
            predictability = (max_count / total_moves) * 100
            
            # Calculate win rate
            win_rate = (game_state['stats']['human_win'] / game_state['round'] * 100) if game_state['round'] > 0 else 0
            
            # Calculate entropy (randomness)
            p1, p2, p3 = paper_count/total_moves, stone_count/total_moves, scissor_count/total_moves
            entropy = 0
            for p in [p1, p2, p3]:
                if p > 0:
                    entropy -= p * math.log2(p)
            randomness = (entropy / math.log2(3)) * 100 if entropy > 0 else 0
            
            analytics_data = {
                'timestamp': datetime.now().isoformat(),
                'total_games': game_state['round'],
                'total_moves': total_moves,
                'win_rate': round(win_rate, 2),
                'predictability_score': round(predictability, 2),
                'randomness_level': round(randomness, 2),
                'move_distribution': {
                    'paper': {'count': paper_count, 'percentage': round((paper_count/total_moves)*100, 2)},
                    'stone': {'count': stone_count, 'percentage': round((stone_count/total_moves)*100, 2)},
                    'scissor': {'count': scissor_count, 'percentage': round((scissor_count/total_moves)*100, 2)}
                },
                'stats': game_state['stats'],
                'strategy_changes': len(game_state.get('change_points', [])),
                'recent_history': history[-20:],  # Last 20 moves
                'game_metadata': {
                    'difficulty': game_state['difficulty'],
                    'multiplayer': game_state['multiplayer']
                }
            }
        
        if format_type == 'json':
            response = jsonify(analytics_data)
            response.headers['Content-Disposition'] = f'attachment; filename=rps_analytics_{datetime.now().strftime("%Y%m%d")}.json'
            return response
        else:
            return jsonify({'error': 'Unsupported format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    # Return stats as JSON for AJAX
    return jsonify({
        'stats': game_state['stats'],
        'human_history': game_state['human_history'],
        'robot_history': game_state['robot_history'],
        'result_history': game_state['result_history'],
        'round': game_state['round'],
        'accuracy': game_state['accuracy'],
        'ai_difficulty': game_state.get('ai_difficulty', game_state.get('difficulty', 'enhanced')),
        'human_strategy_label': game_state.get('human_strategy_label', game_state.get('current_strategy', 'unknown'))
    })

@app.route('/history', methods=['GET'])
def history():
    # Return full game history including round details
    return jsonify({
        'stats': game_state['stats'],
        'human_history': game_state['human_history'],
        'robot_history': game_state['robot_history'],
        'result_history': game_state['result_history'],
        'round_history': game_state.get('round_history', []),
        'round': game_state['round'],
        'accuracy': game_state['accuracy'],
        'change_points': game_state.get('change_points', []),
        'current_strategy': game_state.get('current_strategy', 'unknown'),
        'human_strategy_label': game_state.get('human_strategy_label', game_state.get('current_strategy', 'unknown')),
        'ai_difficulty': game_state.get('ai_difficulty', game_state.get('difficulty', 'enhanced'))
    })

@app.route('/reset', methods=['POST', 'GET'])
def reset():
    game_state['human_history'].clear()
    game_state['robot_history'].clear()
    game_state['result_history'].clear()
    game_state['round'] = 0
    game_state['stats'] = {'human_win': 0, 'robot_win': 0, 'tie': 0}
    game_state['difficulty'] = 'enhanced'  # Default to enhanced
    game_state['ai_difficulty'] = 'enhanced'
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
    elif (robot_move == 'paper' and human_move == 'stone') or \
         (robot_move == 'stone' and human_move == 'scissor') or \
         (robot_move == 'scissor' and human_move == 'paper'):
        return 'robot'
    else:
        return 'human'

# Tournament Routes
@app.route('/tournament', methods=['GET'])
def tournament_dashboard():
    """Get tournament dashboard data"""
    try:
        leaderboard = tournament_system.get_leaderboard(10)
        recent_matches = []
        
        # Get recent completed matches
        for match in tournament_system.matches.values():
            if match.status == 'completed':
                recent_matches.append(match.to_dict())
        
        recent_matches.sort(key=lambda m: m['completed_at'] or '', reverse=True)
        
        return jsonify({
            'leaderboard': leaderboard,
            'recent_matches': recent_matches[:10],
            'total_players': len(tournament_system.players),
            'total_matches': len([m for m in tournament_system.matches.values() if m.status == 'completed'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tournament/player', methods=['POST'])
def create_player():
    """Create a new tournament player"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        
        if not name:
            return jsonify({'error': 'Player name is required'}), 400
        
        # Check if player already exists
        existing_player = tournament_system.get_player_by_name(name)
        if existing_player:
            return jsonify({'error': 'Player name already exists'}), 400
        
        player = tournament_system.create_player(name)
        return jsonify({'success': True, 'player': player.to_dict()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

# ========================================
# REPLAY SYSTEM ENDPOINTS
# ========================================

@app.route('/replay/save', methods=['POST'])
def save_current_replay():
    """Save the current game replay"""
    try:
        if game_state['current_replay'] and len(game_state['current_replay'].moves) > 0:
            filepath = replay_manager.save_replay(game_state['current_replay'])
            return jsonify({
                'success': True,
                'session_id': game_state['current_replay'].session_id,
                'filepath': filepath,
                'total_rounds': len(game_state['current_replay'].moves)
            })
        else:
            return jsonify({'success': False, 'error': 'No replay data to save'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/replay/list')
def list_replays():
    """List all saved replays"""
    try:
        replays = replay_manager.list_replays()
        return jsonify({'replays': replays})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/replay/<session_id>')
def get_replay(session_id):
    """Get a specific replay by session ID"""
    try:
        replay = replay_manager.load_replay(session_id)
        if replay:
            return jsonify(replay.to_dict())
        else:
            return jsonify({'error': 'Replay not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/replay/<session_id>/analyze')
def analyze_replay(session_id):
    """Analyze a specific replay"""
    try:
        replay = replay_manager.load_replay(session_id)
        if replay:
            analysis = replay_analyzer.analyze_replay(replay)
            return jsonify(analysis)
        else:
            return jsonify({'error': 'Replay not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/replay/<session_id>/export')
def export_replay(session_id):
    """Export replay to CSV"""
    try:
        csv_data = replay_manager.export_replay_csv(session_id)
        if csv_data:
            from flask import Response
            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=replay_{session_id}.csv'}
            )
        else:
            return jsonify({'error': 'Replay not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/replay/<session_id>/annotate', methods=['POST'])
def annotate_replay(session_id):
    """Add annotation to a replay round"""
    try:
        data = request.get_json()
        round_number = data.get('round')
        annotation = data.get('annotation')
        
        replay = replay_manager.load_replay(session_id)
        if replay:
            replay.add_annotation(round_number, annotation)
            replay_manager.save_replay(replay)
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Replay not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/replay/dashboard')
def replay_dashboard():
    """Replay dashboard page"""
    try:
        replays = replay_manager.list_replays()
        return render_template('replay_dashboard.html', replays=replays)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/replay/viewer/<session_id>')
def replay_viewer(session_id):
    """Replay viewer page"""
    try:
        replay = replay_manager.load_replay(session_id)
        if replay:
            analysis = replay_analyzer.analyze_replay(replay)
            return render_template('replay_viewer.html', replay=replay.to_dict(), analysis=analysis)
        else:
            return jsonify({'error': 'Replay not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Reset game now also resets replay
@app.route('/reset', methods=['POST'])
def reset_game():
    # Save current replay if it has moves
    if game_state['current_replay'] and len(game_state['current_replay'].moves) > 0:
        try:
            replay_manager.save_replay(game_state['current_replay'])
        except Exception as e:
            print(f"Error saving replay: {e}")
    
    # Reset game state
    game_state['human_history'] = []
    game_state['robot_history'] = []
    game_state['result_history'] = []
    game_state['round_history'] = []
    game_state['round'] = 0
    game_state['stats'] = {'human_win': 0, 'robot_win': 0, 'tie': 0}
    game_state['change_points'] = []
    game_state['current_replay'] = None  # Reset replay
    
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


# AI Coach API Endpoints
@app.route('/ai_coach/status')
def ai_coach_status():
    """Get AI coach availability status"""
    return jsonify({
        'available': AI_COACH_AVAILABLE,
        'features': {
            'metrics_aggregation': AI_COACH_AVAILABLE,
            'enhanced_coaching': AI_COACH_AVAILABLE,
            'langchain_integration': AI_COACH_AVAILABLE
        },
        'status': 'ready' if AI_COACH_AVAILABLE else 'disabled'
    })


@app.route('/ai_coach/realtime', methods=['POST'])
def ai_coach_realtime():
    """Get real-time AI coaching advice with enhanced features"""
    if not AI_COACH_AVAILABLE:
        return jsonify({'error': 'AI Coach not available'}), 503
    
    try:
        import time
        start_time = time.time()
        
        # Get comprehensive metrics from current game state
        metrics_aggregator = get_metrics_aggregator()
        enhanced_coach = get_enhanced_coach()
        
        # Extract game data and options from request
        data = request.get_json() or {}
        
        # Extract toggleable options from request (with defaults for backward compatibility)
        llm_type = data.get('llm_type', 'mock')
        coaching_style = data.get('coaching_style', enhanced_coach.get_coaching_style())
        include_metrics = data.get('include_metrics', False)
        include_metadata = data.get('include_metadata', False)
        
        # Set coaching style if different
        current_style = enhanced_coach.get_coaching_style()
        if coaching_style != current_style:
            enhanced_coach.set_coaching_style(coaching_style)
        
        # Set LLM type if different (NEW: Actually use the llm_type parameter!)
        current_llm_type = enhanced_coach.get_llm_type()
        if llm_type != current_llm_type:
            llm_switch_result = enhanced_coach.set_llm_type(llm_type)
            if not llm_switch_result.get('success', False):
                print(f"⚠️ Failed to switch to {llm_type} LLM: {llm_switch_result.get('error', 'Unknown error')}")
                # Continue with current LLM type
                llm_type = current_llm_type
        
        # Always prioritize session data if available (real game state)
        if 'human_moves' in session and 'robot_moves' in session:
            human_strategy_label = session.get('human_strategy_label', session.get('current_strategy', 'unknown'))
            ai_difficulty = session.get('ai_difficulty', session.get('difficulty', 'medium'))
            game_data = {
                'human_moves': session['human_moves'],
                'robot_moves': session['robot_moves'],
                'results': session.get('results', []),
                'round': len(session.get('human_moves', [])),
                'current_strategy': human_strategy_label,
                'human_strategy_label': human_strategy_label,
                'strategy_preference': session.get('strategy_preference', 'balanced'),
                'personality': session.get('personality', 'neutral'),
                'accuracy': session.get('accuracy', {}),
                'model_predictions_history': session.get('model_predictions_history', {}),
                'model_confidence_history': session.get('model_confidence_history', {}),
                'correct_predictions': session.get('correct_predictions', {}),
                'total_predictions': session.get('total_predictions', {}),
                'change_points': session.get('change_points', []),
                'confidence': session.get('confidence', 0.5),
                'multiplayer': session.get('multiplayer', False),
                'ai_difficulty': ai_difficulty,
                'current_difficulty': ai_difficulty,
                'difficulty': ai_difficulty
            }
        elif data.get('human_moves'):
            # Use provided game data
            game_data = data
        else:
            # No session data and no meaningful request data - use empty state
            game_data = {
                'human_moves': [],
                'robot_moves': [],
                'results': [],
                'round': 0,
                'current_difficulty': 'medium',
                'ai_difficulty': 'medium',
                'difficulty': 'medium',
                'current_strategy': 'unknown',
                'human_strategy_label': 'unknown',
                'strategy_preference': 'balanced',
                'personality': 'neutral',
                'multiplayer': False
            }
        
        # Get comprehensive metrics
        comprehensive_metrics = metrics_aggregator.aggregate_comprehensive_metrics(game_data)
        
        # Generate real-time advice
        advice = enhanced_coach.generate_coaching_advice(
            game_state=game_data,
            coaching_type='real_time'
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build response
        response = {
            'success': True,
            'advice': advice,
            'raw_response': advice.get('raw_response', ''),
            'natural_language_full': advice.get('natural_language_full', ''),
            'coaching_personality': f"{coaching_style.title()} Coach",
            'metrics_summary': {
                'current_round': comprehensive_metrics.get('core_game', {}).get('current_round', 0),
                'win_rate': comprehensive_metrics.get('core_game', {}).get('win_rates', {}).get('human', 0.0),
                'pattern_type': comprehensive_metrics.get('patterns', {}).get('pattern_type', 'unknown'),
                'ai_mode': enhanced_coach.mode
            },
            'llm_type': llm_type,
            'coaching_style': coaching_style,
            'ai_difficulty': game_data.get('ai_difficulty', game_state.get('ai_difficulty', 'unknown')),
            'human_strategy_label': game_data.get('human_strategy_label', game_state.get('human_strategy_label', 'unknown'))
        }
        
        # Add enhanced metrics if requested
        if include_metrics and game_data['round'] > 0:
            response['enhanced_metrics'] = {
                'pattern_entropy': comprehensive_metrics.get('patterns', {}).get('entropy', 0.0),
                'predictability': comprehensive_metrics.get('patterns', {}).get('predictability', 0.0),
                'recent_trend': comprehensive_metrics.get('performance', {}).get('recent_performance', {}).get('trend', 'stable'),
                'complexity_score': comprehensive_metrics.get('patterns', {}).get('complexity_score', 0.5),
                'adaptation_rate': comprehensive_metrics.get('ai_behavior', {}).get('adaptation_metrics', {}).get('response_time', 1.0)
            }
        
        # Add metadata if requested
        if include_metadata:
            response['metadata'] = {
                'processing_time_ms': round(processing_time * 1000, 2),
                'data_points': game_data['round'],
                'llm_type': llm_type,
                'coaching_style': coaching_style,
                'response_type': 'realtime',
                'timestamp': int(time.time()),
                'has_session_data': 'human_moves' in session and bool(session.get('human_moves'))
            }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"⚠️ AI Coach real-time error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'llm_type': data.get('llm_type', 'mock') if 'data' in locals() else 'mock',
            'fallback_advice': {
                'tips': ['Focus on varying your patterns', 'Observe the AI responses', 'Try unexpected moves'],
                'insights': {'error': 'Failed to generate full analysis'},
                'educational_content': {'focus': 'basic_strategy'},
                'confidence_level': 0.3
            }
        }), 500


@app.route('/ai_coach/comprehensive', methods=['POST'])
def ai_coach_comprehensive():
    """Get comprehensive AI coaching analysis"""
    if not AI_COACH_AVAILABLE:
        return jsonify({'error': 'AI Coach not available'}), 503
    
    try:
        # Get comprehensive metrics from current game state
        metrics_aggregator = get_metrics_aggregator()
        enhanced_coach = get_enhanced_coach()
        
        # Extract game data from request or use current session
        data = request.get_json() or {}
        
        # Extract toggleable options from request (with defaults for backward compatibility)
        llm_type = data.get('llm_type', 'mock')
        coaching_style = data.get('coaching_style', enhanced_coach.get_coaching_style())
        
        # Set coaching style if different
        current_style = enhanced_coach.get_coaching_style()
        if coaching_style != current_style:
            enhanced_coach.set_coaching_style(coaching_style)
        
        # Set LLM type if different (NEW: Actually use the llm_type parameter!)
        current_llm_type = enhanced_coach.get_llm_type()
        if llm_type != current_llm_type:
            llm_switch_result = enhanced_coach.set_llm_type(llm_type)
            if not llm_switch_result.get('success', False):
                print(f"⚠️ Failed to switch to {llm_type} LLM: {llm_switch_result.get('error', 'Unknown error')}")
                # Continue with current LLM type
                llm_type = current_llm_type
        
        # Always prioritize session data if available (real game state)
        if 'human_moves' in session and 'robot_moves' in session:
            human_strategy_label = session.get('human_strategy_label', session.get('current_strategy', 'unknown'))
            ai_difficulty = session.get('ai_difficulty', session.get('difficulty', 'medium'))
            game_data = {
                'human_moves': session['human_moves'],
                'robot_moves': session['robot_moves'],
                'results': session.get('results', []),
                'round': len(session.get('human_moves', [])),
                'current_strategy': human_strategy_label,
                'human_strategy_label': human_strategy_label,
                'ai_difficulty': ai_difficulty,
                'current_difficulty': ai_difficulty,
                'difficulty': ai_difficulty,
                # FIX: Add the missing AI behavior session data
                'accuracy': session.get('accuracy', {}),
                'model_predictions_history': session.get('model_predictions_history', {}),
                'model_confidence_history': session.get('model_confidence_history', {}),
                'correct_predictions': session.get('correct_predictions', {}),
                'total_predictions': session.get('total_predictions', {}),
                'change_points': session.get('change_points', []),
                'strategy_preference': session.get('strategy_preference', 'balanced'),
                'personality': session.get('personality', 'neutral'),
                'confidence': session.get('confidence', 0.5),
                'multiplayer': session.get('multiplayer', False)
            }
        elif data.get('human_moves'):
            # Use provided game data
            game_data = data
        else:
            # No session data and no meaningful request data - use empty state
            game_data = {
                'human_moves': [],
                'robot_moves': [],
                'results': [],
                'round': 0,
                'current_difficulty': 'medium',
                'ai_difficulty': 'medium',
                'difficulty': 'medium',
                'current_strategy': 'unknown',
                'human_strategy_label': 'unknown',
                'strategy_preference': 'balanced',
                'personality': 'neutral',
                'multiplayer': False
            }
        
        # Get comprehensive metrics
        comprehensive_metrics = metrics_aggregator.aggregate_comprehensive_metrics(game_data)
        
        # Generate comprehensive analysis
        analysis = enhanced_coach.generate_coaching_advice(
            game_state=game_data,
            coaching_type='comprehensive'
        )
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'raw_response': analysis.get('raw_response', ''),
            'natural_language_full': analysis.get('natural_language_full', ''),
            'coaching_personality': f"{coaching_style.title()} Coach",
            'metrics_summary': comprehensive_metrics,  # Include full metrics for detailed display
            'session_summary': {
                'total_rounds': comprehensive_metrics.get('core_game', {}).get('current_round', 0),
                'final_win_rate': comprehensive_metrics.get('core_game', {}).get('win_rates', {}).get('human', 0.0),
                'dominant_pattern': comprehensive_metrics.get('patterns', {}).get('pattern_type', 'unknown'),
                'performance_trend': comprehensive_metrics.get('performance', {}).get('recent_performance', {}).get('trend', 'stable'),
                'ai_mode': enhanced_coach.mode,
                'ai_difficulty': game_data.get('ai_difficulty', game_state.get('ai_difficulty', 'unknown')),
                'human_strategy_label': game_data.get('human_strategy_label', game_state.get('human_strategy_label', 'unknown'))
            },
            'llm_type': llm_type,
            'coaching_style': coaching_style,
            'ai_difficulty': game_data.get('ai_difficulty', game_state.get('ai_difficulty', 'unknown')),
            'human_strategy_label': game_data.get('human_strategy_label', game_state.get('human_strategy_label', 'unknown'))
        })
        
    except Exception as e:
        print(f"⚠️ AI Coach comprehensive error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'fallback_analysis': {
                'psychological_patterns': 'Analysis failed - using fallback insights',
                'strategic_evolution': 'Unable to generate comprehensive analysis',
                'learning_recommendations': 'Focus on pattern variation and strategic thinking',
                'educational_summary': 'Session data collected but analysis failed'
            }
        }), 500


@app.route('/ai_coach/metrics', methods=['GET'])
def ai_coach_metrics():
    """Get current comprehensive metrics for debugging/analysis"""
    if not AI_COACH_AVAILABLE:
        return jsonify({'error': 'AI Coach not available'}), 503
    
    try:
        metrics_aggregator = get_metrics_aggregator()
        
        # Get current game state - prioritize session data
        game_data = {}
        if 'human_moves' in session and 'robot_moves' in session:
            human_strategy_label = session.get('human_strategy_label', session.get('current_strategy', 'unknown'))
            ai_difficulty = session.get('ai_difficulty', session.get('difficulty', 'medium'))
            game_data = {
                'human_moves': session['human_moves'],
                'robot_moves': session['robot_moves'],
                'results': session.get('results', []),
                'round': len(session.get('human_moves', [])),
                'current_strategy': human_strategy_label,
                'human_strategy_label': human_strategy_label,
                'strategy_preference': session.get('strategy_preference', 'balanced'),
                'personality': session.get('personality', 'neutral'),
                'accuracy': session.get('accuracy', {}),
                'model_predictions_history': session.get('model_predictions_history', {}),
                'model_confidence_history': session.get('model_confidence_history', {}),
                'correct_predictions': session.get('correct_predictions', {}),
                'total_predictions': session.get('total_predictions', {}),
                'change_points': session.get('change_points', []),
                'confidence': session.get('confidence', 0.5),
                'multiplayer': session.get('multiplayer', False),
                'ai_difficulty': ai_difficulty,
                'current_difficulty': ai_difficulty,
                'difficulty': ai_difficulty
            }
        else:
            # No session data available - return empty state info
            game_data = {
                'human_moves': [],
                'robot_moves': [],
                'results': [],
                'round': 0,
                'current_difficulty': 'medium',
                'ai_difficulty': 'medium',
                'difficulty': 'medium',
                'current_strategy': 'unknown',
                'human_strategy_label': 'unknown',
                'strategy_preference': 'balanced',
                'personality': 'neutral',
                'multiplayer': False
            }
        
        # Get comprehensive metrics
        metrics = metrics_aggregator.aggregate_comprehensive_metrics(game_data)
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'data_sources': {
                'game_rounds': len(game_data.get('human_moves', [])),
                'has_results': bool(game_data.get('results')),
                'difficulty': game_data.get('current_difficulty', 'unknown')
            }
        })
        
    except Exception as e:
        print(f"⚠️ AI Coach metrics error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/ai_coach_demo')
def ai_coach_demo():
    """AI Coach demonstration page"""
    return render_template('ai_coach_demo.html')


@app.route('/ai_coach/toggle_mode', methods=['POST'])
def ai_coach_toggle_mode():
    """Toggle AI coach between AI and basic modes"""
    if not AI_COACH_AVAILABLE:
        return jsonify({'error': 'AI Coach not available'}), 503
    
    try:
        enhanced_coach = get_enhanced_coach()
        data = request.get_json() or {}
        
        new_mode = data.get('mode', 'toggle')
        
        if new_mode == 'toggle':
            new_mode = 'basic' if enhanced_coach.mode == 'ai' else 'ai'
        
        if new_mode in ['ai', 'basic']:
            enhanced_coach.mode = new_mode
            
            return jsonify({
                'success': True,
                'new_mode': new_mode,
                'message': f'AI Coach switched to {new_mode} mode'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Invalid mode: {new_mode}. Use "ai" or "basic"'
            }), 400
            
    except Exception as e:
        print(f"⚠️ AI Coach mode toggle error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/ai_coach/set_style', methods=['POST'])
def ai_coach_set_style():
    """Set AI coach coaching style"""
    if not AI_COACH_AVAILABLE:
        return jsonify({'error': 'AI Coach not available'}), 503
    
    try:
        enhanced_coach = get_enhanced_coach()
        data = request.get_json() or {}
        
        style = data.get('style', 'easy')
        
        result = enhanced_coach.set_coaching_style(style)
        
        if result['success']:
            return jsonify({
                'success': True,
                'style': result['new_style'],
                'description': result['description'],
                'message': f'Coaching style set to {result["new_style"]}'
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
            
    except Exception as e:
        print(f"⚠️ AI Coach style setting error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to set coaching style'
        }), 500


@app.route('/ai_coach/get_style', methods=['GET'])
def ai_coach_get_style():
    """Get current coaching style and available options"""
    if not AI_COACH_AVAILABLE:
        return jsonify({'error': 'AI Coach not available'}), 503
    
    try:
        enhanced_coach = get_enhanced_coach()
        
        return jsonify({
            'success': True,
            'current_style': enhanced_coach.get_coaching_style(),
            'available_styles': enhanced_coach.get_style_description(),
            'style_details': {
                'easy': {
                    'name': 'Easy to Understand',
                    'description': 'Simple, friendly tips that anyone can understand',
                    'example': 'Try to be more unpredictable - mix up your moves!'
                },
                'scientific': {
                    'name': 'Scientific & Detailed',
                    'description': 'Detailed analytics with entropy, Nash equilibrium, and complexity metrics',
                    'example': 'Your entropy is 0.8743 - increase randomness to optimize unpredictability'
                }
            }
        })
            
    except Exception as e:
        print(f"⚠️ AI Coach style retrieval error: {e}")


@app.route('/ai_coach/enhanced_analysis', methods=['POST'])
def ai_coach_enhanced_analysis():
    """Enhanced AI Coach analysis with all toggleable features"""
    if not AI_COACH_AVAILABLE:
        return jsonify({'error': 'AI Coach not available'}), 503
    
    try:
        import time
        start_time = time.time()
        
        enhanced_coach = get_enhanced_coach()
        metrics_aggregator = get_metrics_aggregator()
        data = request.get_json() or {}
        
        # Extract toggleable options from request
        llm_type = data.get('llm_type', 'mock')
        coaching_style = data.get('coaching_style', 'easy')
        include_comprehensive_metrics = data.get('include_comprehensive_metrics', False)
        include_response_metadata = data.get('include_response_metadata', True)
        include_advanced_analytics = data.get('include_advanced_analytics', False)
        processing_mode = data.get('processing_mode', 'realtime')
        analysis_type = data.get('analysis_type', 'realtime')  # realtime or comprehensive
        
        # Set coaching style
        enhanced_coach.set_coaching_style(coaching_style)
        
        # Get current game state
        game_data = {}
        if 'human_moves' in session and 'robot_moves' in session:
            game_data = {
                'human_moves': session['human_moves'],
                'robot_moves': session['robot_moves'],
                'results': session.get('results', []),
                'current_difficulty': session.get('difficulty', 'medium'),
                'round': len(session.get('human_moves', [])),
                'current_strategy': session.get('current_strategy', 'unknown')
            }
        else:
            # Generate demo data for UI testing
            import random
            demo_moves = ['rock', 'paper', 'scissors']
            demo_length = random.randint(8, 15)
            game_data = {
                'human_moves': [random.choice(demo_moves) for _ in range(demo_length)],
                'robot_moves': [random.choice(demo_moves) for _ in range(demo_length)],
                'results': [random.choice(['win', 'lose', 'tie']) for _ in range(demo_length)],
                'current_difficulty': 'medium',
                'round': demo_length,
                'current_strategy': 'adaptive'
            }
        
        # Get comprehensive metrics if requested
        comprehensive_metrics = None
        if include_comprehensive_metrics:
            comprehensive_metrics = metrics_aggregator.aggregate_comprehensive_metrics(game_data)
        
        # Generate analysis based on type
        if analysis_type == 'comprehensive':
            analysis = enhanced_coach.generate_coaching_advice(
                game_state=game_data,
                coaching_type='comprehensive'
            )
        else:
            analysis = enhanced_coach.generate_coaching_advice(
                game_state=game_data,
                coaching_type='realtime'
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build response
        response = {
            'success': True,
            'analysis': analysis,
            'analysis_type': analysis_type,
            'llm_type': llm_type,
            'coaching_style': coaching_style
        }
        
        # Add comprehensive metrics if requested
        if include_comprehensive_metrics and comprehensive_metrics:
            response['comprehensive_metrics'] = comprehensive_metrics
            response['metrics_count'] = sum(len(v) if isinstance(v, dict) else 1 for v in comprehensive_metrics.values())
        
        # Add response metadata if requested
        if include_response_metadata:
            response['response_metadata'] = {
                'processing_time_ms': round(processing_time * 1000, 2),
                'llm_type': llm_type,
                'coaching_style': coaching_style,
                'analysis_type': analysis_type,
                'data_points_analyzed': len(game_data.get('human_moves', [])),
                'confidence_level': round(random.uniform(0.75, 0.95), 3) if llm_type == 'mock' else 'N/A',
                'metrics_included': include_comprehensive_metrics,
                'advanced_analytics': include_advanced_analytics,
                'processing_mode': processing_mode,
                'timestamp': int(time.time())
            }
        
        # Add advanced analytics if requested
        if include_advanced_analytics and comprehensive_metrics:
            advanced_analytics = {
                'entropy_analysis': comprehensive_metrics.get('patterns', {}).get('entropy', 0.0),
                'nash_equilibrium': comprehensive_metrics.get('strategic', {}).get('nash_distance', 0.33),
                'psychological_indicators': {
                    'stress_level': comprehensive_metrics.get('psychological', {}).get('stress_indicators', {}).get('pattern_deviation', 0.0),
                    'confidence_trend': comprehensive_metrics.get('psychological', {}).get('confidence_indicators', {}).get('decision_speed', 'stable'),
                    'adaptation_rate': comprehensive_metrics.get('ai_behavior', {}).get('adaptation_metrics', {}).get('response_time', 1.0)
                },
                'pattern_complexity': comprehensive_metrics.get('patterns', {}).get('complexity_score', 0.5),
                'strategic_evolution': comprehensive_metrics.get('strategic', {}).get('evolution_trajectory', 'stable')
            }
            response['advanced_analytics'] = advanced_analytics
        
        return jsonify(response)
        
    except Exception as e:
        print(f"⚠️ Enhanced AI Coach analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'llm_type': data.get('llm_type', 'mock'),
            'error_details': 'Failed to generate enhanced analysis'
        }), 500


@app.route('/ai_coach/llm_comparison', methods=['POST'])
def ai_coach_llm_comparison():
    """Compare responses between Mock LLM and Real LLM"""
    if not AI_COACH_AVAILABLE:
        return jsonify({'error': 'AI Coach not available'}), 503
    
    try:
        import time
        
        enhanced_coach = get_enhanced_coach()
        data = request.get_json() or {}
        
        coaching_style = data.get('coaching_style', 'easy')
        enhanced_coach.set_coaching_style(coaching_style)
        
        # Get current game state
        game_data = {}
        if 'human_moves' in session and 'robot_moves' in session:
            game_data = {
                'human_moves': session['human_moves'],
                'robot_moves': session['robot_moves'],
                'results': session.get('results', []),
                'current_difficulty': session.get('difficulty', 'medium'),
                'round': len(session.get('human_moves', [])),
                'current_strategy': session.get('current_strategy', 'unknown')
            }
        else:
            # Generate demo data
            import random
            demo_moves = ['rock', 'paper', 'scissors']
            demo_length = 10
            game_data = {
                'human_moves': [random.choice(demo_moves) for _ in range(demo_length)],
                'robot_moves': [random.choice(demo_moves) for _ in range(demo_length)],
                'results': [random.choice(['win', 'lose', 'tie']) for _ in range(demo_length)],
                'current_difficulty': 'medium',
                'round': demo_length,
                'current_strategy': 'adaptive'
            }
        
        # Generate Mock LLM response
        start_time = time.time()
        mock_analysis = enhanced_coach.generate_coaching_advice(
            game_state=game_data,
            coaching_type='realtime'
        )
        mock_processing_time = time.time() - start_time
        
        # Try to generate Real LLM response (fallback to mock if unavailable)
        start_time = time.time()
        try:
            # Placeholder for real LLM integration
            real_analysis = {
                'status': 'Real LLM not configured',
                'message': 'To enable real LLM comparison, configure Ollama or OpenAI integration',
                'recommendation': 'The Mock LLM provides comprehensive analysis for demonstration'
            }
            real_processing_time = 0.0
        except Exception:
            real_analysis = {
                'status': 'Real LLM unavailable',
                'message': 'Real LLM integration not available in current setup'
            }
            real_processing_time = 0.0
        
        return jsonify({
            'success': True,
            'comparison': {
                'mock_llm': {
                    'analysis': mock_analysis,
                    'processing_time_ms': round(mock_processing_time * 1000, 2),
                    'type': 'Mock LLM',
                    'status': 'Available',
                    'confidence': round(random.uniform(0.75, 0.95), 3)
                },
                'real_llm': {
                    'analysis': real_analysis,
                    'processing_time_ms': round(real_processing_time * 1000, 2),
                    'type': 'Real LLM (Ollama/OpenAI)',
                    'status': 'Not Configured',
                    'confidence': 'N/A'
                }
            },
            'coaching_style': coaching_style,
            'data_points': len(game_data.get('human_moves', []))
        })
        
    except Exception as e:
        print(f"⚠️ LLM comparison error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
        return jsonify({
            'success': False,
            'error': 'Failed to get coaching style information'
        }), 500


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
