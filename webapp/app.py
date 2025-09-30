from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy import EnhancedStrategy, FrequencyStrategy, MarkovStrategy
from change_point_detector import ChangePointDetector
from coach_tips import CoachTipsGenerator
from tournament_system import TournamentSystem
from optimized_strategies import ToWinStrategy, NotToLoseStrategy
from personality_engine import get_personality_engine
from replay_system import GameReplay, get_replay_manager, get_replay_analyzer

app = Flask(__name__)

# Initialize strategy instances
enhanced_strategy = EnhancedStrategy(order=2, recency_weight=0.8)
frequency_strategy = FrequencyStrategy()
markov_strategy = MarkovStrategy()
to_win_strategy = ToWinStrategy()
not_to_lose_strategy = NotToLoseStrategy()

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
    'difficulty': 'enhanced',  # Default to enhanced mode
    'strategy_preference': 'balanced',  # Default strategy preference
    'personality': 'neutral',  # Default personality
    'multiplayer': False,
    'change_points': [],  # Store detected strategy changes
    'current_replay': None,  # Current game replay session
    'current_strategy': 'warming up',  # Current strategy label
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
            'to_win': [],
            'not_to_lose': []
        },
        'model_confidence_history': {
            'random': [],
            'frequency': [],
            'markov': [],
            'enhanced': [],
            'to_win': [],
            'not_to_lose': []
        }
    }

MOVES = ['paper', 'scissor', 'stone']
HOTKEYS = {'a': 'paper', 'w': 'scissor', 'd': 'stone'}

@app.route('/')
def index():
    # Serve all game state for single-page UI - using working template temporarily
    return render_template('index_working.html', moves=MOVES, hotkeys=HOTKEYS,
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
                enhanced_strategy.train(history)
                predicted_counter = enhanced_strategy.predict(history)
                confidence = enhanced_strategy.get_confidence()
                reverse_counter = {'scissor': 'paper', 'stone': 'scissor', 'paper': 'stone'}
                predicted = reverse_counter.get(predicted_counter, random.choice(MOVES))
            else:
                predicted = random.choice(MOVES)
                confidence = 0.33
            
            # Convert prediction to robot move
            counter = {'paper': 'scissor', 'scissor': 'stone', 'stone': 'paper'}
            base_move = counter.get(predicted) if predicted else random.choice(MOVES)
        
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
        
        if len(game_state['human_history']) >= 5:  # Need minimum moves for analysis
            change_points = change_detector.get_all_change_points()
            current_strategy = change_detector.get_current_strategy_label()
            game_state['change_points'] = change_points
            game_state['current_strategy'] = current_strategy
        
        if result == 'human':
            game_state['stats']['human_win'] += 1
        elif result == 'robot':
            game_state['stats']['robot_win'] += 1
        else:
            game_state['stats']['tie'] += 1
        
        # Add move to replay system
        if game_state['current_replay']:
            analysis_data = {
                'human_pattern': current_strategy if 'current_strategy' in locals() else 'unknown',
                'change_points_detected': len(change_points) if 'change_points' in locals() else 0,
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
        'strategy_preference': game_state.get('strategy_preference', 'balanced'),
        'personality': game_state.get('personality', 'neutral'),
        'multiplayer': game_state['multiplayer'],
        'accuracy': game_state['accuracy'],
        'confidence': confidence,
        'change_points': game_state.get('change_points', []),
        'current_strategy': game_state.get('current_strategy', 'unknown'),
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
        'accuracy': game_state['accuracy']
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
        'current_strategy': game_state.get('current_strategy', 'unknown')
    })

@app.route('/reset', methods=['POST', 'GET'])
def reset():
    game_state['human_history'].clear()
    game_state['robot_history'].clear()
    game_state['result_history'].clear()
    game_state['round'] = 0
    game_state['stats'] = {'human_win': 0, 'robot_win': 0, 'tie': 0}
    game_state['difficulty'] = 'enhanced'  # Default to enhanced
    game_state['multiplayer'] = False
    game_state['change_points'] = []
    game_state['current_strategy'] = 'unknown'
    
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


import threading
import webbrowser
import time

def open_browser():
    time.sleep(1)  # Wait a moment for the server to start
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    threading.Thread(target=open_browser).start()
    app.run(debug=True)
