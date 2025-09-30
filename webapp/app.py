from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy import EnhancedStrategy, FrequencyStrategy, MarkovStrategy
from change_point_detector import ChangePointDetector
from coach_tips import CoachTipsGenerator

app = Flask(__name__)

# Initialize strategy instances
enhanced_strategy = EnhancedStrategy(order=2, recency_weight=0.8)
frequency_strategy = FrequencyStrategy()
markov_strategy = MarkovStrategy()

# Initialize strategies and change detector with balanced settings for web gameplay
change_detector = ChangePointDetector(window_size=6, chi2_threshold=3.5, min_segment_length=4)

# Initialize coach tips generator
coach = CoachTipsGenerator()

game_state = {
    'human_history': [],
    'robot_history': [],
    'result_history': [],
    'round_history': [],  # Each entry: {'round': n, 'human': move, 'robot': move}
    'round': 0,
    'stats': {'human_win': 0, 'robot_win': 0, 'tie': 0},
    'difficulty': 'enhanced',  # Default to enhanced mode
    'multiplayer': False,
    'change_points': [],  # Store detected strategy changes
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
    multiplayer = data.get('multiplayer', game_state['multiplayer'])
    game_state['difficulty'] = difficulty
    game_state['multiplayer'] = multiplayer
    if move not in MOVES:
        return jsonify({'error': 'Invalid move'}), 400

    import random
    # Difficulty strategies
    def robot_strategy(history, difficulty):
        # Predict next human move, then counter it
        predicted = None
        confidence = 0.33
        
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
        elif difficulty == 'hybrid':
            if len(history) < 5:
                return robot_strategy(history, 'frequency')
            return robot_strategy(history, 'enhanced')
        elif difficulty == 'decision_tree':
            predicted = random.choice(MOVES)
            confidence = 0.33
        else:
            predicted = random.choice(MOVES)
            confidence = 0.33
            
        counter = {'paper': 'scissor', 'scissor': 'stone', 'stone': 'paper'}
        robot_move = counter[predicted]
        
        # Track prediction accuracy
        if len(history) > 0:
            actual_next = history[-1]
            game_state['total_predictions'][difficulty] += 1
            if predicted == actual_next:
                game_state['correct_predictions'][difficulty] += 1
            correct = game_state['correct_predictions'][difficulty]
            total = game_state['total_predictions'][difficulty]
            game_state['accuracy'][difficulty] = round((correct / total) * 100, 2) if total else None
            
        return robot_move, confidence

    results = []
    robot_move, confidence = robot_strategy(game_state['human_history'], difficulty)
    
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
        'multiplayer': game_state['multiplayer'],
        'accuracy': game_state['accuracy'],
        'confidence': confidence,
        'change_points': game_state.get('change_points', []),
        'current_strategy': game_state.get('current_strategy', 'unknown')
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

import threading
import webbrowser
import time

def open_browser():
    time.sleep(1)  # Wait a moment for the server to start
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(debug=True)
