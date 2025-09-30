from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy import EnhancedStrategy, FrequencyStrategy, MarkovStrategy

app = Flask(__name__)

# Initialize strategy instances
enhanced_strategy = EnhancedStrategy(order=2, recency_weight=0.8)
frequency_strategy = FrequencyStrategy()
markov_strategy = MarkovStrategy()

game_state = {
    'human_history': [],
    'robot_history': [],
    'result_history': [],
    'round_history': [],  # Each entry: {'round': n, 'human': move, 'robot': move}
    'round': 0,
    'stats': {'human_win': 0, 'robot_win': 0, 'tie': 0},
    'difficulty': 'markov',
    'multiplayer': False,
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
    # Serve all game state for single-page UI
    return render_template('index.html', moves=MOVES, hotkeys=HOTKEYS,
        stats=game_state['stats'],
        human_history=game_state['human_history'],
        robot_history=game_state['robot_history'],
        result_history=game_state['result_history'],
        round=game_state['round'],
        default_difficulty=game_state['difficulty'])

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
        'confidence': confidence
    })

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

@app.route('/reset', methods=['POST', 'GET'])
def reset():
    game_state['human_history'].clear()
    game_state['robot_history'].clear()
    game_state['result_history'].clear()
    game_state['round'] = 0
    game_state['stats'] = {'human_win': 0, 'robot_win': 0, 'tie': 0}
    game_state['difficulty'] = 'markov'
    game_state['multiplayer'] = False
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
