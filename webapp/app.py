from flask import Flask, render_template, request, redirect, url_for, jsonify
import os

app = Flask(__name__)

game_state = {
    'human_history': [],
    'robot_history': [],
    'result_history': [],
    'round': 0,
    'stats': {'human_win': 0, 'robot_win': 0, 'tie': 0},
    'difficulty': 'random',
    'multiplayer': False,
    'accuracy': {
        'random': None,
        'frequency': None,
        'markov': None,
        'hybrid': None,
        'decision_tree': None
    },
    'correct_predictions': {
        'random': 0,
        'frequency': 0,
        'markov': 0,
        'hybrid': 0,
        'decision_tree': 0
    },
    'total_predictions': {
        'random': 0,
        'frequency': 0,
        'markov': 0,
        'hybrid': 0,
        'decision_tree': 0
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
        round=game_state['round'])

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
        if difficulty == 'random':
            predicted = random.choice(MOVES)
        elif difficulty == 'frequency':
            if not history:
                predicted = random.choice(MOVES)
            else:
                freq = {m: history.count(m) for m in MOVES}
                most_common = max(freq, key=lambda k: freq[k])
                predicted = most_common
        elif difficulty == 'markov':
            if len(history) < 2:
                predicted = random.choice(MOVES)
            else:
                last = history[-1]
                next_moves = [history[i+1] for i in range(len(history)-1) if history[i] == last]
                if not next_moves:
                    predicted = random.choice(MOVES)
                else:
                    predicted = max(set(next_moves), key=next_moves.count)
        elif difficulty == 'hybrid':
            if len(history) < 5:
                return robot_strategy(history, 'frequency')
            return robot_strategy(history, 'markov')
        elif difficulty == 'decision_tree':
            predicted = random.choice(MOVES)
        else:
            predicted = random.choice(MOVES)
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
        return robot_move

    results = []
    robot_move = robot_strategy(game_state['human_history'], difficulty)
    if multiplayer:
        # Expect two moves from frontend (for now, use same move twice for demo)
        move2 = data.get('move2', move)
        result1 = get_result(robot_move, move)
        result2 = get_result(robot_move, move2)
        game_state['human_history'].append(move)
        game_state['human_history'].append(move2)
        game_state['robot_history'].append(robot_move)
        game_state['robot_history'].append(robot_move)
        game_state['result_history'].append(result1)
        game_state['result_history'].append(result2)
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
        'round': game_state['round'],
        'robot_move': robot_move,
        'result': results,
        'difficulty': game_state['difficulty'],
        'multiplayer': game_state['multiplayer'],
        'accuracy': game_state['accuracy']
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
    game_state['difficulty'] = 'random'
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

if __name__ == '__main__':
    app.run(debug=True)
