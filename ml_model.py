# Markov Chain ML model for predicting human moves
import random

class MLModel:
    def __init__(self):
        # Transition table: {prev_move: {next_move: count}}
        self.transitions = {move: {m: 0 for m in ['paper', 'scissor', 'stone']} for move in ['paper', 'scissor', 'stone']}
        self.last_move = None

    def train(self, history):
        # Reset transitions
        self.transitions = {move: {m: 0 for m in ['paper', 'scissor', 'stone']} for move in ['paper', 'scissor', 'stone']}
        if len(history) < 2:
            self.last_move = history[-1] if history else None
            return
        for prev, nxt in zip(history[:-1], history[1:]):
            if prev in self.transitions and nxt in self.transitions[prev]:
                self.transitions[prev][nxt] += 1
        self.last_move = history[-1]

    def predict(self, history):
        counter = {'paper': 'scissor', 'scissor': 'stone', 'stone': 'paper'}
        # Use last human move to predict next
        last = history[-1] if history else self.last_move
        if last and last in self.transitions:
            next_probs = self.transitions[last]
            # Pick the most likely next move
            predicted = max(next_probs, key=next_probs.get)
            # If all counts are zero, fallback to random
            if next_probs[predicted] == 0:
                predicted = random.choice(['paper', 'scissor', 'stone'])
            return counter[predicted]
        else:
            return random.choice(['paper', 'scissor', 'stone'])
