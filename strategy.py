# Implementations for difficulty levels
import random
from collections import defaultdict, deque
from ml_model_enhanced import EnhancedMLModel
from move_mapping import MOVE_TO_NUMBER, NUMBER_TO_MOVE, normalize_move, get_counter_move

class Strategy:
    def predict(self, history):
        raise NotImplementedError()

class RandomStrategy(Strategy):
    def predict(self, history):
        return random.choice(['rock', 'paper', 'scissors'])

class FrequencyStrategy(Strategy):
    def predict(self, history):
        if not history:
            return random.choice(['rock', 'paper', 'scissors'])
        # Normalize history to standard format
        normalized_history = [normalize_move(move) for move in history]
        freq = {move: normalized_history.count(move) for move in ['rock', 'paper', 'scissors']}
        most_common = max(freq.keys(), key=lambda k: freq[k])
        return get_counter_move(most_common)

class MarkovStrategy(Strategy):
    def __init__(self):
        self.model = EnhancedMLModel()
    def train(self, history):
        self.model.train(history)
    def predict(self, history):
        return self.model.predict(history)



