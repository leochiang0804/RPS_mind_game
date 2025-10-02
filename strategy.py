# Strategy interface and implementations for difficulty levels
import random
from collections import defaultdict, deque
from ml_model_enhanced import EnhancedMLModel
from move_mapping import MOVE_TO_NUMBER, NUMBER_TO_MOVE, normalize_move, get_counter_move

class Strategy:
    def predict(self, history):
        raise NotImplementedError()

class DecisionTreeStrategy(Strategy):
    def __init__(self):
        from sklearn.tree import DecisionTreeClassifier
        self.clf = DecisionTreeClassifier()
        self.is_trained = False

    def train(self, history):
        # Use previous move as feature, next move as label
        if len(history) < 2:
            self.is_trained = False
            return
        X = []
        y = []
        for prev, nxt in zip(history[:-1], history[1:]):
            prev_normalized = normalize_move(prev)
            nxt_normalized = normalize_move(nxt)
            X.append([MOVE_TO_NUMBER[prev_normalized]])
            y.append(MOVE_TO_NUMBER[nxt_normalized])
        self.clf.fit(X, y)
        self.is_trained = True

    def predict(self, history):
        import random
        if not self.is_trained or not history:
            return random.choice(['rock', 'paper', 'scissors'])
        last_move = normalize_move(history[-1])
        last_move_number = MOVE_TO_NUMBER[last_move]
        pred_number = self.clf.predict([[last_move_number]])[0]
        predicted_move = NUMBER_TO_MOVE[pred_number]
        # Return counter move
        return get_counter_move(predicted_move)

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

class HybridStrategy(Strategy):
    def __init__(self):
        self.markov = MarkovStrategy()
    def train(self, history):
        self.markov.train(history)
    def predict(self, history):
        # Use Markov if enough history, else frequency
        if len(history) < 5:
            return FrequencyStrategy().predict(history)
        return self.markov.predict(history)

class EnhancedStrategy(Strategy):
    """Enhanced strategy using improved ML model with recency weighting"""
    def __init__(self, order=2, recency_weight=0.8):
        self.model = EnhancedMLModel(order=order, recency_weight=recency_weight)
        self.confidence = 0.5
        
    def train(self, history):
        self.model.train(history)
        
    def predict(self, history):
        robot_move, confidence = self.model.predict(history)
        self.confidence = confidence
        return robot_move
        
    def get_confidence(self):
        """Get confidence level for last prediction"""
        return self.confidence
        
    def get_stats(self):
        """Get detailed model statistics"""
        return self.model.get_stats()
