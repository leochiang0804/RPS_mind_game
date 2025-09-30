# Strategy interface and implementations for difficulty levels
import random
from collections import defaultdict, deque
from ml_model_enhanced import EnhancedMLModel

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
        move_map = {'paper': 0, 'scissor': 1, 'stone': 2}
        for prev, nxt in zip(history[:-1], history[1:]):
            X.append([move_map[prev]])
            y.append(move_map[nxt])
        self.clf.fit(X, y)
        self.is_trained = True

    def predict(self, history):
        import random
        move_map = {'paper': 0, 'scissor': 1, 'stone': 2}
        rev_map = {0: 'paper', 1: 'scissor', 2: 'stone'}
        if not self.is_trained or not history:
            return random.choice(['paper', 'scissor', 'stone'])
        last_move = move_map[history[-1]]
        pred = self.clf.predict([[last_move]])[0]
        # Counter move
        counter = {'paper': 'scissor', 'scissor': 'stone', 'stone': 'paper'}
        return counter[rev_map[pred]]

class RandomStrategy(Strategy):
    def predict(self, history):
        return random.choice(['paper', 'scissor', 'stone'])

class FrequencyStrategy(Strategy):
    def predict(self, history):
        if not history:
            return random.choice(['paper', 'scissor', 'stone'])
        freq = {move: history.count(move) for move in ['paper', 'scissor', 'stone']}
        most_common = max(freq.keys(), key=lambda k: freq[k])
        counter = {'paper': 'scissor', 'scissor': 'stone', 'stone': 'paper'}
        return counter[most_common]

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
