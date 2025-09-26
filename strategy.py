# Strategy interface and implementations for difficulty levels
import random
from ml_model import MLModel

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
        most_common = max(freq, key=freq.get)
        counter = {'paper': 'scissor', 'scissor': 'stone', 'stone': 'paper'}
        return counter[most_common]

class MarkovStrategy(Strategy):
    def __init__(self):
        self.model = MLModel()
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
