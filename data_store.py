# Data storage for human move history
import os

class DataStore:
    def __init__(self, filename='history.txt'):
        self.filename = filename
        self.history = self._load_history()

    def save_move(self, move):
        self.history.append(move)
        with open(self.filename, 'a') as f:
            f.write(move + '\n')

    def get_history(self):
        return self.history

    def _load_history(self):
        if not os.path.exists(self.filename):
            return []
        with open(self.filename, 'r') as f:
            return [line.strip() for line in f if line.strip() in ['paper', 'scissor', 'stone']]
