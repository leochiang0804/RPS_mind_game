# StatsManager: Tracks game statistics and prediction accuracy
from collections import Counter

class StatsManager:
    def __init__(self):
        self.reset()

    def reset(self):
        self.results = Counter({'win': 0, 'loss': 0, 'tie': 0})
        self.total_rounds = 0
        self.prediction_correct = 0
        self.human_moves = []
        self.robot_moves = []
        self.predicted_moves = []

    def record_round(self, human_move, robot_move, predicted_move, result):
        self.human_moves.append(human_move)
        self.robot_moves.append(robot_move)
        self.predicted_moves.append(predicted_move)
        self.results[result] += 1
        self.total_rounds += 1
        if robot_move == predicted_move:
            self.prediction_correct += 1

    def get_stats(self):
        accuracy = (self.prediction_correct / self.total_rounds) * 100 if self.total_rounds else 0
        move_dist = Counter(self.human_moves)
        return {
            'win': self.results['win'],
            'loss': self.results['loss'],
            'tie': self.results['tie'],
            'rounds': self.total_rounds,
            'accuracy': accuracy,
            'move_distribution': dict(move_dist)
        }

    def print_stats(self):
        stats = self.get_stats()
        print("\nGame Statistics:")
        print(f"Rounds played: {stats['rounds']}")
        print(f"Wins: {stats['win']}, Losses: {stats['loss']}, Ties: {stats['tie']}")
        print(f"Robot prediction accuracy: {stats['accuracy']:.2f}%")
        print("Human move distribution:")
        for move, count in stats['move_distribution'].items():
            print(f"  {move}: {count}")
