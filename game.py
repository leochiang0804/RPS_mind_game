# Game logic for Paper-Scissor-Stone ML game
from data_store import DataStore
from stats_manager import StatsManager
from strategy import RandomStrategy, FrequencyStrategy, MarkovStrategy, HybridStrategy

class Game:
    def __init__(self):
        self.data_store = DataStore()
        self.stats = StatsManager()
        self.round = 0
        self.human_history = []
        self.human2_history = []
        self.choices = ['paper', 'scissor', 'stone']
        self.strategy = None
        self.multiplayer = None

    def run(self):
        print("Select difficulty level:")
        print("1. Random\n2. Frequency\n3. Markov Chain\n4. Hybrid\n5. Decision Tree (Advanced)")
        self.strategy = self._select_strategy()
        self.multiplayer = self._ask_multiplayer()
        if self.multiplayer:
            print("Multiplayer mode: Two humans vs robot.")
        from visualizer import print_move_distribution, print_markov_transitions
        while True:
            if self.multiplayer:
                robot_move = self.strategy.predict(self.human_history + self.human2_history)
                predicted_move = robot_move
                human_move = input("Player 1, choose paper, scissor, stone or 'end' to quit, 'reset' to clear session, 'stats' for charts: ").strip().lower()
                if human_move == 'end':
                    print("Game ended. Thanks for playing!")
                    self.stats.print_stats()
                    print_move_distribution(self.stats.human_moves)
                    if hasattr(self.strategy, 'model') and hasattr(self.strategy.model, 'transitions'):
                        print_markov_transitions(self.strategy.model.transitions)
                    break
                if human_move == 'reset':
                    self._reset_session()
                    print("Session reset. History and stats cleared.")
                    continue
                if human_move == 'stats':
                    self.stats.print_stats()
                    print_move_distribution(self.stats.human_moves)
                    if hasattr(self.strategy, 'model') and hasattr(self.strategy.model, 'transitions'):
                        print_markov_transitions(self.strategy.model.transitions)
                    continue
                if human_move not in self.choices:
                    print("Invalid choice. Try again.")
                    continue
                self.human_history.append(human_move)
                self.data_store.save_move(human_move)
                human2_move = input("Player 2, choose paper, scissor, stone: ").strip().lower()
                if human2_move not in self.choices:
                    print("Invalid choice for Player 2. Try again.")
                    continue
                self.human2_history.append(human2_move)
                self.data_store.save_move(human2_move)
                self.round += 1
                # For stats, treat both moves as separate rounds
                result1 = self._get_result(robot_move, human_move)
                result2 = self._get_result(robot_move, human2_move)
                self.stats.record_round(human_move, robot_move, predicted_move, result1)
                self.stats.record_round(human2_move, robot_move, predicted_move, result2)
                self._show_result(robot_move, human_move)
                self._show_result(robot_move, human2_move)
                if isinstance(self.strategy, (MarkovStrategy, HybridStrategy)):
                    self.strategy.train(self.data_store.get_history())
                if hasattr(self.strategy, 'train') and self.strategy.__class__.__name__ == 'DecisionTreeStrategy':
                    self.strategy.train(self.data_store.get_history())
                if self.round % 5 == 0:
                    print(f"Retraining ML model with {self.round} rounds of human history...")
            else:
                robot_move = self.strategy.predict(self.human_history)
                predicted_move = robot_move
                human_move = input("Choose paper, scissor, stone or 'end' to quit, 'reset' to clear session, 'stats' for charts: ").strip().lower()
                if human_move == 'end':
                    print("Game ended. Thanks for playing!")
                    self.stats.print_stats()
                    print_move_distribution(self.stats.human_moves)
                    if hasattr(self.strategy, 'model') and hasattr(self.strategy.model, 'transitions'):
                        print_markov_transitions(self.strategy.model.transitions)
                    break
                if human_move == 'reset':
                    self._reset_session()
                    print("Session reset. History and stats cleared.")
                    continue
                if human_move == 'stats':
                    self.stats.print_stats()
                    print_move_distribution(self.stats.human_moves)
                    if hasattr(self.strategy, 'model') and hasattr(self.strategy.model, 'transitions'):
                        print_markov_transitions(self.strategy.model.transitions)
                    continue
                if human_move not in self.choices:
                    print("Invalid choice. Try again.")
                    continue
                self.human_history.append(human_move)
                self.data_store.save_move(human_move)
                self.round += 1
                result = self._get_result(robot_move, human_move)
                self.stats.record_round(human_move, robot_move, predicted_move, result)
                self._show_result(robot_move, human_move)
                if isinstance(self.strategy, (MarkovStrategy, HybridStrategy)):
                    self.strategy.train(self.data_store.get_history())
                if hasattr(self.strategy, 'train') and self.strategy.__class__.__name__ == 'DecisionTreeStrategy':
                    self.strategy.train(self.data_store.get_history())
                if self.round % 5 == 0:
                    print(f"Retraining ML model with {self.round} rounds of human history...")

    def _ask_multiplayer(self):
        choice = input("Enable multiplayer mode? (y/n): ").strip().lower()
        return choice == 'y'

    def _reset_session(self):
        self.data_store.history = []
        try:
            import os
            if os.path.exists(self.data_store.filename):
                os.remove(self.data_store.filename)
        except Exception:
            pass
        self.stats.reset()
        self.human_history = []
        self.round = 0
    def _select_strategy(self):
        from strategy import DecisionTreeStrategy
        while True:
            choice = input("Enter difficulty (1-5): ").strip()
            if choice == '1':
                return RandomStrategy()
            elif choice == '2':
                return FrequencyStrategy()
            elif choice == '3':
                return MarkovStrategy()
            elif choice == '4':
                return HybridStrategy()
            elif choice == '5':
                return DecisionTreeStrategy()
            else:
                print("Invalid selection. Please enter 1, 2, 3, 4, or 5.")

    def _show_result(self, robot_move, human_move):
        print(f"Robot chose: {robot_move}")
        result = self._get_result(robot_move, human_move)
        if result == 'tie':
            print("It's a tie!")
        elif result == 'loss':
            print("Robot wins!")
        else:
            print("You win!")

    def _get_result(self, robot_move, human_move):
        if robot_move == human_move:
            return 'tie'
        elif (robot_move == 'paper' and human_move == 'stone') or \
             (robot_move == 'stone' and human_move == 'scissor') or \
             (robot_move == 'scissor' and human_move == 'paper'):
            return 'loss'  # robot wins
        else:
            return 'win'  # human wins
