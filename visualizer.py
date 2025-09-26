# Visualizer: ASCII charts/tables for move history and Markov transitions
from collections import Counter

def print_move_distribution(human_moves):
    print("\nHuman Move Distribution:")
    dist = Counter(human_moves)
    for move in ['paper', 'scissor', 'stone']:
        count = dist.get(move, 0)
        print(f"  {move}: {'*' * count} ({count})")

def print_markov_transitions(transitions):
    print("\nMarkov Chain Transition Table:")
    header = "      " + "  ".join(['paper', 'scissor', 'stone'])
    print(header)
    for prev in ['paper', 'scissor', 'stone']:
        row = f"{prev:6}"
        for nxt in ['paper', 'scissor', 'stone']:
            row += f"  {transitions[prev][nxt]:2}"
        print(row)
