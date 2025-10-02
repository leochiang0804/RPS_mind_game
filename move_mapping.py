# Centralized Move Mapping System
# This file defines the unified mapping between moves and numbers across all models

# UNIFIED MAPPING SYSTEM
# 0 = rock (beats scissors)
# 1 = paper (beats rock)  
# 2 = scissors (beats paper)

MOVE_TO_NUMBER = {
    'rock': 0,
    'paper': 1, 
    'scissors': 2
}

NUMBER_TO_MOVE = {
    0: 'rock',
    1: 'paper',
    2: 'scissors'
}

# Counter move mappings (what beats what)
COUNTER_MOVES = {
    'rock': 'paper',      # paper beats rock
    'paper': 'scissors',  # scissors beats paper
    'scissors': 'rock'    # rock beats scissors
}

REVERSE_COUNTER_MOVES = {
    'paper': 'rock',      # if opponent plays paper, they beat rock
    'scissors': 'paper',  # if opponent plays scissors, they beat paper  
    'rock': 'scissors'    # if opponent plays rock, they beat scissors
}

def normalize_move(move):
    """Convert any move name to standard format"""
    move_lower = move.lower()
    if move_lower in ['stone', 'r']:
        return 'rock'
    elif move_lower in ['scissor', 's']:
        return 'scissors'
    elif move_lower in ['p']:
        return 'paper'
    return move_lower

def move_to_number(move):
    """Convert move string to number using unified mapping"""
    normalized = normalize_move(move)
    return MOVE_TO_NUMBER.get(normalized, -1)

def number_to_move(number):
    """Convert number to move string using unified mapping"""
    return NUMBER_TO_MOVE.get(number, 'unknown')

def get_counter_move(move):
    """Get the move that beats the given move"""
    normalized = normalize_move(move)
    return COUNTER_MOVES.get(normalized, 'unknown')

def get_reverse_counter_move(move):
    """Get the move that the given move beats"""
    normalized = normalize_move(move)
    return REVERSE_COUNTER_MOVES.get(normalized, 'unknown')

def validate_move(move):
    """Check if a move is valid"""
    normalized = normalize_move(move)
    return normalized in ['rock', 'paper', 'scissors']

# Standard moves list
MOVES = ['rock', 'paper', 'scissors']