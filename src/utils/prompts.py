"""Prompt templates for game training."""

from typing import Literal


def build_tictactoe_prompt(
    board_state_str: str, 
    player: Literal['X', 'O']
) -> str:
    """
    Build a prompt for Tic-Tac-Toe.
    
    Args:
        board_state_str: Formatted board state string.
        player: Current player ('X' or 'O').
        
    Returns:
        Formatted prompt string.
    """
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. You first think about the reasoning process "
        "in the mind and then provides the user with the answer.<|im_end|>\n"
        "<|im_start|>user\n"
        "You are playing Tic Tac Toe and are aiming to win in as few moves as possible.\n"
        f"You play as {player} on a 3 by 3 grid. '{player}' is you, "
        f"'{'O' if player == 'X' else 'X'}' is your opponent, and 'free' is an empty slot.\n"
        "Current board state:\n"
        f"{board_state_str}"
        "Return the final answer in <answer> </answer> tags. "
        "For example <answer>1, 2</answer>.\n<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>"
    )


def build_tictactoe_sft_prompt(
    board_state_str: str, 
    player: Literal['X', 'O']
) -> str:
    """
    Build a prompt for Tic-Tac-Toe supervised fine-tuning.
    
    Args:
        board_state_str: Formatted board state string.
        player: Current player ('X' or 'O').
        
    Returns:
        Formatted prompt string.
    """
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. You first thinks about the reasoning process "
        "in the mind and then provides the user with the answer.<|im_end|>\n"
        "<|im_start|>user\n"
        "You are playing Tic Tac Toe aiming to win in as few moves as possible.\n"
        f"You play as {player} on a 3 by 3 grid. '{player}' is you, "
        f"'{'O' if player == 'X' else 'X'}' is your opponent, and 'free' is an empty slot.\n"
        "Current board state:\n"
        f"{board_state_str}"
        "Show your work in <think> </think> tags, and return the final answer in "
        "<answer> </answer> tags. For example <think>I can win by...</think> "
        "<answer>1, 2</answer>.<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Let me solve this step by step.\n"
    )


def build_connect_four_prompt(
    board_state_str: str, 
    player: Literal['X', 'O']
) -> str:
    """
    Build a prompt for Connect Four.
    
    Args:
        board_state_str: Formatted board state string.
        player: Current player ('X' or 'O').
        
    Returns:
        Formatted prompt string.
    """
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. You first think about the reasoning process "
        "in the mind and then provides the user with the answer.<|im_end|>\n"
        "<|im_start|>user\n"
        "You are playing Connect Four and are aiming to win in as few moves as possible.\n"
        f"You play as {player} on a 6 by 7 grid. '{player}' is you, "
        f"'{'O' if player == 'X' else 'X'}' is your opponent, and 'free' is an empty slot.\n"
        "Current board state:\n"
        f"{board_state_str}"
        "Return the column number (1-7) where you want to place your piece in "
        "<answer> </answer> tags. For example <answer>4</answer>.\n<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>"
    )