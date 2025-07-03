"""Utilities for parsing LLM responses."""

import re
from typing import Optional, Tuple


def extract_tictactoe_move(response: str) -> Optional[Tuple[int, int]]:
    """
    Extract a Tic-Tac-Toe move from an LLM response.
    
    Args:
        response: The LLM's response string.
        
    Returns:
        Tuple of (row, col) if found, None otherwise.
    """
    # Look for answer tags
    answer_pattern = r'<answer>\s*(\d+)\s*,\s*(\d+)\s*</answer>'
    match = re.search(answer_pattern, response)
    
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        return (row, col)
    
    return None


def extract_connect_four_move(response: str) -> Optional[int]:
    """
    Extract a Connect Four move from an LLM response.
    
    Args:
        response: The LLM's response string.
        
    Returns:
        Column number if found, None otherwise.
    """
    # Look for answer tags
    answer_pattern = r'<answer>\s*(\d+)\s*</answer>'
    match = re.search(answer_pattern, response)
    
    if match:
        col = int(match.group(1))
        return col
    
    return None


def check_response_format(response: str) -> bool:
    """
    Check if the response has the correct format with think and answer tags.
    
    Args:
        response: The LLM's response string.
        
    Returns:
        True if format is correct, False otherwise.
    """
    # Check for exactly one of each tag
    think_open_count = response.count("<think>")
    think_close_count = response.count("</think>")
    answer_open_count = response.count("<answer>")
    answer_close_count = response.count("</answer>")
    
    return (think_open_count == 1 and 
            think_close_count == 1 and 
            answer_open_count == 1 and 
            answer_close_count == 1)


def validate_response_structure(response: str) -> bool:
    """
    Validate that the response has proper structure.
    
    Args:
        response: The LLM's response string.
        
    Returns:
        True if structure is valid, False otherwise.
    """
    # Pattern to match: <think>...</think>...<answer>...</answer>
    pattern = r"<think>([\s\S]*?)</think>([\s\S]*?)<answer>([\s\S]*?)</answer>"
    
    # Prepend <think> if it's missing (for GRPO responses)
    if not response.startswith("<think>"):
        response = "<think>" + response
        
    return bool(re.match(pattern, response))