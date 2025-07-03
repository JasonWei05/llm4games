"""Reward functions for GRPO training."""

import copy
from typing import List, Dict, Any
import re

from ..games import TicTacToe, ConnectFour
from ..utils.parsing import (
    extract_tictactoe_move, 
    extract_connect_four_move,
    check_response_format,
    validate_response_structure
)
from ..utils.connect4_solver import Connect4Solver


class TicTacToeRewardCalculator:
    """Calculate rewards for Tic-Tac-Toe GRPO training."""
    
    def __init__(
        self, 
        minimax_weight: float = 0.3,
        monte_carlo_weight: float = 0.7,
        num_simulations: int = 5000,
        invalid_move_penalty: float = -2.0,
        format_bonus: float = 0.1
    ):
        """
        Initialize the reward calculator.
        
        Args:
            minimax_weight: Weight for minimax score in reward calculation.
            monte_carlo_weight: Weight for Monte Carlo win percentage.
            num_simulations: Number of simulations for Monte Carlo evaluation.
            invalid_move_penalty: Penalty for invalid moves or format.
            format_bonus: Bonus for correct response format.
        """
        self.minimax_weight = minimax_weight
        self.monte_carlo_weight = monte_carlo_weight
        self.num_simulations = num_simulations
        self.invalid_move_penalty = invalid_move_penalty
        self.format_bonus = format_bonus
    
    def calculate_correctness_rewards(
        self,
        prompts: List[str],
        completions: List[str],
        game_states: List[TicTacToe],
        **kwargs
    ) -> List[float]:
        """
        Calculate correctness rewards for Tic-Tac-Toe moves.
        
        Args:
            prompts: List of prompts (unused but required by interface).
            completions: List of model completions.
            game_states: List of game states.
            
        Returns:
            List of reward scores.
        """
        rewards = []
        
        for completion, game in zip(completions, game_states):
            # Check format
            if not check_response_format(completion):
                rewards.append(self.invalid_move_penalty)
                continue
            
            # Extract move
            move = extract_tictactoe_move(completion)
            if move is None:
                rewards.append(self.invalid_move_penalty)
                continue
            
            row, col = move[0] - 1, move[1] - 1  # Convert to 0-based
            
            # Check validity
            if not game.is_valid_move(row, col):
                rewards.append(self.invalid_move_penalty)
                continue
            
            # Calculate reward based on move quality
            # Current position evaluation
            current_win_pct = TicTacToe.calculate_win_percentage(
                copy.deepcopy(game), 
                self.num_simulations
            )
            current_minimax, _, _, _ = TicTacToe.minimax(
                copy.deepcopy(game), 
                game.player
            )
            
            # Position after move
            next_win_pct = TicTacToe.calculate_win_percentage_after_move(
                copy.deepcopy(game), 
                row, 
                col, 
                self.num_simulations
            )
            next_minimax, _, _, _ = TicTacToe.minimax_after_move(
                copy.deepcopy(game), 
                row, 
                col, 
                game.player
            )
            
            # Calculate advantages
            win_pct_advantage = next_win_pct - current_win_pct
            minimax_advantage = next_minimax - current_minimax
            
            # Combined reward
            reward = (
                self.minimax_weight * minimax_advantage + 
                self.monte_carlo_weight * win_pct_advantage
            )
            
            rewards.append(max(self.invalid_move_penalty, reward))
        
        return rewards
    
    def calculate_format_rewards(
        self,
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        Calculate format rewards for responses.
        
        Args:
            completions: List of model completions.
            
        Returns:
            List of format reward scores.
        """
        rewards = []
        
        for completion in completions:
            # Prepend <think> for GRPO completions
            if not completion.startswith("<think>"):
                completion = "<think>" + completion
                
            if validate_response_structure(completion):
                rewards.append(self.format_bonus)
            else:
                rewards.append(0.0)
                
        return rewards


class ConnectFourRewardCalculator:
    """Calculate rewards for Connect Four GRPO training."""
    
    def __init__(
        self,
        solver_path: str,
        opening_book_path: str,
        solver_weight: float = 0.5,
        monte_carlo_weight: float = 0.5,
        num_simulations: int = 2500,
        invalid_move_penalty: float = -2.0,
        format_bonus: float = 0.1
    ):
        """
        Initialize the reward calculator.
        
        Args:
            solver_path: Path to Connect4 solver executable.
            opening_book_path: Path to opening book CSV.
            solver_weight: Weight for solver score in reward calculation.
            monte_carlo_weight: Weight for Monte Carlo win percentage.
            num_simulations: Number of simulations for Monte Carlo evaluation.
            invalid_move_penalty: Penalty for invalid moves or format.
            format_bonus: Bonus for correct response format.
        """
        self.solver = Connect4Solver(solver_path, opening_book_path)
        self.solver_weight = solver_weight
        self.monte_carlo_weight = monte_carlo_weight
        self.num_simulations = num_simulations
        self.invalid_move_penalty = invalid_move_penalty
        self.format_bonus = format_bonus
    
    def calculate_correctness_rewards(
        self,
        prompts: List[str],
        completions: List[str],
        game_states: List[ConnectFour],
        position_strings: List[str],
        **kwargs
    ) -> List[float]:
        """
        Calculate correctness rewards for Connect Four moves.
        
        Args:
            prompts: List of prompts (unused but required by interface).
            completions: List of model completions.
            game_states: List of game states.
            position_strings: List of position strings for solver.
            
        Returns:
            List of reward scores.
        """
        rewards = []
        
        for completion, game, pos_str in zip(completions, game_states, position_strings):
            # Check format
            if not check_response_format(completion):
                rewards.append(self.invalid_move_penalty)
                continue
            
            # Extract move
            move = extract_connect_four_move(completion)
            if move is None:
                rewards.append(self.invalid_move_penalty)
                continue
            
            col = move - 1  # Convert to 0-based
            
            # Check validity
            if not game.is_valid_move(col):
                rewards.append(self.invalid_move_penalty)
                continue
            
            # Get solver evaluations
            try:
                all_scores = self.solver.solve_position(pos_str, analyze=True)
                current_score = self.solver.solve_position(pos_str)
                
                # Score after move
                next_pos = pos_str + str(col)
                next_score = self.solver.solve_position(next_pos)
                
                # Solver advantage (normalized)
                solver_advantage = (next_score - current_score) / 100.0
                
            except Exception as e:
                print(f"Solver error: {e}")
                solver_advantage = 0.0
            
            # Monte Carlo evaluation
            current_win_pct = ConnectFour.calculate_win_percentage(
                copy.deepcopy(game),
                self.num_simulations
            )
            next_win_pct = ConnectFour.calculate_win_percentage_after_move(
                copy.deepcopy(game),
                col,
                self.num_simulations
            )
            win_pct_advantage = next_win_pct - current_win_pct
            
            # Combined reward
            reward = (
                self.solver_weight * solver_advantage +
                self.monte_carlo_weight * win_pct_advantage
            )
            
            rewards.append(max(self.invalid_move_penalty, reward))
        
        return rewards
    
    def calculate_format_rewards(
        self,
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        Calculate format rewards for responses.
        
        Args:
            completions: List of model completions.
            
        Returns:
            List of format reward scores.
        """
        rewards = []
        
        for completion in completions:
            # Prepend <think> for GRPO completions
            if not completion.startswith("<think>"):
                completion = "<think>" + completion
                
            if validate_response_structure(completion):
                rewards.append(self.format_bonus)
            else:
                rewards.append(0.0)
                
        return rewards