"""Tic-Tac-Toe game implementation with minimax AI and Monte Carlo simulation."""

from __future__ import annotations
import random
import copy
from typing import List, Tuple, Optional, Literal


class TicTacToe:
    """Tic-Tac-Toe game with various AI strategies."""
    
    BOARD_SIZE = 3
    
    def __init__(self):
        """Initialize a new Tic-Tac-Toe game."""
        self.reset()

    def reset(self) -> None:
        """Reset the game to initial state."""
        self.board_state: List[List[str]] = [['.'] * self.BOARD_SIZE for _ in range(self.BOARD_SIZE)]
        self.player: Literal['X', 'O'] = 'X'
        self.moves: int = 0

    def print_board(self) -> None:
        """Print the current board state to console."""
        for row in self.board_state:
            print(" ".join(row))
        print()

    def get_board_string(self) -> str:
        """
        Get board state as a formatted string.
        
        Returns:
            Formatted string representation of the board.
        """
        result = ""
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                cell = self.board_state[row][col]
                state = "free" if cell == '.' else cell
                result += f"({row+1}, {col+1}): {state}, "
            result += "\n"
        return result

    def is_valid_move(self, row: int, col: int) -> bool:
        """
        Check if a move is valid.
        
        Args:
            row: Row index (0-based).
            col: Column index (0-based).
            
        Returns:
            True if the move is valid, False otherwise.
        """
        if row < 0 or row >= self.BOARD_SIZE or col < 0 or col >= self.BOARD_SIZE:
            return False
        return self.board_state[row][col] == '.'

    def make_move(self, row: int, col: int) -> Literal["Not Valid", "Winner", "Draw", "Valid Move"]:
        """
        Make a move on the board.
        
        Args:
            row: Row index (0-based).
            col: Column index (0-based).
            
        Returns:
            Status of the move: "Not Valid", "Winner", "Draw", or "Valid Move".
        """
        if not self.is_valid_move(row, col):
            return "Not Valid"
            
        self.board_state[row][col] = self.player
        self.moves += 1
        
        if self.check_winner():
            return "Winner"
            
        self.switch_player()
        
        if self.moves == self.BOARD_SIZE ** 2:
            return "Draw"
            
        return "Valid Move"

    def make_random_move(self) -> Literal["Draw", "Winner", "Valid Move"]:
        """
        Make a random valid move.
        
        Returns:
            Status of the move: "Draw", "Winner", or "Valid Move".
        """
        valid_moves = [
            (r, c) 
            for r in range(self.BOARD_SIZE) 
            for c in range(self.BOARD_SIZE) 
            if self.is_valid_move(r, c)
        ]
        
        if not valid_moves:
            return "Draw"
            
        row, col = random.choice(valid_moves)
        return self.make_move(row, col)

    def switch_player(self) -> None:
        """Switch the current player."""
        self.player = 'O' if self.player == 'X' else 'X'

    def check_winner(self) -> bool:
        """
        Check if there is a winner.
        
        Returns:
            True if there is a winner, False otherwise.
        """
        # Check rows
        for i in range(self.BOARD_SIZE):
            if (self.board_state[i][0] == self.board_state[i][1] == 
                self.board_state[i][2] != '.'):
                return True
                
        # Check columns
        for i in range(self.BOARD_SIZE):
            if (self.board_state[0][i] == self.board_state[1][i] == 
                self.board_state[2][i] != '.'):
                return True
                
        # Check diagonals
        if (self.board_state[0][0] == self.board_state[1][1] == 
            self.board_state[2][2] != '.'):
            return True
            
        if (self.board_state[0][2] == self.board_state[1][1] == 
            self.board_state[2][0] != '.'):
            return True
            
        return False
    
    @staticmethod
    def calculate_win_percentage(
        game: TicTacToe, 
        num_simulations: int = 5000
    ) -> float:
        """
        Calculate win percentage using Monte Carlo simulation.
        
        Args:
            game: Current game state.
            num_simulations: Number of simulations to run.
            
        Returns:
            Win percentage as (wins - losses) / num_simulations.
        """
        initial_player = game.player
        wins = 0
        losses = 0
        
        for _ in range(num_simulations):
            temp_game = copy.deepcopy(game)
            
            while True:
                result = temp_game.make_random_move()
                
                if result == "Winner":
                    # The player who just moved is the winner
                    winning_player = 'X' if temp_game.player == 'O' else 'O'
                    if winning_player == initial_player:
                        wins += 1
                    else:
                        losses += 1
                    break
                elif result == "Draw":
                    break
                    
        return (wins - losses) / num_simulations
    
    @staticmethod
    def calculate_win_percentage_after_move(
        game: TicTacToe, 
        row: int, 
        col: int, 
        num_simulations: int = 5000
    ) -> float:
        """
        Calculate win percentage after making a specific move.
        
        Args:
            game: Current game state.
            row: Row index of the move.
            col: Column index of the move.
            num_simulations: Number of simulations to run.
            
        Returns:
            Win percentage after the move.
        """
        initial_player = game.player
        game_copy = copy.deepcopy(game)
        game_copy.make_move(row, col)
        
        wins = 0
        losses = 0
        
        for _ in range(num_simulations):
            temp_game = copy.deepcopy(game_copy)
            
            while True:
                result = temp_game.make_random_move()
                
                if result == "Winner":
                    winning_player = 'X' if temp_game.player == 'O' else 'O'
                    if winning_player == initial_player:
                        wins += 1
                    else:
                        losses += 1
                    break
                elif result == "Draw":
                    break
                    
        return (wins - losses) / num_simulations
    
    @staticmethod
    def minimax(
        game: TicTacToe, 
        bot_symbol: Literal['X', 'O']
    ) -> Tuple[int, Optional[int], Optional[int], Optional[List[Tuple[int, int]]]]:
        """
        Minimax algorithm for optimal play.
        
        Args:
            game: Current game state.
            bot_symbol: Symbol of the bot player.
            
        Returns:
            Tuple of (score, best_row, best_col, all_best_moves).
        """
        # Terminal states
        if game.check_winner():
            winner = 'X' if game.player == 'O' else 'O'  # Last player who moved
            score = 10 - game.moves if winner == bot_symbol else -(10 - game.moves)
            return score, None, None, None

        if game.moves == game.BOARD_SIZE ** 2:  # Draw
            return 0, None, None, None

        # Generate moves (prioritize center, then corners, then edges)
        move_order = [(1, 1)] + \
                    [(0, 0), (0, 2), (2, 0), (2, 2)] + \
                    [(0, 1), (1, 0), (1, 2), (2, 1)]
        possible_moves = [(r, c) for (r, c) in move_order if game.is_valid_move(r, c)]

        score_move_pairs = []
        for r, c in possible_moves:
            next_game = copy.deepcopy(game)
            next_game.make_move(r, c)
            child_score, *_ = TicTacToe.minimax(next_game, bot_symbol)
            score_move_pairs.append(((r, c), child_score))

        # Maximize or minimize based on current player
        if game.player == bot_symbol:  # Bot's turn - maximize
            best_score = max(score for _, score in score_move_pairs)
        else:  # Opponent's turn - minimize
            best_score = min(score for _, score in score_move_pairs)

        best_moves = [move for move, score in score_move_pairs if score == best_score]
        chosen_move = random.choice(best_moves)

        return best_score, chosen_move[0], chosen_move[1], best_moves
    
    @staticmethod
    def minimax_after_move(
        game: TicTacToe, 
        row: int, 
        col: int, 
        bot_symbol: Literal['X', 'O']
    ) -> Tuple[int, Optional[int], Optional[int], Optional[List[Tuple[int, int]]]]:
        """
        Run minimax after making a specific move.
        
        Args:
            game: Current game state.
            row: Row index of the move.
            col: Column index of the move.
            bot_symbol: Symbol of the bot player.
            
        Returns:
            Minimax result after the move.
        """
        game_copy = copy.deepcopy(game)
        game_copy.make_move(row, col)
        return TicTacToe.minimax(game_copy, bot_symbol)

    @staticmethod
    def get_optimal_move(game: TicTacToe) -> Tuple[int, int]:
        """
        Get the optimal move for the current player.
        
        Args:
            game: Current game state.
            
        Returns:
            Tuple of (row, col) for the optimal move.
        """
        _, row, col, _ = TicTacToe.minimax(game, game.player)
        return row, col