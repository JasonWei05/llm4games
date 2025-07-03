"""Connect Four game implementation with Monte Carlo simulation."""

from __future__ import annotations
import random
import copy
from typing import List, Tuple, Literal, Optional


class ConnectFour:
    """Connect Four game with Monte Carlo evaluation."""
    
    ROWS = 6
    COLS = 7
    WIN_LENGTH = 4
    
    def __init__(self):
        """Initialize a new Connect Four game."""
        self.reset()

    def reset(self) -> None:
        """Reset the game to initial state."""
        self.board_state: List[List[str]] = [['.'] * self.COLS for _ in range(self.ROWS)]
        self.player: Literal['X', 'O'] = 'X'
        self.moves: int = 0

    def print_board(self) -> None:
        """Print the current board state to console."""
        for row in self.board_state:
            print(" ".join(row))
        print(" ".join(str(i+1) for i in range(self.COLS)))
        print()

    def get_board_string(self) -> str:
        """
        Get board state as a formatted string.
        
        Returns:
            Formatted string representation of the board.
        """
        result = ""
        for r in reversed(range(self.ROWS)):
            for c in range(self.COLS):
                cell = self.board_state[self.ROWS - r - 1][c]
                state = "free" if cell == '.' else cell
                result += f"({r+1}, {c+1}): {state}, "
            result += "\n"
        return result

    def is_valid_move(self, col: int) -> bool:
        """
        Check if a move is valid.
        
        Args:
            col: Column index (0-based).
            
        Returns:
            True if the move is valid, False otherwise.
        """
        return 0 <= col < self.COLS and self.board_state[0][col] == '.'

    def make_move(self, col: int) -> Literal["Not Valid", "Winner", "Draw", "Valid Move"]:
        """
        Make a move in the specified column.
        
        Args:
            col: Column index (0-based).
            
        Returns:
            Status of the move: "Not Valid", "Winner", "Draw", or "Valid Move".
        """
        if not self.is_valid_move(col):
            return "Not Valid"

        row = self._find_lowest_empty_row(col)
        self.board_state[row][col] = self.player
        self.moves += 1

        if self.check_winner(col):
            return "Winner"

        if self.moves == self.ROWS * self.COLS:
            return "Draw"

        self.switch_player()
        return "Valid Move"

    def make_random_move(self) -> Literal["Draw", "Winner", "Valid Move"]:
        """
        Make a random valid move.
        
        Returns:
            Status of the move: "Draw", "Winner", or "Valid Move".
        """
        if self.moves == self.ROWS * self.COLS:
            return "Draw"
            
        valid_cols = [c for c in range(self.COLS) if self.is_valid_move(c)]
        col = random.choice(valid_cols)
        return self.make_move(col)

    def switch_player(self) -> None:
        """Switch the current player."""
        self.player = 'O' if self.player == 'X' else 'X'

    def check_winner(self, col: int) -> bool:
        """
        Check if the last move in the given column resulted in a win.
        
        Args:
            col: Column of the last move.
            
        Returns:
            True if there is a winner, False otherwise.
        """
        # Find the row of the last placed piece
        row = next(
            (r for r in range(self.ROWS) if self.board_state[r][col] != '.'),
            None
        )
        if row is None:
            return False

        piece = self.board_state[row][col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            run_length = (
                self._count_in_direction(row, col, dr, dc, piece) +
                self._count_in_direction(row, col, -dr, -dc, piece) - 1
            )
            if run_length >= self.WIN_LENGTH:
                return True
        return False

    def _find_lowest_empty_row(self, col: int) -> int:
        """
        Find the lowest empty row in a column.
        
        Args:
            col: Column index.
            
        Returns:
            Row index of the lowest empty position.
            
        Raises:
            ValueError: If column is full.
        """
        for r in reversed(range(self.ROWS)):
            if self.board_state[r][col] == '.':
                return r
        raise ValueError("Column is full")

    def _count_in_direction(
        self, 
        row: int, 
        col: int, 
        delta_row: int, 
        delta_col: int, 
        piece: str
    ) -> int:
        """
        Count consecutive pieces in a given direction.
        
        Args:
            row: Starting row.
            col: Starting column.
            delta_row: Row direction (-1, 0, or 1).
            delta_col: Column direction (-1, 0, or 1).
            piece: Piece to count.
            
        Returns:
            Number of consecutive pieces.
        """
        count = 0
        r, c = row, col
        
        while (0 <= r < self.ROWS and 0 <= c < self.COLS and 
               self.board_state[r][c] == piece):
            count += 1
            r += delta_row
            c += delta_col
            
        return count

    @staticmethod
    def calculate_win_percentage(
        game: ConnectFour, 
        num_simulations: int = 2500
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
        game: ConnectFour, 
        col: int, 
        num_simulations: int = 2500
    ) -> float:
        """
        Calculate win percentage after making a specific move.
        
        Args:
            game: Current game state.
            col: Column index of the move.
            num_simulations: Number of simulations to run.
            
        Returns:
            Win percentage after the move.
            
        Raises:
            ValueError: If the move is invalid.
        """
        initial_player = game.player
        game_after_move = copy.deepcopy(game)
        move_result = game_after_move.make_move(col)
        
        if move_result == "Not Valid":
            raise ValueError(f"Column {col} is not a valid move")

        wins = 0
        losses = 0

        for _ in range(num_simulations):
            temp_game = copy.deepcopy(game_after_move)
            
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