"""Python interface to the C++ Connect4 solver."""

import subprocess
import os
import csv
from typing import List, Dict, Optional, Tuple


class Connect4Solver:
    """Python interface to the C++ Connect4 solver for game-theory optimal play."""
    
    INVALID_MOVE = -100  # Score for invalid moves
    
    def __init__(
        self, 
        executable_path: str = "/llm4games/connect4/connect4",
        opening_book_path: str = "/llm4games/connect4/opening_book.csv"
    ):
        """
        Initialize the Connect4 solver.
        
        Args:
            executable_path: Path to the compiled C++ solver executable.
            opening_book_path: Path to the opening book CSV file.
            
        Raises:
            FileNotFoundError: If the executable or opening book is not found.
        """
        self.executable_path = executable_path
        self.opening_book_path = opening_book_path
        
        # Verify executable exists
        if not os.path.isfile(executable_path):
            raise FileNotFoundError(f"Connect4 solver executable not found at {executable_path}")
        
        # Load opening book
        self.opening_book: Dict[str, List[int]] = {}
        if opening_book_path and os.path.isfile(opening_book_path):
            with open(opening_book_path, newline='') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header row
                for row in reader:
                    position = row[0]
                    scores = list(map(int, row[1:]))
                    self.opening_book[position] = scores
    
    def solve_position(
        self, 
        position: str, 
        weak: bool = False, 
        analyze: bool = False
    ) -> Optional[int] | List[int]:
        """
        Solve a Connect4 position using the C++ solver.
        
        Args:
            position: String of moves (e.g., "4455" means columns 4,4,5,5).
                     Column numbers are 0-based in input, converted to 1-based for solver.
            weak: Use weak solver (faster but less accurate).
            analyze: Return scores for all possible moves instead of position score.
            
        Returns:
            If analyze=False: Score of the position (int).
            If analyze=True: List of scores for each column.
        """
        if not isinstance(position, str):
            raise ValueError("Position must be a string")
        
        # Convert 0-based to 1-based column indices
        if position:
            position_chars = list(position)
            for i, move in enumerate(position_chars):
                move_int = int(move) + 1
                position_chars[i] = str(move_int)
            position = "".join(position_chars)
            
        # Check opening book for short positions
        if len(position) <= 3:
            book_key = "-1" if position == "" else position
            if book_key in self.opening_book:
                return self.opening_book[book_key]
                
        # Build command
        cmd = [self.executable_path]
        
        if weak:
            cmd.append("-w")
        
        if analyze:
            cmd.append("-a")
        
        if self.opening_book_path:
            cmd.extend(["-b", self.opening_book_path])
        
        # Run solver
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=position + "\n", timeout=60)
            
            # Parse output
            lines = stdout.strip().split("\n")
            if not lines:
                raise ValueError("No output from solver")
            
            result = lines[0].split()
            
            if analyze:
                # Return scores for all columns
                if len(result) <= 1:
                    raise ValueError(f"Invalid output format for position {position}: {result}")
                scores = [int(score) for score in result[1:]]
                return scores
            else:
                # Return position score
                if len(result) != 2:
                    raise ValueError(f"Invalid output format: {result}")
                return int(result[1])
                
        except subprocess.TimeoutExpired:
            process.kill()
            raise TimeoutError("Solver process timed out")
        except Exception as e:
            raise RuntimeError(f"Error running solver: {e}")
    
    def batch_solve(
        self, 
        positions: List[str], 
        weak: bool = False, 
        analyze: bool = False
    ) -> List[Optional[int] | List[int]]:
        """
        Solve multiple positions in batch.
        
        Args:
            positions: List of position strings.
            weak: Use weak solver.
            analyze: Return scores for all moves.
            
        Returns:
            List of results (scores or lists of scores).
        """
        if not positions:
            return []
            
        # Convert positions to 1-based
        converted_positions = []
        for pos in positions:
            if pos:
                chars = [str(int(c) + 1) for c in pos]
                converted_positions.append("".join(chars))
            else:
                converted_positions.append("")
        
        # Build command
        cmd = [self.executable_path]
        
        if weak:
            cmd.append("-w")
        
        if analyze:
            cmd.append("-a")
            
        if self.opening_book_path:
            cmd.extend(["-b", self.opening_book_path])
        
        input_data = "\n".join(converted_positions) + "\n"
        
        # Run solver
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=input_data, timeout=600)
            
            # Parse output
            results = []
            lines = stdout.strip().split("\n")
            
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                
                if analyze:
                    # First item is position, rest are scores
                    if len(parts) <= 1:
                        results.append(None)
                    else:
                        scores = [int(score) for score in parts[1:]]
                        results.append(scores)
                else:
                    # Just the score
                    if len(parts) != 2:
                        results.append(None)
                    else:
                        results.append(int(parts[1]))
            
            return results
                
        except subprocess.TimeoutExpired:
            process.kill()
            raise TimeoutError("Solver process timed out")
        except Exception as e:
            raise RuntimeError(f"Error running batch solver: {e}")
    
    def get_best_move(self, position: str) -> Tuple[int, int]:
        """
        Get the best move for a given position.
        
        Args:
            position: Current position string.
            
        Returns:
            Tuple of (column, score) for the best move.
        """
        scores = self.solve_position(position, analyze=True)
        
        if not isinstance(scores, list):
            raise ValueError("Expected list of scores from analyze mode")
            
        # Find best valid move
        best_score = float('-inf')
        best_col = None
        
        for col, score in enumerate(scores):
            if score > self.INVALID_MOVE and score > best_score:
                best_score = score
                best_col = col
                
        if best_col is None:
            raise ValueError("No valid moves available")
            
        return best_col, best_score