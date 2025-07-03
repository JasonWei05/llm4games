"""Evaluation framework for game-playing LLMs."""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Literal, Optional
from dataclasses import dataclass, asdict

import torch
from unsloth import FastLanguageModel
import google.generativeai as genai

from ..games import TicTacToe, ConnectFour
from ..utils.prompts import build_tictactoe_prompt, build_connect_four_prompt
from ..utils.parsing import extract_tictactoe_move, extract_connect_four_move


@dataclass
class EvaluationResult:
    """Results from an evaluation run."""
    
    wins: int = 0
    draws: int = 0
    losses: int = 0
    invalid_moves: int = 0
    format_errors: int = 0
    total_games: int = 0
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games
    
    @property
    def valid_move_rate(self) -> float:
        """Calculate valid move rate."""
        total_moves = self.wins + self.draws + self.losses + self.invalid_moves
        if total_moves == 0:
            return 0.0
        return 1.0 - (self.invalid_moves / total_moves)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with computed metrics."""
        result = asdict(self)
        result['win_rate'] = self.win_rate
        result['valid_move_rate'] = self.valid_move_rate
        return result


class GameEvaluator:
    """Evaluator for game-playing LLMs."""
    
    def __init__(
        self,
        model_path: str,
        max_seq_length: int = 3072,
        device: str = "cuda",
        load_in_4bit: bool = True
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model.
            max_seq_length: Maximum sequence length.
            device: Device to run on.
            load_in_4bit: Whether to load in 4-bit quantization.
        """
        self.model_path = model_path
        self.device = device
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)
        print("Model loaded!")
    
    def get_model_move_tictactoe(
        self, 
        game: TicTacToe,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 64,
        max_retries: int = 3
    ) -> Optional[Tuple[int, int]]:
        """
        Get a move from the model for Tic-Tac-Toe.
        
        Args:
            game: Current game state.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            max_retries: Maximum retries for invalid responses.
            
        Returns:
            Move as (row, col) tuple or None if failed.
        """
        prompt = build_tictactoe_prompt(game.get_board_string(), game.player)
        
        for _ in range(max_retries):
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract move
            move = extract_tictactoe_move(response)
            
            if move and game.is_valid_move(move[0] - 1, move[1] - 1):
                return (move[0] - 1, move[1] - 1)  # Convert to 0-based
        
        return None
    
    def evaluate_vs_random_tictactoe(
        self,
        num_games: int = 100,
        player_side: Literal['X', 'O'] = 'X',
        verbose: bool = False
    ) -> EvaluationResult:
        """
        Evaluate the model against a random player in Tic-Tac-Toe.
        
        Args:
            num_games: Number of games to play.
            player_side: Which side the model plays.
            verbose: Whether to print game progress.
            
        Returns:
            Evaluation results.
        """
        results = EvaluationResult(total_games=num_games)
        
        for game_num in range(num_games):
            if verbose and game_num % 10 == 0:
                print(f"Playing game {game_num + 1}/{num_games}")
            
            game = TicTacToe()
            model_turn = (player_side == 'X')
            
            while True:
                if model_turn:
                    # Model's turn
                    move = self.get_model_move_tictactoe(game)
                    
                    if move is None:
                        results.format_errors += 1
                        results.losses += 1
                        break
                    
                    result = game.make_move(move[0], move[1])
                    
                    if result == "Not Valid":
                        results.invalid_moves += 1
                        results.losses += 1
                        break
                    elif result == "Winner":
                        results.wins += 1
                        break
                    elif result == "Draw":
                        results.draws += 1
                        break
                else:
                    # Random opponent's turn
                    result = game.make_random_move()
                    
                    if result == "Winner":
                        results.losses += 1
                        break
                    elif result == "Draw":
                        results.draws += 1
                        break
                
                model_turn = not model_turn
        
        return results
    
    def evaluate_vs_optimal_tictactoe(
        self,
        num_games: int = 100,
        player_side: Literal['X', 'O'] = 'X',
        verbose: bool = False
    ) -> EvaluationResult:
        """
        Evaluate the model against optimal minimax player in Tic-Tac-Toe.
        
        Args:
            num_games: Number of games to play.
            player_side: Which side the model plays.
            verbose: Whether to print game progress.
            
        Returns:
            Evaluation results.
        """
        results = EvaluationResult(total_games=num_games)
        
        for game_num in range(num_games):
            if verbose and game_num % 10 == 0:
                print(f"Playing game {game_num + 1}/{num_games}")
            
            game = TicTacToe()
            model_turn = (player_side == 'X')
            
            while True:
                if model_turn:
                    # Model's turn
                    move = self.get_model_move_tictactoe(game)
                    
                    if move is None:
                        results.format_errors += 1
                        results.losses += 1
                        break
                    
                    result = game.make_move(move[0], move[1])
                    
                    if result == "Not Valid":
                        results.invalid_moves += 1
                        results.losses += 1
                        break
                    elif result == "Winner":
                        results.wins += 1
                        break
                    elif result == "Draw":
                        results.draws += 1
                        break
                else:
                    # Optimal opponent's turn
                    optimal_move = TicTacToe.get_optimal_move(game)
                    result = game.make_move(optimal_move[0], optimal_move[1])
                    
                    if result == "Winner":
                        results.losses += 1
                        break
                    elif result == "Draw":
                        results.draws += 1
                        break
                
                model_turn = not model_turn
        
        return results
    
    def evaluate_vs_llm_tictactoe(
        self,
        opponent_model: str,
        api_key: str,
        num_games: int = 50,
        player_side: Literal['X', 'O'] = 'X',
        verbose: bool = False
    ) -> EvaluationResult:
        """
        Evaluate the model against another LLM in Tic-Tac-Toe.
        
        Args:
            opponent_model: Name of the opponent model (e.g., "gemini-2.0-flash").
            api_key: API key for the opponent model.
            num_games: Number of games to play.
            player_side: Which side the model plays.
            verbose: Whether to print game progress.
            
        Returns:
            Evaluation results.
        """
        # Initialize opponent
        genai.configure(api_key=api_key)
        opponent = genai.GenerativeModel(opponent_model)
        generation_config = genai.types.GenerationConfig(
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            max_output_tokens=1024,
            candidate_count=1
        )
        
        results = EvaluationResult(total_games=num_games)
        
        for game_num in range(num_games):
            if verbose:
                print(f"Playing game {game_num + 1}/{num_games}")
            
            game = TicTacToe()
            model_turn = (player_side == 'X')
            
            while True:
                if model_turn:
                    # Our model's turn
                    move = self.get_model_move_tictactoe(game)
                    
                    if move is None:
                        results.format_errors += 1
                        results.losses += 1
                        break
                    
                    result = game.make_move(move[0], move[1])
                    
                    if result == "Not Valid":
                        results.invalid_moves += 1
                        results.losses += 1
                        break
                    elif result == "Winner":
                        results.wins += 1
                        break
                    elif result == "Draw":
                        results.draws += 1
                        break
                else:
                    # Opponent LLM's turn
                    opponent_player = 'O' if player_side == 'X' else 'X'
                    prompt = build_tictactoe_prompt(game.get_board_string(), opponent_player)
                    
                    # Get opponent move
                    move = None
                    for _ in range(3):
                        try:
                            response = opponent.generate_content(
                                prompt, 
                                generation_config=generation_config
                            )
                            move = extract_tictactoe_move(response.text)
                            if move and game.is_valid_move(move[0] - 1, move[1] - 1):
                                move = (move[0] - 1, move[1] - 1)
                                break
                        except Exception:
                            continue
                    
                    if move is None:
                        # Opponent made invalid move
                        results.wins += 1
                        break
                    
                    result = game.make_move(move[0], move[1])
                    
                    if result == "Winner":
                        results.losses += 1
                        break
                    elif result == "Draw":
                        results.draws += 1
                        break
                
                model_turn = not model_turn
        
        return results
    
    def save_results(self, results: EvaluationResult, filepath: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results.
            filepath: Path to save results.
        """
        with open(filepath, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"Results saved to {filepath}")