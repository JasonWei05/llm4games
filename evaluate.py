#!/usr/bin/env python3
"""Script to evaluate trained game-playing LLMs."""

import argparse
from pathlib import Path

from src.evaluation.evaluator import GameEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained game-playing LLM")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["random", "optimal", "llm"],
        default="random",
        help="Type of opponent to play against"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of games to play"
    )
    parser.add_argument(
        "--player-side",
        type=str,
        choices=["X", "O"],
        default="X",
        help="Which side the model plays"
    )
    parser.add_argument(
        "--game",
        type=str,
        choices=["tictactoe"],
        default="tictactoe",
        help="Which game to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress during evaluation"
    )
    
    # LLM opponent settings
    parser.add_argument(
        "--opponent-model",
        type=str,
        default="gemini-2.0-flash",
        help="Model name for LLM opponent"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for LLM opponent"
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = GameEvaluator(args.model_path)
    
    # Run evaluation
    if args.game == "tictactoe":
        if args.opponent == "random":
            results = evaluator.evaluate_vs_random_tictactoe(
                num_games=args.num_games,
                player_side=args.player_side,
                verbose=args.verbose
            )
        elif args.opponent == "optimal":
            results = evaluator.evaluate_vs_optimal_tictactoe(
                num_games=args.num_games,
                player_side=args.player_side,
                verbose=args.verbose
            )
        elif args.opponent == "llm":
            if not args.api_key:
                raise ValueError("API key required for LLM opponent")
            results = evaluator.evaluate_vs_llm_tictactoe(
                opponent_model=args.opponent_model,
                api_key=args.api_key,
                num_games=args.num_games,
                player_side=args.player_side,
                verbose=args.verbose
            )
    else:
        raise NotImplementedError(f"Evaluation not implemented for {args.game}")
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Total games: {results.total_games}")
    print(f"Wins: {results.wins} ({results.win_rate:.1%})")
    print(f"Draws: {results.draws}")
    print(f"Losses: {results.losses}")
    print(f"Invalid moves: {results.invalid_moves}")
    print(f"Format errors: {results.format_errors}")
    print(f"Valid move rate: {results.valid_move_rate:.1%}")
    
    # Save results
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()