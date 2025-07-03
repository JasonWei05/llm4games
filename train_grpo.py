#!/usr/bin/env python3
"""Script to run GRPO training for game-playing LLMs."""

import argparse
from pathlib import Path

from src.config.training_config import ModelConfig, GRPOConfig, RewardConfig, DataConfig
from src.training.grpo_trainer import GameGRPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train LLM with GRPO on game data")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Path to base model (can be SFT checkpoint)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--game",
        type=str,
        choices=["tictactoe", "connect_four"],
        default="tictactoe",
        help="Which game to train on"
    )
    parser.add_argument(
        "--minimax-weight",
        type=float,
        default=0.3,
        help="Weight for minimax/solver score in reward"
    )
    parser.add_argument(
        "--mc-weight",
        type=float,
        default=0.7,
        help="Weight for Monte Carlo win percentage in reward"
    )
    
    args = parser.parse_args()
    
    # Create configurations
    model_config = ModelConfig(
        model_name=args.model_name,
        max_seq_length=3072,
        load_in_4bit=True
    )
    
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate
    )
    
    reward_config = RewardConfig(
        ttt_minimax_weight=args.minimax_weight,
        ttt_monte_carlo_weight=args.mc_weight,
        c4_solver_weight=args.minimax_weight,
        c4_monte_carlo_weight=args.mc_weight
    )
    
    data_config = DataConfig()
    
    # Create and run trainer
    trainer = GameGRPOTrainer(model_config, grpo_config, reward_config, data_config)
    trainer.run(game=args.game)


if __name__ == "__main__":
    main()