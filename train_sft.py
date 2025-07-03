#!/usr/bin/env python3
"""Script to run supervised fine-tuning for game-playing LLMs."""

import argparse
from pathlib import Path

from src.config.training_config import ModelConfig, SFTConfig, DataConfig
from src.training.sft_trainer import GameSFTTrainer


def main():
    parser = argparse.ArgumentParser(description="Train LLM with SFT on game data")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Path to base model"
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
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--game",
        type=str,
        choices=["tictactoe"],
        default="tictactoe",
        help="Which game to train on"
    )
    
    args = parser.parse_args()
    
    # Create configurations
    model_config = ModelConfig(
        model_name=args.model_name,
        max_seq_length=3072,
        load_in_4bit=True
    )
    
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    data_config = DataConfig()
    
    # Create and run trainer
    trainer = GameSFTTrainer(model_config, sft_config, data_config)
    trainer.run(game=args.game)


if __name__ == "__main__":
    main()