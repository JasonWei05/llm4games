"""Group Relative Policy Optimization (GRPO) trainer for game-playing LLMs."""

import json
import random
import copy
from pathlib import Path
from typing import List, Dict, Any, Literal

import torch
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer as TRLGRPOTrainer
from unsloth import FastLanguageModel

from ..config.training_config import ModelConfig, GRPOConfig, DataConfig, RewardConfig
from ..utils.prompts import build_tictactoe_prompt, build_connect_four_prompt
from ..games import TicTacToe, ConnectFour
from .rewards import TicTacToeRewardCalculator, ConnectFourRewardCalculator


class GameGRPOTrainer:
    """Trainer for GRPO training on game data."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        grpo_config: GRPOConfig,
        reward_config: RewardConfig,
        data_config: DataConfig
    ):
        """
        Initialize the GRPO trainer.
        
        Args:
            model_config: Model configuration.
            grpo_config: GRPO training configuration.
            reward_config: Reward function configuration.
            data_config: Data loading configuration.
        """
        self.model_config = model_config
        self.grpo_config = grpo_config
        self.reward_config = reward_config
        self.data_config = data_config
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.reward_calculator = None
    
    def load_model(self) -> None:
        """Load and prepare the model for training."""
        print("Loading model...")
        
        # Load base model (can be pre-trained SFT checkpoint)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config.model_name,
            max_seq_length=self.model_config.max_seq_length,
            load_in_4bit=self.model_config.load_in_4bit,
            load_in_8bit=self.model_config.load_in_8bit,
        )
        
        # Apply LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.model_config.lora_r,
            target_modules=self.model_config.lora_target_modules,
            lora_alpha=self.model_config.lora_alpha,
            lora_dropout=self.model_config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        print("Model loaded successfully!")
    
    def load_tictactoe_data(self) -> List[Dict[str, Any]]:
        """
        Load Tic-Tac-Toe training data for GRPO.
        
        Returns:
            List of dataset entries with prompts and game states.
        """
        print("Loading Tic-Tac-Toe GRPO data...")
        
        data_dir = Path(self.data_config.ttt_data_dir)
        dataset = []
        
        # Load data for both players
        for player, filename in self.data_config.ttt_grpo_files.items():
            filepath = data_dir / filename
            
            with open(filepath, 'r') as f:
                board_states = json.load(f)
            
            for board in board_states:
                # Create game state
                game = TicTacToe()
                game.board_state = [row[:] for row in board]
                
                # Count moves to determine whose turn it is
                moves = sum(1 for row in board for cell in row if cell != '.')
                game.moves = moves
                game.player = player
                
                # Build prompt
                prompt = build_tictactoe_prompt(game.get_board_string(), player)
                
                dataset.append({
                    "prompt": prompt,
                    "game_state": game
                })
        
        # Shuffle data
        random.shuffle(dataset)
        print(f"Loaded {len(dataset)} training samples")
        
        return dataset
    
    def load_connect_four_data(self) -> List[Dict[str, Any]]:
        """
        Load Connect Four training data for GRPO.
        
        Returns:
            List of dataset entries with prompts and game states.
        """
        print("Loading Connect Four GRPO data...")
        
        data_dir = Path(self.data_config.c4_data_dir)
        dataset = []
        
        # Load data for both players
        for player, filename in self.data_config.c4_grpo_files.items():
            filepath = data_dir / filename
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for entry in data:
                board = entry['board']
                position_string = entry.get('position', '')
                
                # Create game state
                game = ConnectFour()
                game.board_state = [row[:] for row in board]
                
                # Count moves
                moves = sum(1 for row in board for cell in row if cell != '.')
                game.moves = moves
                game.player = player
                
                # Build prompt
                prompt = build_connect_four_prompt(game.get_board_string(), player)
                
                dataset.append({
                    "prompt": prompt,
                    "game_state": game,
                    "position_string": position_string
                })
        
        # Shuffle data
        random.shuffle(dataset)
        print(f"Loaded {len(dataset)} training samples")
        
        return dataset
    
    def setup_tictactoe_trainer(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Set up the GRPO trainer for Tic-Tac-Toe.
        
        Args:
            dataset: Training dataset.
        """
        # Initialize reward calculator
        self.reward_calculator = TicTacToeRewardCalculator(
            minimax_weight=self.reward_config.ttt_minimax_weight,
            monte_carlo_weight=self.reward_config.ttt_monte_carlo_weight,
            num_simulations=self.reward_config.ttt_num_simulations,
            invalid_move_penalty=self.reward_config.invalid_move_penalty,
            format_bonus=self.reward_config.format_bonus
        )
        
        # GRPO configuration
        training_args = TRLGRPOConfig(
            learning_rate=self.grpo_config.learning_rate,
            beta=self.grpo_config.beta,
            adam_beta1=self.grpo_config.adam_beta1,
            adam_beta2=self.grpo_config.adam_beta2,
            warmup_ratio=self.grpo_config.warmup_ratio,
            lr_scheduler_type=self.grpo_config.lr_scheduler_type,
            optim=self.grpo_config.optim,
            logging_steps=self.grpo_config.logging_steps,
            per_device_train_batch_size=self.grpo_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.grpo_config.gradient_accumulation_steps,
            num_generations=self.grpo_config.num_generations,
            max_prompt_length=self.grpo_config.max_prompt_length,
            max_completion_length=self.grpo_config.max_completion_length,
            max_steps=self.grpo_config.max_steps,
            save_steps=self.grpo_config.save_steps,
            max_grad_norm=self.grpo_config.max_grad_norm,
            temperature=self.grpo_config.temperature,
            report_to=self.grpo_config.report_to,
            output_dir=self.grpo_config.output_dir,
            save_total_limit=self.grpo_config.save_total_limit,
        )
        
        # Prepare dataset format
        trainer_dataset = [
            {
                "prompt": entry["prompt"],
                "answer": entry["game_state"]  # Pass game state for reward calculation
            }
            for entry in dataset
        ]
        
        # Initialize trainer
        self.trainer = TRLGRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[
                self.reward_calculator.calculate_format_rewards,
                self.reward_calculator.calculate_correctness_rewards,
            ],
            args=training_args,
            train_dataset=trainer_dataset
        )
    
    def setup_connect_four_trainer(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Set up the GRPO trainer for Connect Four.
        
        Args:
            dataset: Training dataset.
        """
        # Initialize reward calculator
        self.reward_calculator = ConnectFourRewardCalculator(
            solver_path=self.reward_config.c4_solver_path,
            opening_book_path=self.reward_config.c4_opening_book_path,
            solver_weight=self.reward_config.c4_solver_weight,
            monte_carlo_weight=self.reward_config.c4_monte_carlo_weight,
            num_simulations=self.reward_config.c4_num_simulations,
            invalid_move_penalty=self.reward_config.invalid_move_penalty,
            format_bonus=self.reward_config.format_bonus
        )
        
        # GRPO configuration
        training_args = TRLGRPOConfig(
            learning_rate=self.grpo_config.learning_rate,
            beta=self.grpo_config.beta,
            adam_beta1=self.grpo_config.adam_beta1,
            adam_beta2=self.grpo_config.adam_beta2,
            warmup_ratio=self.grpo_config.warmup_ratio,
            lr_scheduler_type=self.grpo_config.lr_scheduler_type,
            optim=self.grpo_config.optim,
            logging_steps=self.grpo_config.logging_steps,
            per_device_train_batch_size=self.grpo_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.grpo_config.gradient_accumulation_steps,
            num_generations=self.grpo_config.num_generations,
            max_prompt_length=self.grpo_config.max_prompt_length,
            max_completion_length=self.grpo_config.max_completion_length,
            max_steps=self.grpo_config.max_steps,
            save_steps=self.grpo_config.save_steps,
            max_grad_norm=self.grpo_config.max_grad_norm,
            temperature=self.grpo_config.temperature,
            report_to=self.grpo_config.report_to,
            output_dir=self.grpo_config.output_dir,
            save_total_limit=self.grpo_config.save_total_limit,
        )
        
        # Prepare dataset format
        trainer_dataset = [
            {
                "prompt": entry["prompt"],
                "answer": entry["game_state"],
                "position_string": entry["position_string"]
            }
            for entry in dataset
        ]
        
        # Initialize trainer
        self.trainer = TRLGRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[
                self.reward_calculator.calculate_format_rewards,
                self.reward_calculator.calculate_correctness_rewards,
            ],
            args=training_args,
            train_dataset=trainer_dataset
        )
    
    def train(self) -> None:
        """Run the training process."""
        print("Starting GRPO training...")
        
        # Train
        self.trainer.train()
        
        print("Training complete!")
        
        # Save the final model
        final_model_path = Path(self.grpo_config.output_dir) / "final_model"
        print(f"Saving model to {final_model_path}")
        
        self.model.save_pretrained(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
    
    def run(self, game: Literal["tictactoe", "connect_four"] = "tictactoe") -> None:
        """
        Run the complete GRPO training pipeline.
        
        Args:
            game: Which game to train on.
        """
        # Load model
        self.load_model()
        
        # Load data and setup trainer
        if game == "tictactoe":
            dataset = self.load_tictactoe_data()
            self.setup_tictactoe_trainer(dataset)
        elif game == "connect_four":
            dataset = self.load_connect_four_data()
            self.setup_connect_four_trainer(dataset)
        else:
            raise ValueError(f"Unknown game: {game}")
        
        # Train
        self.train()