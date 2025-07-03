"""Supervised Fine-Tuning (SFT) trainer for game-playing LLMs."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

from ..config.training_config import ModelConfig, SFTConfig, DataConfig
from ..utils.prompts import build_tictactoe_sft_prompt
from ..games import TicTacToe


class GameSFTTrainer:
    """Trainer for supervised fine-tuning on game data."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        sft_config: SFTConfig,
        data_config: DataConfig
    ):
        """
        Initialize the SFT trainer.
        
        Args:
            model_config: Model configuration.
            sft_config: SFT training configuration.
            data_config: Data loading configuration.
        """
        self.model_config = model_config
        self.sft_config = sft_config
        self.data_config = data_config
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def load_model(self) -> None:
        """Load and prepare the model for training."""
        print("Loading model...")
        
        # Load base model
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
            random_state=self.sft_config.seed,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Set tokenizer properties
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        print("Model loaded successfully!")
    
    def load_tictactoe_data(self) -> Dataset:
        """
        Load Tic-Tac-Toe training data.
        
        Returns:
            Hugging Face Dataset object.
        """
        print("Loading Tic-Tac-Toe data...")
        
        data_dir = Path(self.data_config.ttt_data_dir)
        
        # Load data for both players
        samples = []
        
        for player, filename in self.data_config.ttt_sft_files.items():
            filepath = data_dir / filename
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for entry in data:
                # Build the full sample
                board = entry['board']
                llm_output = entry['llm_output']
                
                # Create game state for board string
                game = TicTacToe()
                game.board_state = [row[:] for row in board]
                board_str = game.get_board_string()
                
                # Build prompt and combine with output
                prompt = build_tictactoe_sft_prompt(board_str, player)
                full_text = prompt + llm_output + self.tokenizer.eos_token
                
                samples.append(full_text)
        
        # Shuffle data
        random.seed(self.sft_config.seed)
        random.shuffle(samples)
        
        print(f"Loaded {len(samples)} training samples")
        
        # Create dataset
        return Dataset.from_dict({"text": samples})
    
    def setup_trainer(self, dataset: Dataset) -> None:
        """
        Set up the SFT trainer.
        
        Args:
            dataset: Training dataset.
        """
        # Training arguments
        training_args = TrainingArguments(
            # Basic settings
            output_dir=self.sft_config.output_dir,
            num_train_epochs=self.sft_config.num_train_epochs,
            per_device_train_batch_size=self.sft_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.sft_config.gradient_accumulation_steps,
            
            # Optimizer settings
            learning_rate=self.sft_config.learning_rate,
            warmup_ratio=self.sft_config.warmup_ratio,
            optim=self.sft_config.optim,
            lr_scheduler_type=self.sft_config.lr_scheduler_type,
            max_grad_norm=self.sft_config.max_grad_norm,
            
            # Precision settings
            fp16=self.sft_config.fp16 and not is_bfloat16_supported(),
            bf16=self.sft_config.bf16 and is_bfloat16_supported(),
            
            # Logging and saving
            logging_steps=self.sft_config.logging_steps,
            save_total_limit=self.sft_config.save_total_limit,
            report_to=self.sft_config.report_to,
            
            # Other settings
            seed=self.sft_config.seed,
            remove_unused_columns=True,
        )
        
        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=dataset,
            dataset_text_field=self.sft_config.dataset_text_field,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer),
            max_seq_length=self.model_config.max_seq_length,
            packing=self.sft_config.packing,
        )
        
        # Configure response-only training
        print("Configuring response-only training...")
        
        # Define where user input ends and assistant response begins
        user_header = "<|im_start|>user"
        assistant_header = "<|im_start|>assistant\nLet me solve this step by step.\n"
        
        self.trainer = train_on_responses_only(
            self.trainer, 
            user_header, 
            assistant_header
        )
    
    def train(self) -> None:
        """Run the training process."""
        print("Starting training...")
        
        # Train
        trainer_stats = self.trainer.train()
        
        print("Training complete!")
        
        # Save the final model
        best_model_path = Path(self.sft_config.output_dir) / "final_model"
        print(f"Saving model to {best_model_path}")
        
        self.trainer.save_model(str(best_model_path))
        self.tokenizer.save_pretrained(str(best_model_path))
        
        return trainer_stats
    
    def run(self, game: str = "tictactoe") -> None:
        """
        Run the complete SFT training pipeline.
        
        Args:
            game: Which game to train on ("tictactoe" or "connect_four").
        """
        # Load model
        self.load_model()
        
        # Load data
        if game == "tictactoe":
            dataset = self.load_tictactoe_data()
        else:
            raise NotImplementedError(f"SFT not implemented for {game}")
        
        # Setup trainer
        self.setup_trainer(dataset)
        
        # Train
        self.train()