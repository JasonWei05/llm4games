"""Training configuration classes."""

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    
    model_name: str
    max_seq_length: int = 3072
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    
    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: list = None
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning."""
    
    # Training parameters
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 2
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    
    # Optimizer settings
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.2
    
    # Output and logging
    output_dir: str = "./outputs-sft"
    logging_steps: int = 1
    save_total_limit: int = 2
    report_to: str = "wandb"
    seed: int = 3407
    
    # Model settings
    fp16: bool = False
    bf16: bool = True
    
    # Data settings
    dataset_text_field: str = "text"
    packing: bool = False


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    
    # Training parameters
    learning_rate: float = 2e-6
    beta: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "constant_with_warmup"
    
    # Batch settings
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 12
    num_generations: int = 16
    
    # Generation settings
    max_prompt_length: int = 256
    max_completion_length: int = 2816  # max_seq_length - max_prompt_length
    temperature: float = 1.0
    
    # Training duration
    max_steps: int = 200
    save_steps: int = 50
    
    # Output and logging
    output_dir: str = "./outputs-grpo"
    logging_steps: int = 1
    save_total_limit: int = 3
    report_to: str = "wandb"
    
    # Optimizer settings
    optim: str = "paged_adamw_8bit"
    max_grad_norm: float = 0.1


@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    
    # Tic-Tac-Toe rewards
    ttt_minimax_weight: float = 0.3
    ttt_monte_carlo_weight: float = 0.7
    ttt_num_simulations: int = 5000
    
    # Connect Four rewards
    c4_solver_weight: float = 0.5
    c4_monte_carlo_weight: float = 0.5
    c4_num_simulations: int = 2500
    
    # Common settings
    invalid_move_penalty: float = -2.0
    format_bonus: float = 0.1
    
    # Connect Four solver paths
    c4_solver_path: str = "/jet/home/billyli/data_folder/jwei/jwei/llm4games/connect4/connect4"
    c4_opening_book_path: str = "/jet/home/billyli/data_folder/jwei/jwei/llm4games/opening_book.csv"


@dataclass
class DataConfig:
    """Configuration for data loading."""
    
    # Data paths
    ttt_data_dir: str = "/jet/home/billyli/data_folder/jwei/jwei/llm4games"
    c4_data_dir: str = "/jet/home/billyli/data_folder/jwei/jwei/llm4games"
    
    # File names
    ttt_sft_files: dict = None
    ttt_grpo_files: dict = None
    c4_grpo_files: dict = None
    
    def __post_init__(self):
        if self.ttt_sft_files is None:
            self.ttt_sft_files = {
                'X': 'ttt_deepseek_data_x.json',
                'O': 'ttt_deepseek_data_o.json'
            }
        
        if self.ttt_grpo_files is None:
            self.ttt_grpo_files = {
                'X': 'ttt_grpo_x.json',
                'O': 'ttt_grpo_o.json'
            }
            
        if self.c4_grpo_files is None:
            self.c4_grpo_files = {
                'X': 'c4_grpo_x.json',
                'O': 'c4_grpo_o.json'
            }