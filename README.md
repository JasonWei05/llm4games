# LLM4Games: Training LLMs to Play Games with RL and SFT

This repository contains a clean implementation for training Large Language Models (LLMs) to play games using Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).

## Features

- **Games Supported**:
  - Tic-Tac-Toe with minimax and Monte Carlo evaluation
  - Connect Four with C++ solver integration and Monte Carlo evaluation. The code for this is from (\href{https://github.com/PascalPons/connect4})

- **Training Methods**:
  - Supervised Fine-Tuning (SFT) on expert game data
  - Group Relative Policy Optimization (GRPO) with custom reward functions

- **Reward Design**:
  - Hybrid rewards combining game-theory optimal play and Monte Carlo simulations
  - Format rewards to ensure proper response structure
  - Configurable weights for different reward components

## Project Structure

```
llm4games/
├── src/
│   ├── games/              # Game implementations
│   │   ├── tictactoe.py    # Tic-Tac-Toe with minimax
│   │   └── connect_four.py # Connect Four game
│   ├── training/           # Training modules
│   │   ├── sft_trainer.py  # SFT trainer
│   │   ├── grpo_trainer.py # GRPO trainer
│   │   └── rewards.py      # Reward functions
│   ├── evaluation/         # Evaluation framework
│   │   └── evaluator.py    # Model evaluation
│   ├── utils/              # Utilities
│   │   ├── prompts.py      # Prompt templates
│   │   ├── parsing.py      # Response parsing
│   │   └── connect4_solver.py # C++ solver interface
│   └── config/             # Configuration
│       └── training_config.py # Training configurations
├── data/                   # Training data (JSON files)
├── models/                 # Saved models
├── connect4/               # C++ Connect4 solver
├── train_sft.py           # SFT training script
├── train_grpo.py          # GRPO training script
└── evaluate.py            # Evaluation script
```

## Installation

1. Install required packages:
```bash
pip install torch transformers datasets trl unsloth google-generativeai
```

2. Compile the Connect Four solver (if using Connect Four):
```bash
cd connect4
bash compile.sh
```

## Usage

### Supervised Fine-Tuning

Train a model with SFT on Tic-Tac-Toe data:

```bash
python train_sft.py \
    --model-name /path/to/base/model \
    --output-dir ./outputs-sft-ttt \
    --batch-size 16 \
    --epochs 2 \
    --learning-rate 1e-4 \
    --game tictactoe
```

### GRPO Training

Train with GRPO starting from an SFT checkpoint:

```bash
python train_grpo.py \
    --model-name ./outputs-sft-ttt/final_model \
    --output-dir ./outputs-grpo-ttt \
    --batch-size 16 \
    --max-steps 200 \
    --learning-rate 2e-6 \
    --game tictactoe \
    --minimax-weight 0.3 \
    --mc-weight 0.7
```

### Evaluation

Evaluate a trained model:

```bash
# Against random player
python evaluate.py \
    --model-path ./outputs-grpo-ttt/final_model \
    --opponent random \
    --num-games 100 \
    --game tictactoe

# Against optimal player
python evaluate.py \
    --model-path ./outputs-grpo-ttt/final_model \
    --opponent optimal \
    --num-games 100 \
    --game tictactoe

# Against another LLM
python evaluate.py \
    --model-path ./outputs-grpo-ttt/final_model \
    --opponent llm \
    --opponent-model gemini-2.0-flash \
    --api-key YOUR_API_KEY \
    --num-games 50 \
    --game tictactoe
```
