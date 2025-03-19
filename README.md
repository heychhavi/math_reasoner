# AI Math Reasoning

A framework for enhancing mathematical reasoning capabilities in large language models (LLMs) through various techniques including Group Relative Policy Optimization (GRPO), Preference Reward Modeling (PRM), and multi-agent verification.

## Overview

This repository contains the implementation of our Stanford CS329A (Self-Improving AI Agents) final project, which focuses on developing techniques to improve the mathematical reasoning capabilities of large language models.

Our approach combines several techniques:

1. **Group Relative Policy Optimization (GRPO)**: A novel reinforcement learning method that uses group-relative rewards for more stable training.
2. **Preference Reward Modeling (PRM)**: A technique to rank solution quality using learned preferences.
3. **Multi-agent reasoning**: Using multiple agent instances to generate diverse solutions and select the best ones.
4. **Self-verification**: Enabling models to verify their own solutions, either through direct verification or comparing solutions from multiple agents.

## Features

- **Modular design**: Easily swap different base models (DeepSeek, Qwen, etc.)
- **Efficient fine-tuning**: Support for quantization (8-bit, 4-bit) and parameter-efficient fine-tuning (LoRA)
- **Multi-agent reasoning**: Generate and rank multiple solution attempts to improve accuracy
- **Solution verification**: Verify mathematical solutions step-by-step or by comparing with reference answers
- **Dataset utilities**: Tools for processing and handling mathematical datasets
- **Training pipelines**: Scripts for training models with various techniques (GRPO, PRM, SFT)
- **Evaluation tools**: Benchmark on standard mathematical datasets like AIME

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/AI-Math-Reasoning.git
cd AI-Math-Reasoning

# Install the package and dependencies
pip install -e .
```

## Quick Start

### Simple Inference

```python
from ai_math_reasoning.models import create_model
from ai_math_reasoning.data.processing.math_processing import format_problem_with_template

# Load a pre-trained model
model = create_model(
    model_type="deepseek",
    model_name_or_path="deepseek-ai/deepseek-math-7b-instruct",
    load_in_8bit=True
)

# Solve a math problem
problem = "If 3x + 5y = 15 and 2x - 3y = 13, find the value of x + y."
prompt = format_problem_with_template(problem)
solution = model.generate(prompt, max_new_tokens=512)

print(solution)
```

### Multi-Agent Reasoning

```python
from ai_math_reasoning.models import create_model
from ai_math_reasoning.prm.multi_agent import MultiAgentReasoner
from ai_math_reasoning.prm.reranker import PRMReranker

# Load a base model
base_model = create_model(
    model_type="deepseek",
    model_name_or_path="deepseek-ai/deepseek-math-7b-instruct",
    load_in_8bit=True
)

# Create a reranker
reranker = PRMReranker(base_model=base_model, inference_only=True)

# Create a multi-agent reasoner
reasoner = MultiAgentReasoner(
    solver_model=base_model,
    reranker=reranker,
    num_agents=5
)

# Solve a problem
problem = "Find all real values of x that satisfy the equation x^4 - 5x^2 + 4 = 0."
result = reasoner.solve_problem(problem)

# Print the best solution
print(f"Best solution (score: {result['best_solution']['score']}):")
print(result['best_solution']['solution'])
```

## Training Models

### Fine-tuning with GRPO

```bash
python scripts/train.py \
    --mode grpo \
    --model_type deepseek \
    --model_name_or_path deepseek-ai/deepseek-math-7b-instruct \
    --output_dir ./models/grpo_math_7b \
    --dataset_name bespokelabs/Bespoke-Stratos-17k \
    --num_episodes 1000 \
    --load_in_8bit True \
    --use_lora True
```

### Training a Preference Reward Model

```bash
python scripts/train.py \
    --mode prm \
    --model_type deepseek \
    --model_name_or_path deepseek-ai/deepseek-math-7b-base \
    --output_dir ./models/prm_math_7b \
    --dataset_path ./data/math_preferences.json \
    --num_epochs 3 \
    --load_in_8bit True \
    --use_lora True
```

## Project Structure

```
AI-Math-Reasoning/
├── configs/                # Configuration files
├── examples/               # Example scripts
├── scripts/                # Training and evaluation scripts
├── src/                    # Source code
│   └── ai_math_reasoning/  # Main package
│       ├── data/           # Dataset and data processing utilities
│       ├── distillation/   # Knowledge distillation implementation
│       ├── grpo/           # Group Relative Policy Optimization
│       ├── inference/      # Inference utilities
│       ├── models/         # Model implementations
│       ├── prm/            # Preference Reward Modeling
│       └── utils/          # Utility functions
└── tests/                  # Unit tests
```

## Citation

If you use this codebase in your research, please cite our paper:

```
@article{agarwal2025aimathreasoningenhancing,
  title={Enhancing Mathematical Reasoning in Large Language Models through Group Relative Policy Optimization},
  author={Agarwal, Abhinav and Your-Co-Authors},
  journal={Stanford CS329A Final Project},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- Abhinav Agarwal
- Carlo Baronio
- Shubhra Mishra
- Shree Reddy
- Chhavi Sharma

## Acknowledgments

We would like to thank the Stanford CS329A teaching staff for their guidance and support throughout this project.
