# AI-Math-Reasoning Documentation

This directory contains documentation for the AI-Math-Reasoning project, which enhances mathematical reasoning in large language models through Reasoning Distillation, GRPO, and Multi-agent PRM Reranking.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Repository Structure](#repository-structure)
4. [Training Pipeline](#training-pipeline)
5. [Inference Pipeline](#inference-pipeline)
6. [Evaluation](#evaluation)
7. [Examples](#examples)

## Project Overview

AI-Math-Reasoning is a framework for enhancing mathematical reasoning capabilities in large language models (LLMs) for solving complex mathematical problems like those in the AI Mathematical Olympiad (AIMO) challenge. It combines three key innovations:

1. **Reasoning Capability Distillation**: Transferring mathematical reasoning from DeepSeek R1 to Qwen2.5 32B using a comprehensive Chain-of-Thought (CoT) dataset derived from NuminaMath.

2. **Group Relative Policy Optimization (GRPO)**: Applying reinforcement learning on the distilled Qwen2.5 32B model to further enhance mathematical reasoning capabilities.

3. **Multi-agent PRM Framework**: Implementing a sophisticated multi-agent system utilizing Preference Reward Models (PRMs) for solution reranking and verification.

## Installation

To install AI-Math-Reasoning, use the following commands:

```bash
# Clone the repository
git clone https://github.com/yourusername/AI-Math-Reasoning.git
cd AI-Math-Reasoning

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Repository Structure

The repository is organized as follows:

```
AI-Math-Reasoning/
├── data/                     # Data handling
│   ├── processing/           # Data processing scripts
│   └── datasets/             # Dataset class implementations
├── src/ai_math_reasoning/    # Core source code
│   ├── distillation/         # Reasoning capability distillation
│   ├── grpo/                 # GRPO implementation
│   ├── prm/                  # PRM implementation and multi-agent framework
│   ├── inference/            # Inference pipeline and optimization
│   ├── utils/                # Shared utilities
│   └── models/               # Model implementations
├── scripts/                  # Utility scripts
│   ├── train.py              # Main training script
│   └── eval.py               # Evaluation script
├── examples/                 # Example scripts
│   └── simple_inference.py   # Simple inference example
├── docs/                     # Documentation
└── configs/                  # Configuration files
```

## Training Pipeline

The training process consists of three main phases:

### Phase 1: Knowledge Transfer

The first phase involves distilling reasoning capabilities from the DeepSeek R1 model to the Qwen2.5 32B model using a comprehensive Chain-of-Thought dataset.

To run distillation training:

```bash
python scripts/train.py \
    --mode distillation \
    --teacher_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --student_model Qwen/Qwen2.5-32B \
    --dataset bespokelabs/Bespoke-Stratos-17k \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 5e-6 \
    --temperature 1.0 \
    --alpha 0.5 \
    --output_dir outputs/distillation
```

### Phase 2: Reinforcement Learning with GRPO

The second phase involves applying Group Relative Policy Optimization (GRPO) to further enhance the model's mathematical reasoning capabilities.

To run GRPO training:

```bash
python scripts/train.py \
    --mode grpo \
    --model path/to/distilled/model \
    --dataset path/to/openr1/dataset \
    --batch_size 16 \
    --num_episodes 1000 \
    --learning_rate 5e-6 \
    --kl_coef 0.0 \
    --output_dir outputs/grpo
```

### Phase 3: Multi-agent PRM Framework

The final phase involves setting up a multi-agent framework with Preference Reward Models (PRMs) for solution reranking and verification.

## Inference Pipeline

The inference pipeline combines multi-agent generation, PRM reranking, verification, and ensemble methods to solve mathematical problems.

To perform inference:

```bash
python scripts/train.py \
    --mode inference \
    --model path/to/trained/model \
    --prm_model path/to/prm/model \
    --num_agents 5 \
    --max_attempts 2 \
    --problems "Your math problem here" \
    --output_file outputs/results.json
```

Or use the simple inference example:

```bash
# Set environment variables for model paths (optional)
export MATH_MODEL_PATH=path/to/your/model
export PRM_MODEL_PATH=path/to/prm/model

# Run the example
python examples/simple_inference.py
```

## Evaluation

To evaluate a model on standard benchmarks:

```bash
python scripts/eval.py \
    --model path/to/trained/model \
    --model_type qwen \
    --benchmark gsm8k \
    --num_agents 5 \
    --optimization accuracy \
    --output_dir outputs/evaluation
```

Available benchmarks include:
- MATH
- GSM8K
- AIME (custom format)

## Examples

The `examples/` directory contains simple scripts demonstrating how to use the library:

- `simple_inference.py`: Shows how to use the inference pipeline to solve a math problem.

For more advanced usage, refer to the scripts in the `scripts/` directory.
