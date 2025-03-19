#!/usr/bin/env python
"""
Evaluate AI-Math-Reasoning models on mathematical datasets.

This script evaluates models on mathematical reasoning datasets,
with options for multi-agent evaluation, verification, and ensemble methods.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import torch
from tqdm import tqdm

# Add parent directory to path to import from ai_math_reasoning
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_math_reasoning.models import create_model
from ai_math_reasoning.data.datasets.math_dataset import MathReasoningDataset
from ai_math_reasoning.inference.pipeline import MathInferencePipeline
from ai_math_reasoning.utils.training_utils import setup_logging, set_seed
from ai_math_reasoning.utils.data_utils import get_tokenizer, load_math_dataset
from ai_math_reasoning.data.processing.math_processing import (
    extract_final_answer,
    compare_answers
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate mathematical reasoning models")
    
    # Model parameters
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the model to evaluate"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="qwen",
        choices=["qwen", "deepseek", "base"],
        help="Type of model to use"
    )
    parser.add_argument(
        "--prm_model_path", 
        type=str, 
        default=None,
        help="Path to PRM model (optional)"
    )
    parser.add_argument(
        "--verifier_model_path", 
        type=str, 
        default=None,
        help="Path to verifier model (optional)"
    )
    parser.add_argument(
        "--precision", 
        type=str, 
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Precision to use for model weights"
    )
    
    # Multi-agent parameters
    parser.add_argument(
        "--num_agents", 
        type=int, 
        default=3,
        help="Number of agents to use"
    )
    parser.add_argument(
        "--max_attempts", 
        type=int, 
        default=1,
        help="Maximum solution attempts per agent"
    )
    parser.add_argument(
        "--use_verification", 
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to use solution verification"
    )
    parser.add_argument(
        "--use_ensemble", 
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to use ensemble solution selection"
    )
    parser.add_argument(
        "--temperature_min", 
        type=float, 
        default=0.7,
        help="Minimum temperature for sampling"
    )
    parser.add_argument(
        "--temperature_max", 
        type=float, 
        default=0.9,
        help="Maximum temperature for sampling"
    )
    parser.add_argument(
        "--top_p_min", 
        type=float, 
        default=0.9,
        help="Minimum nucleus sampling parameter"
    )
    parser.add_argument(
        "--top_p_max", 
        type=float, 
        default=0.99,
        help="Maximum nucleus sampling parameter"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=1024,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--timeout", 
        type=float, 
        default=None,
        help="Timeout in seconds per problem"
    )
    
    # Dataset parameters
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="numina/numina-math-filtered",
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test",
        choices=["train", "test", "validation"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=1000,
        help="Maximum number of samples to evaluate"
    )
    
    # Output parameters
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None,
        help="Path to output file"
    )
    parser.add_argument(
        "--include_solutions", 
        action="store_true",
        help="Whether to include full solutions in the output"
    )
    parser.add_argument(
        "--include_problems", 
        action="store_true",
        help="Whether to include full problems in the output"
    )
    
    # Other parameters
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def load_problems_and_answers(
    dataset_path: str,
    split: str,
    max_samples: int = 1000
) -> Tuple[List[str], List[str]]:
    """Load problems and reference answers from a dataset.
    
    Args:
        dataset_path: Path to dataset
        split: Dataset split
        max_samples: Maximum number of samples to load
        
    Returns:
        Tuple of (problems, answers)
    """
    # Load dataset
    logging.info(f"Loading dataset {dataset_path} (split: {split})")
    train_dataset, eval_dataset = load_math_dataset(
        dataset_paths=dataset_path,
        max_samples=max_samples,
        seed=42,
        train_split="train",
        eval_split=split if split != "train" else None
    )
    
    # Use appropriate dataset based on split
    dataset = train_dataset if split == "train" else eval_dataset
    
    # Extract problems and answers
    problems = []
    answers = []
    
    for item in dataset:
        # Extract problem
        problem = item.get("problem", item.get("question", item.get("prompt", "")))
        
        # Extract answer
        answer = item.get("answer", None)
        if answer is None:
            # Try to extract from solution
            solution = item.get("solution", item.get("response", ""))
            answer = extract_final_answer(solution)
        
        # Add to lists if both are available
        if problem and answer:
            problems.append(problem)
            answers.append(answer)
    
    logging.info(f"Loaded {len(problems)} problems with answers")
    
    return problems, answers


def evaluate_model(args):
    """Evaluate the model on a dataset.
    
    Args:
        args: Command line arguments
    """
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"eval_{timestamp}.log" if args.output_file is None else f"{os.path.splitext(args.output_file)[0]}.log"
    setup_logging(log_file=log_file)
    
    # Log arguments
    logging.info(f"Arguments: {vars(args)}")
    
    # Create model
    logging.info(f"Loading model from {args.model_path}")
    model = create_model(
        model_type=args.model_type,
        model_name_or_path=args.model_path,
        precision=args.precision
    )
    
    # Parse boolean arguments
    use_verification = args.use_verification.lower() == "true"
    use_ensemble = args.use_ensemble.lower() == "true"
    
    # Create inference pipeline
    logging.info("Creating inference pipeline")
    pipeline = MathInferencePipeline(
        base_model=model,
        prm_model_path=args.prm_model_path,
        verifier_model_path=args.verifier_model_path,
        num_agents=args.num_agents,
        max_attempts=args.max_attempts,
        temperature_range=(args.temperature_min, args.temperature_max),
        top_p_range=(args.top_p_min, args.top_p_max),
        max_new_tokens=args.max_new_tokens,
        use_verification=use_verification,
        use_ensemble=use_ensemble,
        timeout=args.timeout
    )
    
    # Load problems and answers
    problems, answers = load_problems_and_answers(
        dataset_path=args.dataset,
        split=args.split,
        max_samples=args.max_samples
    )
    
    # Evaluate on dataset
    logging.info(f"Evaluating on {len(problems)} problems")
    results = pipeline.evaluate_on_dataset(
        problems=problems,
        reference_answers=answers,
        output_file=args.output_file
    )
    
    # Log results
    logging.info(f"Evaluation results: {json.dumps(results, indent=2)}")
    
    # Print summary
    print("\n===== Evaluation Summary =====")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Number of problems: {results['num_problems']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Verification rate: {results['verification_rate']:.4f}")
    print(f"Answer rate: {results['answer_rate']:.4f}")
    print(f"Average time: {results['avg_time']:.2f}s")
    print(f"Verification precision: {results['verification_precision']:.4f}")
    print(f"Verification recall: {results['verification_recall']:.4f}")
    print(f"Verification F1: {results['verification_f1']:.4f}")
    print("=============================")
    
    # Return results
    return results


def main():
    """Main function."""
    args = parse_args()
    evaluate_model(args)


if __name__ == "__main__":
    main()
