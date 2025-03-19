#!/usr/bin/env python
"""
Benchmark mathematical reasoning models on AIME problems.

This script evaluates mathematical reasoning models on American Invitational
Mathematics Examination (AIME) problems, a challenging set of high school
level math competitions.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add parent directory to path to import from ai_math_reasoning
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_math_reasoning.models import create_model
from ai_math_reasoning.inference.pipeline import MathInferencePipeline
from ai_math_reasoning.utils.training_utils import setup_logging, set_seed
from ai_math_reasoning.data.processing.math_processing import (
    extract_final_answer,
    compare_answers
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark LLMs on AIME problems")
    
    # Model parameters
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the model or model identifier"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="deepseek",
        choices=["deepseek", "qwen", "transformer"],
        help="Type of model to use"
    )
    parser.add_argument(
        "--load_8bit", 
        action="store_true",
        help="Whether to load the model in 8-bit precision"
    )
    parser.add_argument(
        "--use_peft", 
        action="store_true",
        help="Whether to load a PEFT/LoRA adapter"
    )
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        default=None,
        help="Path to the PEFT/LoRA adapter"
    )
    parser.add_argument(
        "--precision", 
        type=str, 
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Precision to use for model weights"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="evalchemy/AIME24",
        help="Path to the AIME dataset"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for evaluation"
    )
    
    # Multi-agent parameters
    parser.add_argument(
        "--num_agents", 
        type=int, 
        default=3,
        help="Number of agents to use"
    )
    parser.add_argument(
        "--use_verification", 
        action="store_true",
        help="Whether to use solution verification"
    )
    parser.add_argument(
        "--use_ensemble", 
        action="store_true",
        help="Whether to use ensemble solution selection"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.95,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=4096,
        help="Maximum number of new tokens to generate"
    )
    
    # Other parameters
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def load_model_with_adapter(args):
    """Load a model with optional adapter.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Loaded model and tokenizer
    """
    logging.info(f"Loading base model: {args.model_path}")
    
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if args.precision == "bf16" else 
                        torch.float16 if args.precision == "fp16" else torch.float32,
        "trust_remote_code": True
    }
    
    if args.load_8bit:
        model_kwargs["load_in_8bit"] = True
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = "</s>"
    
    # Load adapter if requested
    if args.use_peft and args.adapter_path:
        logging.info(f"Loading adapter: {args.adapter_path}")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter_path)
            logging.info("Successfully loaded PEFT adapter")
        except Exception as e:
            logging.error(f"Failed to load adapter: {str(e)}")
    
    return model, tokenizer


def load_aime_benchmark(dataset_path: str) -> List[Dict[str, Any]]:
    """Load AIME benchmark problems.
    
    Args:
        dataset_path: Path to the AIME dataset
        
    Returns:
        List of problems with their answers
    """
    logging.info(f"Loading AIME benchmark from: {dataset_path}")
    
    try:
        from evalchemy.tasks.aime.aime import AIME24
        eval_dataset = AIME24(dataset_path)
        
        problems = []
        for idx in range(len(eval_dataset)):
            item = eval_dataset[idx]
            problems.append({
                "input": item["input"],
                "target": item["target"],
                "id": f"aime_{idx}"
            })
        
        logging.info(f"Loaded {len(problems)} AIME problems")
        return problems
    except ImportError:
        logging.error("Failed to import evalchemy. Attempting to load directly.")
        
        # Try to load as direct JSON file
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                problems = json.load(f)
            logging.info(f"Loaded {len(problems)} AIME problems from file")
            return problems
        else:
            raise ValueError(f"Dataset path does not exist: {dataset_path}")


def run_aime_benchmark(args):
    """Run the AIME benchmark.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Benchmark results
    """
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "benchmark_aime.log"))
    
    # Load model
    if args.use_peft:
        # Use manual loading for PEFT models
        model, tokenizer = load_model_with_adapter(args)
        from ai_math_reasoning.models.base_model import TransformerModel
        wrapped_model = TransformerModel(
            model=model,
            tokenizer=tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
            precision=args.precision
        )
    else:
        # Use the factory function for standard models
        wrapped_model = create_model(
            model_type=args.model_type,
            model_name_or_path=args.model_path,
            precision=args.precision
        )
    
    # Create inference pipeline
    pipeline = MathInferencePipeline(
        base_model=wrapped_model,
        num_agents=args.num_agents,
        max_attempts=1,
        temperature_range=(args.temperature, args.temperature),
        top_p_range=(args.top_p, args.top_p),
        max_new_tokens=args.max_new_tokens,
        use_verification=args.use_verification,
        use_ensemble=args.use_ensemble
    )
    
    # Load AIME problems
    problems = load_aime_benchmark(args.dataset_path)
    
    # Run evaluation
    results = []
    correct_count = 0
    
    for i, problem in enumerate(tqdm(problems, desc="Evaluating")):
        problem_text = problem["input"]
        reference_answer = problem["target"]
        problem_id = problem.get("id", f"problem_{i}")
        
        logging.info(f"Processing problem {i+1}/{len(problems)}: {problem_id}")
        
        try:
            # Solve the problem
            result = pipeline.solve(problem_text, reference_answer)
            
            # Check correctness
            is_correct = result.get("is_correct", False)
            if is_correct:
                correct_count += 1
                logging.info(f"✓ Problem {problem_id} solved correctly")
            else:
                logging.info(f"✗ Problem {problem_id} solved incorrectly")
                logging.info(f"  Expected: {reference_answer}")
                logging.info(f"  Got: {result.get('answer', 'No answer')}")
            
            # Add to results
            result["problem_id"] = problem_id
            results.append(result)
            
        except Exception as e:
            logging.error(f"Error processing problem {problem_id}: {str(e)}")
            results.append({
                "problem_id": problem_id,
                "problem": problem_text,
                "reference_answer": reference_answer,
                "error": str(e),
                "is_correct": False
            })
    
    # Calculate accuracy
    accuracy = correct_count / len(problems)
    
    # Compile results
    benchmark_results = {
        "model": args.model_path,
        "adapter": args.adapter_path if args.use_peft else None,
        "num_problems": len(problems),
        "num_correct": correct_count,
        "accuracy": accuracy,
        "results": results
    }
    
    # Save results
    results_path = os.path.join(args.output_dir, f"aime_results_{args.model_type}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    
    logging.info(f"Saved results to {results_path}")
    
    # Log summary
    logging.info("\n===== AIME Benchmark Summary =====")
    logging.info(f"Model: {args.model_path}")
    if args.use_peft:
        logging.info(f"Adapter: {args.adapter_path}")
    logging.info(f"Number of problems: {len(problems)}")
    logging.info(f"Number correct: {correct_count}")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info("================================")
    
    return benchmark_results


def main():
    """Main function."""
    args = parse_args()
    run_aime_benchmark(args)


if __name__ == "__main__":
    main()
