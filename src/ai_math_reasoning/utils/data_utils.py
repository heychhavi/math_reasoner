"""Utilities for data manipulation and loading."""

import os
import json
import logging
import random
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import numpy as np
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from ai_math_reasoning.data.datasets.math_dataset import (
    MathReasoningDataset,
    HuggingFaceMathDataset,
    PairedPreferenceDataset,
    create_prm_dataset
)


def load_math_dataset(
    dataset_path: str,
    split: str = "train",
    tokenizer = None,
    problem_key: str = "problem",
    solution_key: str = "solution",
    answer_key: Optional[str] = "answer",
    max_length: int = 2048,
    max_samples: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None
) -> MathReasoningDataset:
    """Load a mathematical reasoning dataset.
    
    Args:
        dataset_path: Path to the dataset
        split: Dataset split
        tokenizer: Tokenizer for tokenization (optional)
        problem_key: Key for problems in the dataset
        solution_key: Key for solutions in the dataset
        answer_key: Key for answers in the dataset (optional)
        max_length: Maximum sequence length
        max_samples: Maximum number of samples to load (optional)
        seed: Random seed for sampling
        cache_dir: Directory to cache the dataset (optional)
        
    Returns:
        Math reasoning dataset
    """
    # Check if dataset is a local file
    if os.path.exists(dataset_path):
        # Load from local file
        logging.info(f"Loading dataset from local file: {dataset_path}")
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract problems and solutions
        problems = []
        solutions = []
        answers = []
        
        for item in data:
            # Extract fields
            problem = item.get(problem_key)
            solution = item.get(solution_key)
            answer = item.get(answer_key) if answer_key is not None else None
            
            # Skip invalid items
            if not problem or not solution:
                continue
            
            # Add to lists
            problems.append(problem)
            solutions.append(solution)
            if answer is not None:
                answers.append(answer)
        
        # Sample if requested
        if max_samples is not None and max_samples < len(problems):
            # Set seed for reproducibility
            random.seed(seed)
            
            # Sample indices
            indices = random.sample(range(len(problems)), max_samples)
            
            # Subset data
            problems = [problems[i] for i in indices]
            solutions = [solutions[i] for i in indices]
            answers = [answers[i] for i in indices] if answers else None
        
        # Create dataset
        return MathReasoningDataset(
            problems=problems,
            solutions=solutions,
            answers=answers,
            tokenizer=tokenizer,
            max_length=max_length
        )
    else:
        # Load from HuggingFace
        logging.info(f"Loading dataset from HuggingFace: {dataset_path}")
        
        return HuggingFaceMathDataset(
            dataset_path=dataset_path,
            split=split,
            tokenizer=tokenizer,
            problem_key=problem_key,
            solution_key=solution_key,
            answer_key=answer_key,
            max_length=max_length,
            max_samples=max_samples,
            seed=seed
        )


def load_prm_dataset(
    dataset_path: str,
    tokenizer = None,
    max_length: int = 2048,
    problem_key: str = "problem",
    better_solution_key: str = "better_solution",
    worse_solution_key: str = "worse_solution",
    max_samples: Optional[int] = None,
    seed: int = 42
) -> PairedPreferenceDataset:
    """Load a paired preference dataset for PRM training.
    
    Args:
        dataset_path: Path to the paired preference dataset
        tokenizer: Tokenizer for tokenization (optional)
        max_length: Maximum sequence length
        problem_key: Key for problems in the dataset
        better_solution_key: Key for better solutions in the dataset
        worse_solution_key: Key for worse solutions in the dataset
        max_samples: Maximum number of samples to load (optional)
        seed: Random seed for sampling
        
    Returns:
        Paired preference dataset
    """
    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract fields
    problems = []
    better_solutions = []
    worse_solutions = []
    
    for item in data:
        # Extract fields
        problem = item.get(problem_key)
        better_solution = item.get(better_solution_key)
        worse_solution = item.get(worse_solution_key)
        
        # Skip invalid items
        if not problem or not better_solution or not worse_solution:
            continue
        
        # Add to lists
        problems.append(problem)
        better_solutions.append(better_solution)
        worse_solutions.append(worse_solution)
    
    # Sample if requested
    if max_samples is not None and max_samples < len(problems):
        # Set seed for reproducibility
        random.seed(seed)
        
        # Sample indices
        indices = random.sample(range(len(problems)), max_samples)
        
        # Subset data
        problems = [problems[i] for i in indices]
        better_solutions = [better_solutions[i] for i in indices]
        worse_solutions = [worse_solutions[i] for i in indices]
    
    # Create dataset
    return PairedPreferenceDataset(
        problems=problems,
        better_solutions=better_solutions,
        worse_solutions=worse_solutions,
        tokenizer=tokenizer,
        max_length=max_length
    )


def create_dataloaders(
    train_dataset,
    eval_dataset = None,
    batch_size: int = 8,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create data loaders for training and evaluation.
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation (optional)
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        Tuple of (train_loader, eval_loader)
    """
    # Set up evaluation batch size
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    # Create training data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Create evaluation data loader if dataset provided
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return train_loader, eval_loader


def create_training_splits(
    dataset,
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """Split a dataset into training and evaluation sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio of training samples
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Get dataset size
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    
    # Create splits
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]
    
    # Create datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    eval_dataset = torch.utils.data.Subset(dataset, eval_indices)
    
    return train_dataset, eval_dataset


def prepare_math_problem_for_model(
    problem: str,
    template: Optional[str] = None
) -> str:
    """Prepare a mathematical problem for input to a model.
    
    Args:
        problem: Problem statement
        template: Template for formatting (optional)
        
    Returns:
        Formatted problem
    """
    if template is None:
        template = (
            "You are a mathematical problem-solving assistant who is very careful and thorough. "
            "Solve the following problem step by step, showing your reasoning. "
            "End your solution with 'Answer: [your final answer]'.\n\n"
            "Problem: {problem}\n\n"
            "Solution:"
        )
    
    return template.format(problem=problem)


def prepare_batch_for_distillation(
    batch: Dict[str, torch.Tensor],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048
) -> Dict[str, torch.Tensor]:
    """Prepare a batch for distillation training.
    
    Args:
        batch: Batch of data
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Prepared batch
    """
    # Extract problem-solution pairs
    problems = batch["problem"]
    solutions = batch["solution"]
    
    # Combine problems and solutions
    combined_texts = [
        f"Problem: {problem}\n\nSolution: {solution}"
        for problem, solution in zip(problems, solutions)
    ]
    
    # Tokenize
    encodings = tokenizer(
        combined_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Prepare labels for language modeling
    input_ids = encodings["input_ids"].clone()
    labels = input_ids.clone()
    
    # Mask problem tokens in labels
    for i, (problem, solution) in enumerate(zip(problems, solutions)):
        problem_encoding = tokenizer(
            f"Problem: {problem}\n\nSolution:",
            add_special_tokens=False
        )
        problem_length = len(problem_encoding["input_ids"])
        
        # Mask problem tokens with -100
        labels[i, :problem_length] = -100
    
    # Create batch
    prepared_batch = {
        "input_ids": input_ids,
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    }
    
    return prepared_batch
