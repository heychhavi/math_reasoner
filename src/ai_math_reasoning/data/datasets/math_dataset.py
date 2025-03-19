"""Mathematical reasoning datasets for training and evaluation."""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets


class MathReasoningDataset(Dataset):
    """Base dataset for mathematical reasoning.
    
    This dataset provides a common interface for different mathematical
    reasoning datasets to be used for training and evaluation.
    """
    
    def __init__(
        self,
        problems: List[str],
        solutions: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        problem_prefix: str = "Problem: ",
        solution_prefix: str = "Solution: ",
        tokenizer = None,
        max_length: int = 2048,
        is_eval: bool = False
    ):
        """Initialize a mathematical reasoning dataset.
        
        Args:
            problems: List of mathematical problems
            solutions: List of solutions (optional)
            answers: List of answers (optional)
            problem_prefix: Prefix for problems
            solution_prefix: Prefix for solutions
            tokenizer: Tokenizer for encoding data
            max_length: Maximum sequence length
            is_eval: Whether this is an evaluation dataset
        """
        self.problems = problems
        self.solutions = solutions
        self.answers = answers
        self.problem_prefix = problem_prefix
        self.solution_prefix = solution_prefix
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_eval = is_eval
        
        # Validate data
        if solutions is not None and len(problems) != len(solutions):
            raise ValueError(f"Number of problems ({len(problems)}) and solutions ({len(solutions)}) must match")
        
        if answers is not None and len(problems) != len(answers):
            raise ValueError(f"Number of problems ({len(problems)}) and answers ({len(answers)}) must match")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.problems)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item at index.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary with item data
        """
        problem = self.problems[idx]
        
        # Create item dictionary
        item = {
            "problem": problem,
            "problem_with_prefix": f"{self.problem_prefix}{problem}",
        }
        
        # Add solution if available
        if self.solutions is not None:
            solution = self.solutions[idx]
            item["solution"] = solution
            item["solution_with_prefix"] = f"{self.solution_prefix}{solution}"
        
        # Add answer if available
        if self.answers is not None:
            item["answer"] = self.answers[idx]
        
        # If tokenizer is provided, tokenize the data
        if self.tokenizer is not None:
            # Tokenize problem
            problem_with_prefix = item["problem_with_prefix"]
            
            # For evaluation, we only need the input_ids and attention_mask
            if self.is_eval:
                tokenized_problem = self.tokenizer(
                    problem_with_prefix,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                item["input_ids"] = tokenized_problem["input_ids"].squeeze(0)
                item["attention_mask"] = tokenized_problem["attention_mask"].squeeze(0)
            
            # For training, we need input_ids, attention_mask, and labels
            elif self.solutions is not None:
                solution_with_prefix = item["solution_with_prefix"]
                full_text = f"{problem_with_prefix} {solution_with_prefix}"
                
                tokenized_full = self.tokenizer(
                    full_text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                item["input_ids"] = tokenized_full["input_ids"].squeeze(0)
                item["attention_mask"] = tokenized_full["attention_mask"].squeeze(0)
                item["labels"] = tokenized_full["input_ids"].squeeze(0).clone()
                
                # For supervised fine-tuning, we mask out the loss for prompt tokens
                if "problem_length" not in item:
                    tokenized_problem = self.tokenizer(
                        problem_with_prefix,
                        return_tensors="pt"
                    )
                    problem_length = tokenized_problem["input_ids"].size(1)
                    item["problem_length"] = problem_length
                
                # Set labels for prompt tokens to -100 (ignored in loss computation)
                problem_length = item["problem_length"]
                item["labels"][:problem_length] = -100
        
        return item
    
    def to_prompt_solution_pairs(self) -> List[Dict[str, str]]:
        """Convert dataset to list of prompt/solution pairs.
        
        Returns:
            List of dictionaries with prompt and solution
        """
        pairs = []
        
        for i in range(len(self)):
            item = self[i]
            
            # Skip items without solutions
            if self.solutions is None or i >= len(self.solutions):
                continue
            
            pairs.append({
                "prompt": item["problem_with_prefix"],
                "solution": self.solutions[i]
            })
        
        return pairs
    
    def to_grpo_training_data(self) -> Dict[str, List]:
        """Convert dataset to GRPO training data format.
        
        Returns:
            Dictionary with problems and reference answers
        """
        return {
            "problems": self.problems,
            "answers": self.answers if self.answers is not None else [None] * len(self.problems)
        }


class HuggingFaceMathDataset(MathReasoningDataset):
    """Mathematical reasoning dataset loaded from Hugging Face.
    
    This class extends MathReasoningDataset to load data from
    Hugging Face datasets.
    """
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        problem_key: str = "question",
        solution_key: Optional[str] = "response",
        answer_key: Optional[str] = "answer",
        system_key: Optional[str] = "system",
        conversation_key: Optional[str] = "conversations",
        problem_prefix: str = "Problem: ",
        solution_prefix: str = "Solution: ",
        tokenizer = None,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
        is_eval: bool = False,
        seed: int = 42
    ):
        """Initialize a Hugging Face mathematical reasoning dataset.
        
        Args:
            dataset_path: Path to Hugging Face dataset
            split: Dataset split
            problem_key: Key for problems in the dataset
            solution_key: Key for solutions in the dataset (optional)
            answer_key: Key for answers in the dataset (optional)
            system_key: Key for system message in chat datasets (optional)
            conversation_key: Key for conversations in chat datasets (optional)
            problem_prefix: Prefix for problems
            solution_prefix: Prefix for solutions
            tokenizer: Tokenizer for encoding data
            max_length: Maximum sequence length
            max_samples: Maximum number of samples to load
            is_eval: Whether this is an evaluation dataset
            seed: Random seed for shuffling
        """
        # Load dataset from Hugging Face
        try:
            dataset = load_dataset(dataset_path, split=split)
            logging.info(f"Loaded {len(dataset)} examples from {dataset_path} ({split})")
            
            # Shuffle dataset
            dataset = dataset.shuffle(seed=seed)
            
            # Limit number of samples if requested
            if max_samples is not None:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
                logging.info(f"Using {len(dataset)} examples after limiting")
            
            # Extract data
            problems = []
            solutions = [] if solution_key or conversation_key else None
            answers = [] if answer_key else None
            
            # Handle regular datasets
            if not conversation_key:
                for example in dataset:
                    # Extract problem
                    problem = example.get(problem_key, "")
                    problems.append(problem)
                    
                    # Extract solution
                    if solution_key and solution_key in example:
                        solutions.append(example[solution_key])
                    
                    # Extract answer
                    if answer_key and answer_key in example:
                        answers.append(example[answer_key])
            
            # Handle conversation datasets (like Bespoke-Stratos-17k)
            else:
                for example in dataset:
                    # Check if example has conversations
                    if conversation_key not in example:
                        continue
                    
                    # Extract system message if available
                    system = example.get(system_key, "")
                    
                    # Process conversations
                    problem = ""
                    solution = ""
                    
                    for turn in example[conversation_key]:
                        role = turn.get("from", "")
                        content = turn.get("value", "")
                        
                        if role.lower() in ["human", "user"]:
                            # This is a problem
                            problem += content + " "
                        elif role.lower() in ["assistant", "gpt"]:
                            # This is a solution
                            solution += content + " "
                    
                    # Add system message if available
                    if system:
                        problem = system + " " + problem
                    
                    # Add to lists
                    problems.append(problem.strip())
                    solutions.append(solution.strip())
                    
                    # Extract answer if available
                    if answer_key and answer_key in example:
                        answers.append(example[answer_key])
                    elif solution:
                        # Try to extract answer from solution
                        from ai_math_reasoning.data.processing.math_processing import extract_final_answer
                        answer = extract_final_answer(solution)
                        if answers is not None:
                            answers.append(answer)
            
            # Initialize base class
            super().__init__(
                problems=problems,
                solutions=solutions,
                answers=answers,
                problem_prefix=problem_prefix,
                solution_prefix=solution_prefix,
                tokenizer=tokenizer,
                max_length=max_length,
                is_eval=is_eval
            )
            
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_path} ({split}): {str(e)}")
            raise


class JSONMathDataset(MathReasoningDataset):
    """Mathematical reasoning dataset loaded from JSON file.
    
    This class extends MathReasoningDataset to load data from
    JSON files with customizable key mapping.
    """
    
    def __init__(
        self,
        file_path: str,
        problem_key: str = "problem",
        solution_key: Optional[str] = "solution",
        answer_key: Optional[str] = "answer",
        problem_prefix: str = "Problem: ",
        solution_prefix: str = "Solution: ",
        tokenizer = None,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
        is_eval: bool = False,
        seed: int = 42
    ):
        """Initialize a JSON mathematical reasoning dataset.
        
        Args:
            file_path: Path to JSON file
            problem_key: Key for problems in the JSON
            solution_key: Key for solutions in the JSON (optional)
            answer_key: Key for answers in the JSON (optional)
            problem_prefix: Prefix for problems
            solution_prefix: Prefix for solutions
            tokenizer: Tokenizer for encoding data
            max_length: Maximum sequence length
            max_samples: Maximum number of samples to load
            is_eval: Whether this is an evaluation dataset
            seed: Random seed for shuffling
        """
        # Load data from JSON file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            logging.info(f"Loaded {len(data)} examples from {file_path}")
            
            # Ensure data is a list
            if not isinstance(data, list):
                raise ValueError(f"JSON data must be a list, got {type(data)}")
            
            # Shuffle data
            import random
            random.seed(seed)
            random.shuffle(data)
            
            # Limit number of samples if requested
            if max_samples is not None:
                data = data[:min(max_samples, len(data))]
                logging.info(f"Using {len(data)} examples after limiting")
            
            # Extract data
            problems = []
            solutions = [] if solution_key else None
            answers = [] if answer_key else None
            
            for example in data:
                # Extract problem
                if problem_key in example:
                    problems.append(example[problem_key])
                else:
                    logging.warning(f"Skipping example, missing problem key: {problem_key}")
                    continue
                
                # Extract solution
                if solution_key and solution_key in example:
                    solutions.append(example[solution_key])
                
                # Extract answer
                if answer_key and answer_key in example:
                    answers.append(example[answer_key])
            
            # Initialize base class
            super().__init__(
                problems=problems,
                solutions=solutions,
                answers=answers,
                problem_prefix=problem_prefix,
                solution_prefix=solution_prefix,
                tokenizer=tokenizer,
                max_length=max_length,
                is_eval=is_eval
            )
            
        except Exception as e:
            logging.error(f"Failed to load dataset from {file_path}: {str(e)}")
            raise


class PrefRankMathDataset(Dataset):
    """Preference ranking dataset for mathematical reasoning.
    
    This dataset provides paired solutions for preference-based training,
    such as for Preference Reward Modeling (PRM).
    """
    
    def __init__(
        self,
        problems: List[str],
        better_solutions: List[str],
        worse_solutions: List[str],
        problem_prefix: str = "Problem: ",
        solution_prefix: str = "Solution: ",
        tokenizer = None,
        max_length: int = 2048
    ):
        """Initialize a preference ranking dataset.
        
        Args:
            problems: List of mathematical problems
            better_solutions: List of preferred solutions
            worse_solutions: List of less preferred solutions
            problem_prefix: Prefix for problems
            solution_prefix: Prefix for solutions
            tokenizer: Tokenizer for encoding data
            max_length: Maximum sequence length
        """
        if len(problems) != len(better_solutions) or len(problems) != len(worse_solutions):
            raise ValueError("Number of problems, better solutions, and worse solutions must match")
        
        self.problems = problems
        self.better_solutions = better_solutions
        self.worse_solutions = worse_solutions
        self.problem_prefix = problem_prefix
        self.solution_prefix = solution_prefix
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.problems)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item at index.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary with item data
        """
        problem = self.problems[idx]
        better_solution = self.better_solutions[idx]
        worse_solution = self.worse_solutions[idx]
        
        # Format with prefixes
        problem_with_prefix = f"{self.problem_prefix}{problem}"
        better_with_prefix = f"{self.solution_prefix}{better_solution}"
        worse_with_prefix = f"{self.solution_prefix}{worse_solution}"
        
        # Create item dictionary
        item = {
            "problem": problem,
            "problem_with_prefix": problem_with_prefix,
            "better_solution": better_solution,
            "worse_solution": worse_solution,
            "better_with_prefix": better_with_prefix,
            "worse_with_prefix": worse_with_prefix,
        }
        
        # If tokenizer is provided, tokenize the data
        if self.tokenizer is not None:
            # Tokenize problem
            tokenized_problem = self.tokenizer(
                problem_with_prefix,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenize better solution
            better_input = f"{problem_with_prefix} {better_with_prefix}"
            tokenized_better = self.tokenizer(
                better_input,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenize worse solution
            worse_input = f"{problem_with_prefix} {worse_with_prefix}"
            tokenized_worse = self.tokenizer(
                worse_input,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Add tokenized data to item
            item["problem_input_ids"] = tokenized_problem["input_ids"].squeeze(0)
            item["problem_attention_mask"] = tokenized_problem["attention_mask"].squeeze(0)
            
            item["better_input_ids"] = tokenized_better["input_ids"].squeeze(0)
            item["better_attention_mask"] = tokenized_better["attention_mask"].squeeze(0)
            
            item["worse_input_ids"] = tokenized_worse["input_ids"].squeeze(0)
            item["worse_attention_mask"] = tokenized_worse["attention_mask"].squeeze(0)
        
        return item
    
    @classmethod
    def from_json(
        cls,
        file_path: str,
        problem_key: str = "problem",
        better_key: str = "better_solution",
        worse_key: str = "worse_solution",
        **kwargs
    ):
        """Create dataset from JSON file.
        
        Args:
            file_path: Path to JSON file
            problem_key: Key for problems in the JSON
            better_key: Key for better solutions in the JSON
            worse_key: Key for worse solutions in the JSON
            **kwargs: Additional arguments for the constructor
            
        Returns:
            PrefRankMathDataset instance
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract data
            problems = []
            better_solutions = []
            worse_solutions = []
            
            for example in data:
                if problem_key in example and better_key in example and worse_key in example:
                    problems.append(example[problem_key])
                    better_solutions.append(example[better_key])
                    worse_solutions.append(example[worse_key])
            
            return cls(
                problems=problems,
                better_solutions=better_solutions,
                worse_solutions=worse_solutions,
                **kwargs
            )
            
        except Exception as e:
            logging.error(f"Failed to load dataset from {file_path}: {str(e)}")
            raise
    
    @classmethod
    def from_solution_pairs(
        cls,
        file_paths: List[str],
        problem_key: str = "problem",
        solutions_key: str = "solutions",
        ranking_key: str = "ranking",
        **kwargs
    ):
        """Create dataset from files with solution pairs and rankings.
        
        Args:
            file_paths: List of paths to JSON files
            problem_key: Key for problems in the JSON
            solutions_key: Key for solutions list in the JSON
            ranking_key: Key for rankings in the JSON
            **kwargs: Additional arguments for the constructor
            
        Returns:
            PrefRankMathDataset instance
        """
        problems = []
        better_solutions = []
        worse_solutions = []
        
        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                for example in data:
                    if problem_key not in example or solutions_key not in example:
                        continue
                    
                    # Get problem and solutions
                    problem = example[problem_key]
                    solutions = example[solutions_key]
                    
                    # Skip if not enough solutions
                    if len(solutions) < 2:
                        continue
                    
                    # Get rankings if available
                    if ranking_key in example:
                        rankings = example[ranking_key]
                        
                        # Pair solutions based on rankings
                        for i in range(len(rankings)):
                            for j in range(i + 1, len(rankings)):
                                if rankings[i] > rankings[j]:
                                    # j is better than i
                                    problems.append(problem)
                                    better_solutions.append(solutions[j])
                                    worse_solutions.append(solutions[i])
                                elif rankings[i] < rankings[j]:
                                    # i is better than j
                                    problems.append(problem)
                                    better_solutions.append(solutions[i])
                                    worse_solutions.append(solutions[j])
                    else:
                        # No rankings, use all pairs
                        for i in range(len(solutions)):
                            for j in range(i + 1, len(solutions)):
                                # Arbitrarily choose i as better
                                problems.append(problem)
                                better_solutions.append(solutions[i])
                                worse_solutions.append(solutions[j])
                
            except Exception as e:
                logging.error(f"Failed to load dataset from {file_path}: {str(e)}")
        
        return cls(
            problems=problems,
            better_solutions=better_solutions,
            worse_solutions=worse_solutions,
            **kwargs
        )
