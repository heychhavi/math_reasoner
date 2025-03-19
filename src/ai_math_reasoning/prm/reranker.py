"""Preference Reward Model (PRM) for ranking mathematical solutions."""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from ai_math_reasoning.models import MathReasoningModel
from ai_math_reasoning.data.processing.math_processing import extract_final_answer, analyze_solution_steps


class PRMReranker:
    """Preference Reward Model for ranking mathematical solutions.
    
    The PRM Reranker is used to score solutions based on quality and correctness,
    and can be used to rank multiple solution attempts for the same problem.
    """
    
    def __init__(
        self,
        base_model: MathReasoningModel,
        score_template: Optional[str] = None,
        inference_only: bool = False,
        beta: float = 0.1,
        learning_rate: float = 5e-6,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        use_amp: bool = False
    ):
        """Initialize a PRM reranker.
        
        Args:
            base_model: Base language model for scoring
            score_template: Template for scoring prompts (optional)
            inference_only: Whether to use for inference only
            beta: Regularization coefficient
            learning_rate: Learning rate
            weight_decay: Weight decay
            max_grad_norm: Maximum gradient norm
            use_amp: Whether to use automatic mixed precision
        """
        self.model = base_model
        self.device = base_model.device
        self.inference_only = inference_only
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        
        # Default scoring template
        self.score_template = score_template or (
            "Rate the quality of the following solution to a mathematical problem "
            "on a scale from 0 to 10, where 0 is completely incorrect and 10 is perfect.\n\n"
            "Problem: {problem}\n\n"
            "Solution: {solution}\n\n"
            "Rating:"
        )
        
        # Initialize optimizer if not inference only
        if not inference_only:
            # Create optimizer
            self.optimizer = AdamW(
                params=[p for p in self.model.model.parameters() if p.requires_grad],
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            # Create scaler for AMP
            self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    def score_solution(
        self,
        problem: str,
        solution: str,
        extract_score: bool = True
    ) -> Union[float, str]:
        """Score a solution to a problem.
        
        Args:
            problem: Problem text
            solution: Solution text
            extract_score: Whether to extract numerical score
            
        Returns:
            Score (0-10) or raw model output
        """
        # Format prompt
        prompt = self.score_template.format(
            problem=problem,
            solution=solution
        )
        
        # Generate score
        score_text = self.model.generate(
            prompt=prompt,
            max_new_tokens=20,
            temperature=0.0,  # Use deterministic generation for scoring
            do_sample=False
        )
        
        # Extract numerical score if requested
        if extract_score:
            try:
                # Try to extract a number from the score text
                for token in score_text.split():
                    try:
                        score = float(token)
                        if 0 <= score <= 10:
                            return score
                    except ValueError:
                        continue
                
                # If no valid score found, try other strategies
                # Look for numbers in the text with regex
                import re
                numbers = re.findall(r'\b\d+(?:\.\d+)?\b', score_text)
                if numbers:
                    score = float(numbers[0])
                    if 0 <= score <= 10:
                        return score
                
                # Default to mid-range score if extraction fails
                logging.warning(f"Failed to extract score from: {score_text}")
                return 5.0
                
            except Exception as e:
                logging.error(f"Error extracting score: {str(e)}")
                return 5.0
        else:
            return score_text
    
    def rank_solutions(
        self,
        problem: str,
        solutions: List[str],
        normalize: bool = True,
        add_analysis: bool = True
    ) -> List[Dict[str, Any]]:
        """Rank a list of solutions to a problem.
        
        Args:
            problem: Problem text
            solutions: List of solution texts
            normalize: Whether to normalize scores
            add_analysis: Whether to add solution analysis
            
        Returns:
            List of dictionaries with solutions and scores, sorted by score
        """
        # Score each solution
        scored_solutions = []
        for i, solution in enumerate(solutions):
            # Score the solution
            score = self.score_solution(problem, solution)
            
            # Create result dictionary
            result = {
                "solution": solution,
                "score": score,
                "rank": 0  # Will be filled in later
            }
            
            # Extract answer if possible
            answer = extract_final_answer(solution)
            if answer:
                result["answer"] = answer
            
            # Add analysis if requested
            if add_analysis:
                analysis = analyze_solution_steps(solution)
                result["analysis"] = analysis
            
            scored_solutions.append(result)
        
        # Sort by score (descending)
        scored_solutions.sort(key=lambda x: x["score"], reverse=True)
        
        # Assign ranks
        for i, solution in enumerate(scored_solutions):
            solution["rank"] = i + 1
        
        # Normalize scores if requested
        if normalize and scored_solutions:
            min_score = min(s["score"] for s in scored_solutions)
            max_score = max(s["score"] for s in scored_solutions)
            score_range = max_score - min_score
            
            if score_range > 0:
                for solution in scored_solutions:
                    solution["normalized_score"] = (solution["score"] - min_score) / score_range
            else:
                # If all scores are the same, assign 1.0 to the first and 0.0 to the rest
                for i, solution in enumerate(scored_solutions):
                    solution["normalized_score"] = 1.0 if i == 0 else 0.0
        
        return scored_solutions
    
    def compute_loss(
        self,
        problem: str,
        better_solution: str,
        worse_solution: str,
        margin: float = 0.0
    ) -> torch.Tensor:
        """Compute loss for preference learning.
        
        Args:
            problem: Problem text
            better_solution: Better (preferred) solution
            worse_solution: Worse solution
            margin: Margin for ranking loss
            
        Returns:
            Loss tensor
        """
        if self.inference_only:
            raise ValueError("Cannot compute loss in inference-only mode")
        
        # Format prompts
        better_prompt = self.score_template.format(
            problem=problem,
            solution=better_solution
        )
        worse_prompt = self.score_template.format(
            problem=problem,
            solution=worse_solution
        )
        
        # Get logits
        better_logits = self.model.get_logits(better_prompt)
        worse_logits = self.model.get_logits(worse_prompt)
        
        # Compute scores (use last token logits)
        better_score = better_logits[0, -1, :].mean()
        worse_score = worse_logits[0, -1, :].mean()
        
        # Compute ranking loss with margin
        ranking_loss = F.relu(worse_score - better_score + margin)
        
        # Add regularization
        reg_loss = self.beta * (better_score**2 + worse_score**2)
        
        # Total loss
        loss = ranking_loss + reg_loss
        
        return loss
    
    def train_on_batch(
        self,
        problems: List[str],
        better_solutions: List[str],
        worse_solutions: List[str],
        margin: float = 0.0
    ) -> Dict[str, float]:
        """Train PRM on a batch of preference pairs.
        
        Args:
            problems: List of problem texts
            better_solutions: List of better solutions
            worse_solutions: List of worse solutions
            margin: Margin for ranking loss
            
        Returns:
            Dictionary with training metrics
        """
        if self.inference_only:
            raise ValueError("Cannot train in inference-only mode")
        
        # Ensure model is in training mode
        self.model.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Track metrics
        batch_loss = 0.0
        batch_ranking_loss = 0.0
        batch_reg_loss = 0.0
        batch_accuracy = 0.0
        
        # Process each example in batch
        with torch.cuda.amp.autocast() if self.use_amp else torch.no_grad():
            for problem, better, worse in zip(problems, better_solutions, worse_solutions):
                # Compute loss
                loss = self.compute_loss(problem, better, worse, margin)
                
                # Accumulate loss
                batch_loss += loss.item() / len(problems)
                
                # Score solutions individually (for metrics only)
                better_score = self.score_solution(problem, better)
                worse_score = self.score_solution(problem, worse)
                
                # Compute accuracy (proportion of pairs correctly ranked)
                if better_score > worse_score:
                    batch_accuracy += 1.0 / len(problems)
                
                # Backward pass
                if self.use_amp:
                    scaled_loss = self.scaler.scale(loss / len(problems))
                    scaled_loss.backward()
                else:
                    (loss / len(problems)).backward()
        
        # Clip gradients and optimize
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.model.parameters(),
                max_norm=self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.model.parameters(),
                max_norm=self.max_grad_norm
            )
            self.optimizer.step()
        
        return {
            "loss": batch_loss,
            "accuracy": batch_accuracy
        }
    
    def train(
        self,
        train_dataset: Union[Dataset, List[Dict[str, str]]],
        eval_dataset: Optional[Union[Dataset, List[Dict[str, str]]]] = None,
        num_epochs: int = 3,
        batch_size: int = 8,
        log_interval: int = 10,
        eval_interval: int = 50,
        save_interval: int = 200,
        output_dir: str = "./models/prm",
        margin: float = 0.5,
        warmup_steps: int = 100,
        callback: Optional[callable] = None
    ) -> Dict[str, List[float]]:
        """Train the PRM using preference learning.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            num_epochs: Number of training epochs
            batch_size: Batch size
            log_interval: Logging interval
            eval_interval: Evaluation interval
            save_interval: Checkpoint saving interval
            output_dir: Output directory
            margin: Margin for ranking loss
            warmup_steps: Number of warmup steps
            callback: Callback function (optional)
            
        Returns:
            Dictionary with training statistics
        """
        if self.inference_only:
            raise ValueError("Cannot train in inference-only mode")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert datasets to list format if needed
        train_data = []
        if isinstance(train_dataset, Dataset):
            for item in train_dataset:
                train_data.append({
                    "problem": item["problem"],
                    "better_solution": item["better_solution"],
                    "worse_solution": item["worse_solution"]
                })
        else:
            train_data = train_dataset
        
        eval_data = None
        if eval_dataset is not None:
            eval_data = []
            if isinstance(eval_dataset, Dataset):
                for item in eval_dataset:
                    eval_data.append({
                        "problem": item["problem"],
                        "better_solution": item["better_solution"],
                        "worse_solution": item["worse_solution"]
                    })
            else:
                eval_data = eval_dataset
        
        # Create scheduler
        num_training_steps = num_epochs * len(train_data) // batch_size
        scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Initialize stats
        stats = {
            "loss": [],
            "accuracy": [],
            "eval_loss": [],
            "eval_accuracy": []
        }
        
        # Training loop
        logging.info(f"Starting PRM training for {num_epochs} epochs")
        global_step = 0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Shuffle data
            import random
            random.shuffle(train_data)
            
            # Create batches
            num_batches = (len(train_data) + batch_size - 1) // batch_size
            
            # Process each batch
            for batch_idx in range(num_batches):
                # Get batch data
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(train_data))
                batch_data = train_data[start_idx:end_idx]
                
                # Extract batch components
                problems = [item["problem"] for item in batch_data]
                better_solutions = [item["better_solution"] for item in batch_data]
                worse_solutions = [item["worse_solution"] for item in batch_data]
                
                # Train on batch
                batch_stats = self.train_on_batch(
                    problems=problems,
                    better_solutions=better_solutions,
                    worse_solutions=worse_solutions,
                    margin=margin
                )
                
                # Update stats
                for key, value in batch_stats.items():
                    stats[key].append(value)
                
                # Step scheduler
                scheduler.step()
                
                # Log progress
                global_step += 1
                if global_step % log_interval == 0:
                    # Log to console
                    logging.info(
                        f"Epoch {epoch+1}/{num_epochs}, "
                        f"Batch {batch_idx+1}/{num_batches}, "
                        f"Loss: {batch_stats['loss']:.4f}, "
                        f"Accuracy: {batch_stats['accuracy']:.4f}"
                    )
                    
                    # Callback
                    if callback:
                        callback(self, stats, epoch, batch_idx)
                
                # Evaluate if needed
                if eval_data and global_step % eval_interval == 0:
                    eval_stats = self.evaluate(eval_data, batch_size=batch_size)
                    
                    # Update stats
                    stats["eval_loss"].append(eval_stats["loss"])
                    stats["eval_accuracy"].append(eval_stats["accuracy"])
                    
                    # Log to console
                    logging.info(
                        f"Evaluation: "
                        f"Loss: {eval_stats['loss']:.4f}, "
                        f"Accuracy: {eval_stats['accuracy']:.4f}"
                    )
                    
                    # Callback
                    if callback:
                        callback(self, stats, epoch, batch_idx)
                
                # Save checkpoint if needed
                if global_step % save_interval == 0:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    self.model.save_model(checkpoint_path)
                    
                    # Save stats
                    with open(os.path.join(checkpoint_path, "stats.json"), "w") as f:
                        json.dump(stats, f, indent=2)
                    
                    logging.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Log epoch stats
            epoch_time = time.time() - epoch_start_time
            epoch_loss = sum(stats["loss"][-num_batches:]) / num_batches
            epoch_accuracy = sum(stats["accuracy"][-num_batches:]) / num_batches
            
            logging.info(
                f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, "
                f"Loss: {epoch_loss:.4f}, "
                f"Accuracy: {epoch_accuracy:.4f}"
            )
        
        # Save final model
        final_path = os.path.join(output_dir, "final")
        self.model.save_model(final_path)
        
        # Save stats
        with open(os.path.join(output_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        
        logging.info(f"Training completed. Final model saved to {final_path}")
        
        return stats
    
    def evaluate(
        self,
        eval_dataset: Union[Dataset, List[Dict[str, str]]],
        batch_size: int = 8
    ) -> Dict[str, float]:
        """Evaluate the PRM on a dataset.
        
        Args:
            eval_dataset: Evaluation dataset
            batch_size: Batch size
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Ensure model is in evaluation mode
        self.model.model.eval()
        
        # Convert dataset to list format if needed
        eval_data = []
        if isinstance(eval_dataset, Dataset):
            for item in eval_dataset:
                eval_data.append({
                    "problem": item["problem"],
                    "better_solution": item["better_solution"],
                    "worse_solution": item["worse_solution"]
                })
        else:
            eval_data = eval_dataset
        
        # Initialize metrics
        total_loss = 0.0
        total_accuracy = 0.0
        
        # Process in batches
        num_batches = (len(eval_data) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                # Get batch data
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(eval_data))
                batch_data = eval_data[start_idx:end_idx]
                
                # Extract batch components
                problems = [item["problem"] for item in batch_data]
                better_solutions = [item["better_solution"] for item in batch_data]
                worse_solutions = [item["worse_solution"] for item in batch_data]
                
                # Compute loss for each example
                batch_loss = 0.0
                batch_accuracy = 0.0
                
                for problem, better, worse in zip(problems, better_solutions, worse_solutions):
                    # Compute loss
                    loss = self.compute_loss(problem, better, worse)
                    batch_loss += loss.item()
                    
                    # Score solutions individually
                    better_score = self.score_solution(problem, better)
                    worse_score = self.score_solution(problem, worse)
                    
                    # Compute accuracy (proportion of pairs correctly ranked)
                    if better_score > worse_score:
                        batch_accuracy += 1.0
                
                # Normalize by batch size
                batch_size_actual = end_idx - start_idx
                total_loss += batch_loss / batch_size_actual
                total_accuracy += batch_accuracy / batch_size_actual
        
        # Compute averages
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return {
            "loss": avg_loss,
            "accuracy": avg_accuracy
        }
