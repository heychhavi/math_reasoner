"""Group Relative Policy Optimization trainer implementation."""

import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from ai_math_reasoning.models import MathReasoningModel
from ai_math_reasoning.grpo.experience import ExperienceMaker, TrajectoryBatch


class GRPOTrainer:
    """Trainer for Group Relative Policy Optimization (GRPO).
    
    GRPO is a variant of RLHF that uses group-relative rewards rather
    than absolute rewards, making training more stable and efficient.
    """
    
    def __init__(
        self,
        policy_model: MathReasoningModel,
        reward_model: Optional[MathReasoningModel] = None,
        reference_model: Optional[MathReasoningModel] = None,
        group_size: int = 8,
        learning_rate: float = 5e-6,
        weight_decay: float = 0.01,
        kl_coef: float = 0.05,
        clip_range: float = 0.2,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
        dtype: torch.dtype = torch.float16
    ):
        """Initialize GRPO trainer.
        
        Args:
            policy_model: Policy model to optimize
            reward_model: Reward model (optional)
            reference_model: Reference model for KL penalty (optional)
            group_size: Number of solutions per problem
            learning_rate: Learning rate
            weight_decay: Weight decay
            kl_coef: KL penalty coefficient
            clip_range: PPO clipping range
            max_grad_norm: Maximum gradient norm
            use_amp: Whether to use automatic mixed precision
            dtype: Data type for training
        """
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.reference_model = reference_model
        self.group_size = group_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.dtype = dtype
        
        # Device to use
        self.device = policy_model.device
        
        # Create experience maker
        self.experience_maker = ExperienceMaker(
            policy_model=policy_model,
            reward_model=reward_model,
            reference_model=reference_model,
            group_size=group_size,
            use_reference_kl=(reference_model is not None),
            kl_coef=kl_coef
        )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scaler for AMP
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Stats for tracking
        self.stats = {
            "loss": [],
            "reward": [],
            "kl_div": [],
            "pg_loss": [],
            "eval_loss": [],
            "eval_reward": [],
            "eval_kl_div": []
        }
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for policy model.
        
        Returns:
            AdamW optimizer
        """
        # Get trainable parameters
        trainable_params = [p for p in self.policy_model.model.parameters() if p.requires_grad]
        
        # Create optimizer
        optimizer = AdamW(
            params=trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        return optimizer
    
    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_steps: int,
        warmup_steps: int
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer
            num_steps: Total number of training steps
            warmup_steps: Number of warmup steps
            
        Returns:
            Learning rate scheduler
        """
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_steps
        )
    
    def _compute_policy_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Compute policy loss using PPO clipping.
        
        Args:
            logprobs: Log probabilities of current policy
            old_logprobs: Log probabilities of old policy
            advantages: Advantages
            
        Returns:
            Tuple of (policy loss, KL divergence, stats)
        """
        # Calculate ratio between new and old policies
        log_ratio = logprobs - old_logprobs
        ratio = torch.exp(log_ratio)
        
        # Clipped loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        # Calculate KL divergence
        kl_div = (old_logprobs - logprobs).mean()
        
        # Stats
        stats = {
            "pg_loss": pg_loss.item(),
            "kl_div": kl_div.item(),
            "ratio": ratio.mean().item(),
            "ratio_clipped": (ratio > (1.0 + self.clip_range)).float().mean().item()
        }
        
        return pg_loss, kl_div, stats
    
    def _load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Loaded checkpoint data
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.policy_model.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state if available
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state if available
        if "scheduler_state_dict" in checkpoint and hasattr(self, "scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load scaler state if available
        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        # Load stats
        if "stats" in checkpoint:
            self.stats = checkpoint["stats"]
        
        # Load episode
        episode = checkpoint.get("episode", 0)
        
        return {
            "episode": episode,
            "model_state_dict": checkpoint.get("model_state_dict"),
            "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
            "scheduler_state_dict": checkpoint.get("scheduler_state_dict"),
            "scaler_state_dict": checkpoint.get("scaler_state_dict"),
            "stats": checkpoint.get("stats")
        }
    
    def _save_checkpoint(
        self,
        output_dir: str,
        episode: int,
        scheduler = None
    ) -> None:
        """Save checkpoint.
        
        Args:
            output_dir: Output directory
            episode: Current episode
            scheduler: Learning rate scheduler (optional)
        """
        # Create checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create checkpoint data
        checkpoint = {
            "model_state_dict": self.policy_model.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode": episode,
            "stats": self.stats
        }
        
        # Add scheduler state if available
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        # Add scaler state if available
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{episode:04d}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint
        latest_path = os.path.join(output_dir, "checkpoint-latest.pt")
        torch.save(checkpoint, latest_path)
        
        logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    def train_on_batch(
        self,
        batch: TrajectoryBatch
    ) -> Dict[str, float]:
        """Train policy on a single batch.
        
        Args:
            batch: Batch of trajectories
            
        Returns:
            Dictionary with training metrics
        """
        # Ensure model is in training mode
        self.policy_model.model.train()
        
        # Prepare data
        prompts = batch.prompts
        solutions = batch.solutions
        rewards = batch.rewards
        
        # Skip if batch is empty or has no rewards
        if not prompts or not rewards:
            return {
                "loss": 0.0,
                "reward": 0.0,
                "kl_div": 0.0,
                "pg_loss": 0.0
            }
        
        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, dtype=self.dtype, device=self.device)
        
        # Compute old log probabilities
        with torch.no_grad():
            old_logprobs_list = []
            for prompt, solution in zip(prompts, solutions):
                old_logprob = self.policy_model.compute_logprobs(prompt, solution)
                old_logprobs_list.append(old_logprob)
            
            old_logprobs = torch.tensor(old_logprobs_list, dtype=self.dtype, device=self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with AMP if enabled
        with torch.cuda.amp.autocast() if self.use_amp else torch.no_grad():
            # Compute current log probabilities
            logprobs_list = []
            for prompt, solution in zip(prompts, solutions):
                logprob = self.policy_model.compute_logprobs(prompt, solution)
                logprobs_list.append(logprob)
            
            logprobs = torch.tensor(logprobs_list, dtype=self.dtype, device=self.device)
            
            # Compute policy loss
            pg_loss, kl_div, pg_stats = self._compute_policy_loss(
                logprobs=logprobs,
                old_logprobs=old_logprobs,
                advantages=rewards_tensor
            )
            
            # Combine losses
            loss = pg_loss + self.kl_coef * kl_div
        
        # Backward pass with AMP if enabled
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.model.parameters(),
                max_norm=self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.model.parameters(),
                max_norm=self.max_grad_norm
            )
            self.optimizer.step()
        
        # Update stats
        stats = {
            "loss": loss.item(),
            "reward": rewards_tensor.mean().item(),
            "kl_div": kl_div.item(),
            "pg_loss": pg_loss.item()
        }
        
        # Update running stats
        for key, value in stats.items():
            if key in self.stats:
                self.stats[key].append(value)
        
        return stats
    
    def evaluate(
        self,
        problems: List[str],
        reference_answers: Optional[List[str]] = None,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """Evaluate policy on a dataset.
        
        Args:
            problems: List of problems
            reference_answers: List of reference answers (optional)
            batch_size: Batch size
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Ensure model is in evaluation mode
        self.policy_model.model.eval()
        
        # Initialize stats
        eval_stats = {
            "eval_loss": 0.0,
            "eval_reward": 0.0,
            "eval_kl_div": 0.0,
            "accuracy": 0.0,
            "num_samples": 0
        }
        
        # Create batches
        num_samples = len(problems)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        # Collect trajectories for each batch
        for batch_idx in range(num_batches):
            # Get batch data
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_problems = problems[start_idx:end_idx]
            batch_answers = reference_answers[start_idx:end_idx] if reference_answers else None
            
            # Collect trajectories
            batch = self.experience_maker.collect_trajectories(
                problems=batch_problems,
                reference_answers=batch_answers
            )
            
            # Update stats
            if batch.rewards:
                eval_stats["eval_reward"] += sum(batch.rewards) / len(batch.rewards) * (end_idx - start_idx)
            
            # Count correct answers
            if reference_answers and batch.answers:
                from ai_math_reasoning.data.processing.math_processing import compare_answers
                
                for i, (problem_idx, answer) in enumerate(zip(range(start_idx, end_idx), batch.answers)):
                    if problem_idx < len(reference_answers) and answer:
                        if compare_answers(answer, reference_answers[problem_idx]):
                            eval_stats["accuracy"] += 1
            
            eval_stats["num_samples"] += end_idx - start_idx
        
        # Compute averages
        if eval_stats["num_samples"] > 0:
            eval_stats["eval_reward"] /= eval_stats["num_samples"]
            eval_stats["accuracy"] /= eval_stats["num_samples"]
        
        # Update running stats
        for key, value in eval_stats.items():
            if key in self.stats and isinstance(value, (int, float)):
                self.stats[key].append(value)
        
        return eval_stats
    
    def train(
        self,
        train_dataset: Union[Dataset, Dict[str, List]],
        eval_dataset: Optional[Union[Dataset, Dict[str, List]]] = None,
        num_episodes: int = 1000,
        batch_size: int = 8,
        log_interval: int = 10,
        eval_interval: int = 100,
        save_interval: int = 500,
        output_dir: str = "./models/grpo",
        warmup_steps: int = 500,
        resume_from: Optional[str] = None,
        callback: Optional[callable] = None
    ) -> Dict[str, List[float]]:
        """Train policy using GRPO.
        
        Args:
            train_dataset: Training dataset or dictionary with problems and answers
            eval_dataset: Evaluation dataset or dictionary (optional)
            num_episodes: Number of training episodes
            batch_size: Batch size
            log_interval: Logging interval
            eval_interval: Evaluation interval
            save_interval: Checkpoint saving interval
            output_dir: Output directory
            warmup_steps: Number of warmup steps
            resume_from: Path to checkpoint to resume from (optional)
            callback: Callback function (optional)
            
        Returns:
            Dictionary with training statistics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract problems and answers from dataset
        if isinstance(train_dataset, dict):
            train_problems = train_dataset["problems"]
            train_answers = train_dataset.get("answers")
        elif hasattr(train_dataset, "to_grpo_training_data"):
            # Use dataset's to_grpo_training_data method
            train_data = train_dataset.to_grpo_training_data()
            train_problems = train_data["problems"]
            train_answers = train_data.get("answers")
        else:
            # Assume it's a HuggingFace dataset
            train_problems = []
            train_answers = []
            for example in train_dataset:
                problem = example.get("problem", example.get("question", ""))
                answer = example.get("answer", None)
                train_problems.append(problem)
                train_answers.append(answer)
        
        # Do the same for evaluation dataset
        eval_problems = None
        eval_answers = None
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_problems = eval_dataset["problems"]
                eval_answers = eval_dataset.get("answers")
            elif hasattr(eval_dataset, "to_grpo_training_data"):
                eval_data = eval_dataset.to_grpo_training_data()
                eval_problems = eval_data["problems"]
                eval_answers = eval_data.get("answers")
            else:
                # Assume it's a HuggingFace dataset
                eval_problems = []
                eval_answers = []
                for example in eval_dataset:
                    problem = example.get("problem", example.get("question", ""))
                    answer = example.get("answer", None)
                    eval_problems.append(problem)
                    eval_answers.append(answer)
        
        # Create learning rate scheduler
        num_steps = num_episodes * len(train_problems) // batch_size
        scheduler = self._create_scheduler(
            optimizer=self.optimizer,
            num_steps=num_steps,
            warmup_steps=warmup_steps
        )
        
        # Resume from checkpoint if provided
        start_episode = 0
        if resume_from:
            checkpoint = self._load_checkpoint(resume_from)
            start_episode = checkpoint["episode"] + 1
            logging.info(f"Resuming from episode {start_episode}")
        
        # Training loop
        logging.info(f"Starting GRPO training for {num_episodes} episodes")
        total_batches = 0
        
        for episode in range(start_episode, num_episodes):
            episode_start_time = time.time()
            episode_stats = {
                "loss": 0.0,
                "reward": 0.0,
                "kl_div": 0.0,
                "pg_loss": 0.0
            }
            
            # Create random batches
            indices = list(range(len(train_problems)))
            random.shuffle(indices)
            
            num_samples = len(train_problems)
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            # Process each batch
            for batch_idx in range(num_batches):
                # Get batch data
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_problems = [train_problems[i] for i in batch_indices]
                batch_answers = [train_answers[i] for i in batch_indices] if train_answers else None
                
                # Collect trajectories
                batch = self.experience_maker.collect_trajectories(
                    problems=batch_problems,
                    reference_answers=batch_answers
                )
                
                # Train on batch
                batch_stats = self.train_on_batch(batch)
                
                # Update episode stats
                for key, value in batch_stats.items():
                    episode_stats[key] += value / num_batches
                
                # Step scheduler
                scheduler.step()
                
                # Log progress
                total_batches += 1
                if total_batches % log_interval == 0:
                    # Log to console
                    logging.info(
                        f"Episode {episode+1}/{num_episodes}, "
                        f"Batch {batch_idx+1}/{num_batches}, "
                        f"Loss: {batch_stats['loss']:.4f}, "
                        f"Reward: {batch_stats['reward']:.4f}, "
                        f"KL: {batch_stats['kl_div']:.4f}, "
                        f"PG Loss: {batch_stats['pg_loss']:.4f}"
                    )
                    
                    # Callback
                    if callback:
                        callback(self, self.stats, episode, batch_idx)
            
            # Log episode stats
            episode_time = time.time() - episode_start_time
            logging.info(
                f"Episode {episode+1}/{num_episodes} completed in {episode_time:.2f}s, "
                f"Loss: {episode_stats['loss']:.4f}, "
                f"Reward: {episode_stats['reward']:.4f}, "
                f"KL: {episode_stats['kl_div']:.4f}, "
                f"PG Loss: {episode_stats['pg_loss']:.4f}"
            )
            
            # Callback for episode
            if callback:
                callback(self, self.stats, episode)
            
            # Evaluate if needed
            if eval_problems and (episode + 1) % eval_interval == 0:
                logging.info(f"Evaluating at episode {episode+1}")
                eval_stats = self.evaluate(
                    problems=eval_problems,
                    reference_answers=eval_answers,
                    batch_size=batch_size
                )
                
                logging.info(
                    f"Evaluation results: "
                    f"Reward: {eval_stats['eval_reward']:.4f}, "
                    f"Accuracy: {eval_stats['accuracy']:.4f}"
                )
            
            # Save checkpoint if needed
            if (episode + 1) % save_interval == 0 or episode == num_episodes - 1:
                self._save_checkpoint(
                    output_dir=output_dir,
                    episode=episode,
                    scheduler=scheduler
                )
        
        # Save final model
        self.policy_model.save_model(
            output_dir=os.path.join(output_dir, "final_model")
        )
        
        logging.info("Training completed")
        
        return self.stats
