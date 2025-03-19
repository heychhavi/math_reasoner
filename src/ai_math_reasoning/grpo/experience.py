"""Implementation of experience collection for Group Relative Policy Optimization."""

import os
import time
import logging
import random
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np

from ai_math_reasoning.models import MathReasoningModel
from ai_math_reasoning.data.processing.math_processing import extract_final_answer


@dataclass
class TrajectoryBatch:
    """A batch of trajectories for GRPO.
    
    Attributes:
        prompts: List of input prompts
        solutions: List of generated solutions
        answers: List of extracted answers
        rewards: List of rewards
        group_indices: List of group indices
        solution_times: List of solution generation times
    """
    prompts: List[str]
    solutions: List[str]
    answers: Optional[List[str]] = None
    rewards: Optional[List[float]] = None
    group_indices: Optional[List[int]] = None
    solution_times: Optional[List[float]] = None


class ExperienceMaker:
    """Experience maker for GRPO.
    
    This class collects trajectories for GRPO training by generating
    solutions with a policy model and computing rewards.
    """
    
    def __init__(
        self,
        policy_model: MathReasoningModel,
        reward_model: Optional[MathReasoningModel] = None,
        reference_model: Optional[MathReasoningModel] = None,
        group_size: int = 8,
        temperature_range: Tuple[float, float] = (0.7, 0.9),
        top_p_range: Tuple[float, float] = (0.9, 0.99),
        max_new_tokens: int = 1024,
        prompt_template: Optional[str] = None,
        reward_template: Optional[str] = None,
        use_reference_kl: bool = True,
        kl_coef: float = 0.1,
        max_workers: int = 4,
        timeout: Optional[float] = None,
        verification_func: Optional[callable] = None
    ):
        """Initialize an experience maker for GRPO.
        
        Args:
            policy_model: The model being optimized
            reward_model: External reward model (optional)
            reference_model: Reference model for KL penalty (optional)
            group_size: Number of solutions per problem
            temperature_range: Range of temperatures for sampling
            top_p_range: Range of nucleus sampling parameters
            max_new_tokens: Maximum number of new tokens to generate
            prompt_template: Template for formatting prompts (optional)
            reward_template: Template for reward calculation (optional)
            use_reference_kl: Whether to use KL penalty from reference model
            kl_coef: Coefficient for KL penalty
            max_workers: Maximum number of worker threads
            timeout: Timeout in seconds for generation
            verification_func: Function for solution verification (optional)
        """
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.reference_model = reference_model
        self.group_size = group_size
        self.temperature_range = temperature_range
        self.top_p_range = top_p_range
        self.max_new_tokens = max_new_tokens
        self.use_reference_kl = use_reference_kl and reference_model is not None
        self.kl_coef = kl_coef
        self.max_workers = max_workers
        self.timeout = timeout
        self.verification_func = verification_func
        
        # Default prompt template
        self.prompt_template = prompt_template or (
            "You are a mathematical problem-solving assistant who is very careful and thorough. "
            "Solve the following problem step by step, showing your reasoning. "
            "End your solution with 'Answer: [your final answer]'.\n\n"
            "Problem: {problem}\n\n"
            "Solution:"
        )
        
        # Default reward template for reward model
        self.reward_template = reward_template or (
            "Rate the quality of the following solution to a mathematical problem on a scale "
            "from 0 to 10, where 0 is completely incorrect and 10 is perfect.\n\n"
            "Problem: {problem}\n\n"
            "Solution: {solution}\n\n"
            "Rating:"
        )
        
        # Initialize statistics
        self.total_problems = 0
        self.total_trajectories = 0
        self.total_generation_time = 0.0
        self.total_reward_time = 0.0
    
    def _format_prompt(self, problem: str) -> str:
        """Format the problem using the prompt template.
        
        Args:
            problem: Math problem to format
            
        Returns:
            Formatted prompt
        """
        return self.prompt_template.format(problem=problem)
    
    def _sample_generation_params(self) -> Tuple[float, float]:
        """Sample temperature and top_p for diverse generation.
        
        Returns:
            Tuple of (temperature, top_p)
        """
        temperature = random.uniform(self.temperature_range[0], self.temperature_range[1])
        top_p = random.uniform(self.top_p_range[0], self.top_p_range[1])
        return temperature, top_p
    
    def _generate_solution(
        self,
        problem: str,
        prompt: str
    ) -> Dict[str, Any]:
        """Generate a solution using the policy model.
        
        Args:
            problem: Original mathematical problem
            prompt: Formatted prompt for the model
            
        Returns:
            Dictionary with solution information
        """
        start_time = time.time()
        
        # Sample generation parameters
        temperature, top_p = self._sample_generation_params()
        
        try:
            # Generate solution
            solution = self.policy_model.generate(
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
            
            # Extract answer
            answer = extract_final_answer(solution)
            
            # Compute solution time
            solution_time = time.time() - start_time
            
            # Record generation statistics
            self.total_generation_time += solution_time
            self.total_trajectories += 1
            
            return {
                "problem": problem,
                "prompt": prompt,
                "solution": solution,
                "answer": answer,
                "temperature": temperature,
                "top_p": top_p,
                "solution_time": solution_time,
            }
            
        except Exception as e:
            logging.error(f"Solution generation failed: {str(e)}")
            
            # Return failure result
            return {
                "problem": problem,
                "prompt": prompt,
                "solution": "",
                "answer": None,
                "temperature": temperature,
                "top_p": top_p,
                "solution_time": time.time() - start_time,
                "error": str(e)
            }
    
    def _compute_reward(
        self,
        problem: str,
        solution: str,
        answer: Optional[str] = None,
        reference_solution: Optional[str] = None
    ) -> float:
        """Compute reward for a solution.
        
        Args:
            problem: Original mathematical problem
            solution: Generated solution
            answer: Extracted answer (optional)
            reference_solution: Reference solution for KL penalty (optional)
            
        Returns:
            Reward score
        """
        start_time = time.time()
        reward = 0.0
        
        # Use reward model if available
        if self.reward_model is not None:
            try:
                # Format reward prompt
                reward_prompt = self.reward_template.format(
                    problem=problem,
                    solution=solution
                )
                
                # Generate reward score
                reward_text = self.reward_model.generate(
                    prompt=reward_prompt,
                    max_new_tokens=10,
                    temperature=0.0,  # Deterministic for reward
                    do_sample=False
                )
                
                # Extract numerical score from reward text
                score_text = reward_text.strip()
                
                # Try to extract the score
                for token in score_text.split():
                    try:
                        score = float(token)
                        if 0 <= score <= 10:
                            reward = score / 10.0  # Normalize to [0, 1]
                            break
                    except ValueError:
                        continue
                
            except Exception as e:
                logging.error(f"Reward calculation failed: {str(e)}")
        
        # Apply verification bonus if requested and available
        if self.verification_func is not None and answer is not None:
            try:
                # Verify the solution
                verification_result, verification_confidence = self.verification_func(
                    problem=problem,
                    solution=solution,
                    answer=answer
                )
                
                # Apply verification bonus/penalty
                if verification_result == "verified":
                    reward += 0.2 * verification_confidence
                elif verification_result == "rejected":
                    reward -= 0.1 * verification_confidence
                
                # Clamp reward to [0, 1]
                reward = max(0.0, min(1.0, reward))
                
            except Exception as e:
                logging.error(f"Verification failed: {str(e)}")
        
        # Apply KL penalty if reference model is available
        if self.use_reference_kl and reference_solution is not None:
            try:
                # Simple approximation of KL divergence based on length difference
                # A more sophisticated approach would compute token-level KL
                len_diff = abs(len(solution) - len(reference_solution)) / max(len(reference_solution), 1)
                kl_penalty = self.kl_coef * min(len_diff, 1.0)
                
                # Apply penalty
                reward -= kl_penalty
                
                # Clamp reward to [0, 1]
                reward = max(0.0, min(1.0, reward))
                
            except Exception as e:
                logging.error(f"KL penalty calculation failed: {str(e)}")
        
        # Record reward calculation time
        self.total_reward_time += time.time() - start_time
        
        return reward
    
    def collect_trajectories(
        self,
        problems: List[str],
        reference_answers: Optional[List[str]] = None
    ) -> TrajectoryBatch:
        """Collect trajectories for a batch of problems.
        
        Args:
            problems: List of mathematical problems
            reference_answers: List of reference answers (optional)
            
        Returns:
            Batch of trajectories
        """
        self.total_problems += len(problems)
        start_time = time.time()
        
        # Format prompts
        prompts = [self._format_prompt(problem) for problem in problems]
        
        # Generate solutions
        all_solutions = []
        all_solution_times = []
        all_group_indices = []
        
        # Process each problem
        for group_idx, (problem, prompt) in enumerate(zip(problems, prompts)):
            # Generate multiple solutions per problem
            group_solutions = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit generation tasks
                future_to_problem = {
                    executor.submit(self._generate_solution, problem, prompt): i
                    for i in range(self.group_size)
                }
                
                # Process results as they complete
                for future in as_completed(future_to_problem):
                    solution_data = future.result()
                    
                    if solution_data["solution"]:
                        group_solutions.append(solution_data)
            
            # Add to overall solutions
            all_solutions.extend(group_solutions)
            all_solution_times.extend([s["solution_time"] for s in group_solutions])
            all_group_indices.extend([group_idx] * len(group_solutions))
        
        # Extract solutions and answers
        solutions = [s["solution"] for s in all_solutions]
        answers = [s["answer"] for s in all_solutions]
        
        # Generate reference solutions for KL penalty if needed
        reference_solutions = None
        if self.use_reference_kl and self.reference_model is not None:
            try:
                reference_solutions = []
                
                for problem, prompt in zip(problems, prompts):
                    # Generate a reference solution with temperature=0
                    ref_solution = self.reference_model.generate(
                        prompt=prompt,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.0,
                        do_sample=False
                    )
                    
                    # Add to list
                    reference_solutions.append(ref_solution)
                
            except Exception as e:
                logging.error(f"Reference solution generation failed: {str(e)}")
                reference_solutions = None
        
        # Compute rewards
        rewards = []
        
        for i, (problem, solution, answer) in enumerate(zip(
            [problems[idx] for idx in all_group_indices],
            solutions,
            answers
        )):
            # Get reference solution for this problem
            ref_solution = None
            if reference_solutions is not None:
                ref_solution = reference_solutions[all_group_indices[i]]
            
            # Get reference answer
            ref_answer = None
            if reference_answers is not None:
                ref_answer = reference_answers[all_group_indices[i]]
            
            # Compute reward
            reward = self._compute_reward(
                problem=problem,
                solution=solution,
                answer=answer,
                reference_solution=ref_solution
            )
            
            rewards.append(reward)
        
        # Normalize rewards within each group
        normalized_rewards = []
        for group_idx in range(len(problems)):
            # Get rewards for this group
            group_rewards = [r for i, r in enumerate(rewards) if all_group_indices[i] == group_idx]
            
            if group_rewards:
                # Compute mean and standard deviation
                mean_reward = sum(group_rewards) / len(group_rewards)
                std_reward = (sum((r - mean_reward) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
                
                # Normalize rewards
                if std_reward > 1e-6:
                    # Z-score normalization
                    normalized_group_rewards = [(r - mean_reward) / std_reward for r in group_rewards]
                else:
                    # If standard deviation is too small, use zeros
                    normalized_group_rewards = [0.0] * len(group_rewards)
                
                # Add to normalized rewards
                for i, idx in enumerate([j for j, g in enumerate(all_group_indices) if g == group_idx]):
                    while len(normalized_rewards) <= idx:
                        normalized_rewards.append(0.0)
                    normalized_rewards[idx] = normalized_group_rewards[i]
        
        # Create batch
        batch = TrajectoryBatch(
            prompts=prompts,
            solutions=solutions,
            answers=answers,
            rewards=normalized_rewards,
            group_indices=all_group_indices,
            solution_times=all_solution_times
        )
        
        logging.info(
            f"Collected {len(solutions)} trajectories for {len(problems)} problems "
            f"in {time.time() - start_time:.2f}s"
        )
        
        return batch
    
    def get_stats(self) -> Dict[str, Any]:
        """Get experience maker statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_problems": self.total_problems,
            "total_trajectories": self.total_trajectories,
            "trajectories_per_problem": self.total_trajectories / max(1, self.total_problems),
            "avg_generation_time": self.total_generation_time / max(1, self.total_trajectories),
            "avg_reward_time": self.total_reward_time / max(1, self.total_trajectories),
        }
        
        return stats
