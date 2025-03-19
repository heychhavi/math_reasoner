"""Implementation of a mathematical reasoning agent."""

import time
import logging
import random
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import numpy as np

from ai_math_reasoning.models import MathReasoningModel
from ai_math_reasoning.data.processing.math_processing import extract_final_answer


class MathAgent:
    """Agent for solving mathematical problems.
    
    This class represents a single mathematical reasoning agent that can
    solve problems and provide confidence scores for its solutions.
    """
    
    def __init__(
        self,
        model: MathReasoningModel,
        agent_id: Optional[int] = None,
        max_attempts: int = 1,
        temperature_range: Tuple[float, float] = (0.7, 0.9),
        top_p_range: Tuple[float, float] = (0.9, 0.99),
        max_new_tokens: int = 1024,
        prompt_template: Optional[str] = None
    ):
        """Initialize a mathematical reasoning agent.
        
        Args:
            model: Base language model
            agent_id: Unique identifier for the agent (optional)
            max_attempts: Maximum number of solution attempts
            temperature_range: Range of temperatures for sampling
            top_p_range: Range of nucleus sampling parameters
            max_new_tokens: Maximum number of new tokens to generate
            prompt_template: Template for formatting prompts (optional)
        """
        self.model = model
        self.agent_id = agent_id if agent_id is not None else random.randint(1000, 9999)
        self.max_attempts = max_attempts
        self.temperature_range = temperature_range
        self.top_p_range = top_p_range
        self.max_new_tokens = max_new_tokens
        
        # Default prompt template
        self.prompt_template = prompt_template or (
            "You are a mathematical problem-solving assistant who is very careful and thorough. "
            "Solve the following problem step by step, showing your reasoning. "
            "End your solution with 'Answer: [your final answer]'.\n\n"
            "Problem: {problem}\n\n"
            "Solution:"
        )
        
        # Track agent statistics
        self.total_problems = 0
        self.successful_solutions = 0
        self.solution_times = []
    
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
    
    def solve(
        self,
        problem: str,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Solve a mathematical problem.
        
        Args:
            problem: Math problem to solve
            timeout: Timeout in seconds (optional)
            
        Returns:
            Dictionary with solution information
        """
        self.total_problems += 1
        start_time = time.time()
        
        # Track multiple attempts
        solutions = []
        confidences = []
        answers = []
        attempt_times = []
        
        # Try multiple attempts
        for attempt in range(self.max_attempts):
            attempt_start = time.time()
            
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                logging.info(f"Agent {self.agent_id} timed out after {attempt} attempts")
                break
            
            # Sample parameters for diversity
            temperature, top_p = self._sample_generation_params()
            
            # Format the prompt
            prompt = self._format_prompt(problem)
            
            # Generate solution
            try:
                solution = self.model.generate(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                # Extract answer
                answer = extract_final_answer(solution)
                
                # Calculate solution time
                attempt_time = time.time() - attempt_start
                
                # Calculate a simple confidence score - can be replaced with a more sophisticated model
                # Higher temperature = lower confidence
                confidence = 1.0 - (temperature - self.temperature_range[0]) / (
                    self.temperature_range[1] - self.temperature_range[0]
                )
                
                # Length-based confidence adjustment
                if answer and len(solution) > 100:
                    # Longer solutions with answers get a slight confidence boost
                    confidence *= 1.1
                
                # Clamp confidence to [0, 1]
                confidence = max(0.0, min(1.0, confidence))
                
                # Track attempt
                solutions.append(solution)
                confidences.append(confidence)
                answers.append(answer)
                attempt_times.append(attempt_time)
                
                logging.info(
                    f"Agent {self.agent_id} generated solution for attempt {attempt+1}/{self.max_attempts} "
                    f"(t={temperature:.2f}, p={top_p:.2f}, conf={confidence:.4f}, time={attempt_time:.2f}s)"
                )
                
            except Exception as e:
                logging.error(f"Agent {self.agent_id} failed during generation: {str(e)}")
                continue
        
        # Calculate total solution time
        solution_time = time.time() - start_time
        self.solution_times.append(solution_time)
        
        # If no successful solutions, return failure
        if not solutions:
            return {
                "status": "failure",
                "solution": "",
                "answer": None,
                "confidence": 0.0,
                "agent_id": self.agent_id,
                "solution_time": solution_time,
                "attempts": 0,
                "temperature": None,
                "top_p": None,
            }
        
        # Select the best solution based on confidence
        best_idx = np.argmax(confidences)
        selected_solution = solutions[best_idx]
        selected_answer = answers[best_idx]
        selected_confidence = confidences[best_idx]
        
        # Track successful solutions
        if selected_answer:
            self.successful_solutions += 1
        
        return {
            "status": "success",
            "solution": selected_solution,
            "answer": selected_answer,
            "confidence": selected_confidence,
            "agent_id": self.agent_id,
            "solution_time": solution_time,
            "attempts": len(solutions),
            "all_solutions": solutions,
            "all_confidences": confidences,
            "all_answers": answers,
            "attempt_times": attempt_times,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.
        
        Returns:
            Dictionary with agent statistics
        """
        success_rate = self.successful_solutions / max(1, self.total_problems)
        avg_time = sum(self.solution_times) / max(1, len(self.solution_times))
        
        return {
            "agent_id": self.agent_id,
            "total_problems": self.total_problems,
            "successful_solutions": self.successful_solutions,
            "success_rate": success_rate,
            "avg_solution_time": avg_time,
        }
