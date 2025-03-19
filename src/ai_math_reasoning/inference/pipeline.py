"""Inference pipeline for mathematical reasoning."""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from datetime import datetime

import torch
import numpy as np

from ai_math_reasoning.models import MathReasoningModel, create_model
from ai_math_reasoning.prm.reranker import PRMReranker
from ai_math_reasoning.prm.agent import MathAgent
from ai_math_reasoning.prm.multi_agent import MultiAgentSystem
from ai_math_reasoning.prm.verifier import SolutionVerifier
from ai_math_reasoning.data.processing.math_processing import (
    extract_final_answer,
    compare_answers
)


class MathInferencePipeline:
    """Pipeline for mathematical reasoning inference.
    
    This pipeline combines a base model, multi-agent system, and PRM
    to solve mathematical problems.
    """
    
    def __init__(
        self,
        base_model: MathReasoningModel,
        prm_model_path: Optional[str] = None,
        verifier_model_path: Optional[str] = None,
        num_agents: int = 3,
        max_attempts: int = 1,
        temperature_range: Tuple[float, float] = (0.7, 0.9),
        top_p_range: Tuple[float, float] = (0.9, 0.99),
        max_new_tokens: int = 1024,
        use_verification: bool = True,
        use_ensemble: bool = True,
        timeout: Optional[float] = None
    ):
        """Initialize the inference pipeline.
        
        Args:
            base_model: Base language model
            prm_model_path: Path to PRM model (optional)
            verifier_model_path: Path to verifier model (optional)
            num_agents: Number of agents to use
            max_attempts: Maximum number of solution attempts per agent
            temperature_range: Range of temperatures for sampling
            top_p_range: Range of nucleus sampling parameters
            max_new_tokens: Maximum number of new tokens to generate
            use_verification: Whether to use solution verification
            use_ensemble: Whether to use ensemble solution selection
            timeout: Timeout in seconds
        """
        self.base_model = base_model
        self.num_agents = num_agents
        self.max_attempts = max_attempts
        self.temperature_range = temperature_range
        self.top_p_range = top_p_range
        self.max_new_tokens = max_new_tokens
        self.use_verification = use_verification
        self.use_ensemble = use_ensemble
        self.timeout = timeout
        
        # Initialize PRM if path provided
        self.prm = None
        if prm_model_path:
            logging.info(f"Loading PRM model from {prm_model_path}")
            model_type = base_model.__class__.__name__.lower().replace("model", "")
            prm_base_model = create_model(
                model_type=model_type,
                model_name_or_path=prm_model_path
            )
            self.prm = PRMReranker(base_model=prm_base_model, inference_only=True)
        
        # Initialize verifier if path provided and verification enabled
        self.verifier = None
        if use_verification:
            if verifier_model_path:
                logging.info(f"Loading verifier model from {verifier_model_path}")
                model_type = base_model.__class__.__name__.lower().replace("model", "")
                verifier_base_model = create_model(
                    model_type=model_type,
                    model_name_or_path=verifier_model_path
                )
                self.verifier = SolutionVerifier(model=verifier_base_model)
            else:
                # Use base model for verification if no specific verifier provided
                self.verifier = SolutionVerifier(model=base_model)
        
        # Create multi-agent system
        self.multi_agent_system = MultiAgentSystem(
            base_model=base_model,
            num_agents=num_agents,
            prm=self.prm,
            verifier=self.verifier,
            max_attempts=max_attempts,
            temperature_range=temperature_range,
            top_p_range=top_p_range,
            max_new_tokens=max_new_tokens,
            use_verification=use_verification,
            use_ensemble=use_ensemble
        )
    
    def solve(
        self,
        problem: str,
        reference_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Solve a mathematical problem.
        
        Args:
            problem: Math problem to solve
            reference_answer: Reference answer (optional)
            
        Returns:
            Dictionary with solution information
        """
        start_time = time.time()
        
        # Solve the problem using the multi-agent system
        solution_data = self.multi_agent_system.solve_problem(
            problem=problem,
            timeout=self.timeout
        )
        
        # Calculate solution time
        solution_time = time.time() - start_time
        
        # Extract the final answer from the solution
        answer = extract_final_answer(solution_data["solution"])
        
        # Add solution metadata
        result = {
            "problem": problem,
            "solution": solution_data["solution"],
            "answer": answer,
            "solution_time": solution_time,
            "method": solution_data["method"],
            "confidence": solution_data.get("confidence", None),
            "verification_result": solution_data.get("verification_result", None),
        }
        
        # Compare with reference answer if provided
        if reference_answer:
            is_correct = compare_answers(answer, reference_answer)
            result["reference_answer"] = reference_answer
            result["is_correct"] = is_correct
        
        return result
    
    def format_final_solution(
        self,
        result: Dict[str, Any],
        include_reasoning: bool = False,
        include_metadata: bool = False
    ) -> str:
        """Format the solution for presentation.
        
        Args:
            result: Solution data
            include_reasoning: Whether to include reasoning steps
            include_metadata: Whether to include solution metadata
            
        Returns:
            Formatted solution
        """
        if not include_reasoning and not include_metadata:
            # Just return the answer
            return result["answer"] if result["answer"] else "No answer found."
        
        formatted_solution = []
        
        # Add answer
        if result["answer"]:
            formatted_solution.append(f"Answer: {result['answer']}")
        else:
            formatted_solution.append("No answer found.")
        
        # Add reasoning if requested
        if include_reasoning:
            formatted_solution.append("\nReasoning:")
            solution_parts = result["solution"].split("Answer:")
            if len(solution_parts) > 1:
                # Only include the reasoning part
                formatted_solution.append(solution_parts[0].strip())
            else:
                # Include the entire solution if it doesn't have a clear Answer: section
                formatted_solution.append(result["solution"].strip())
        
        # Add metadata if requested
        if include_metadata:
            formatted_solution.append("\nMetadata:")
            formatted_solution.append(f"- Solution time: {result['solution_time']:.2f} seconds")
            formatted_solution.append(f"- Method: {result['method']}")
            if result.get("confidence") is not None:
                formatted_solution.append(f"- Confidence: {result['confidence']:.4f}")
            if result.get("verification_result") is not None:
                formatted_solution.append(f"- Verification: {result['verification_result']}")
            if "is_correct" in result:
                formatted_solution.append(f"- Correct: {result['is_correct']}")
                formatted_solution.append(f"- Reference answer: {result['reference_answer']}")
        
        return "\n".join(formatted_solution)
    
    def evaluate_on_dataset(
        self,
        problems: List[str],
        reference_answers: Optional[List[str]] = None,
        output_file: Optional[str] = None,
        include_solutions: bool = False,
        include_problems: bool = False
    ) -> Dict[str, Any]:
        """Evaluate the pipeline on a dataset of problems.
        
        Args:
            problems: List of problems
            reference_answers: List of reference answers (optional)
            output_file: Path to output file (optional)
            include_solutions: Whether to include full solutions in the output
            include_problems: Whether to include full problems in the output
            
        Returns:
            Dictionary with evaluation results
        """
        num_problems = len(problems)
        logging.info(f"Evaluating on {num_problems} problems")
        
        results = []
        correct_count = 0
        verified_count = 0
        answer_count = 0
        solution_times = []
        
        # Verification metrics
        true_positives = 0  # Correctly identified as correct
        false_positives = 0  # Incorrectly identified as correct
        true_negatives = 0  # Correctly identified as incorrect
        false_negatives = 0  # Incorrectly identified as incorrect
        
        # Process each problem
        for i, problem in enumerate(problems):
            logging.info(f"Solving problem {i+1}/{num_problems}")
            
            # Get reference answer if available
            reference_answer = reference_answers[i] if reference_answers else None
            
            # Solve the problem
            result = self.solve(problem, reference_answer)
            
            # Track metrics
            solution_times.append(result["solution_time"])
            
            if result["answer"]:
                answer_count += 1
            
            if result.get("verification_result") == "verified":
                verified_count += 1
            
            if "is_correct" in result and result["is_correct"]:
                correct_count += 1
            
            # Track verification metrics if verification was used and reference answer is available
            if self.use_verification and "is_correct" in result:
                verification_result = result.get("verification_result")
                is_correct = result["is_correct"]
                
                if verification_result == "verified" and is_correct:
                    true_positives += 1
                elif verification_result == "verified" and not is_correct:
                    false_positives += 1
                elif verification_result != "verified" and not is_correct:
                    true_negatives += 1
                elif verification_result != "verified" and is_correct:
                    false_negatives += 1
            
            # Filter output if requested
            output_result = result.copy()
            if not include_solutions:
                output_result.pop("solution", None)
            
            if not include_problems:
                output_result.pop("problem", None)
            
            results.append(output_result)
        
        # Calculate aggregate metrics
        accuracy = correct_count / num_problems if reference_answers else None
        verification_rate = verified_count / num_problems
        answer_rate = answer_count / num_problems
        avg_time = sum(solution_times) / num_problems
        
        # Calculate verification metrics
        verification_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        verification_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        verification_f1 = 2 * (verification_precision * verification_recall) / (verification_precision + verification_recall) if (verification_precision + verification_recall) > 0 else 0
        
        # Compile evaluation results
        evaluation_results = {
            "num_problems": num_problems,
            "accuracy": accuracy,
            "verification_rate": verification_rate,
            "answer_rate": answer_rate,
            "avg_time": avg_time,
            "verification_precision": verification_precision,
            "verification_recall": verification_recall,
            "verification_f1": verification_f1,
            "results": results
        }
        
        # Save to file if specified
        if output_file:
            logging.info(f"Saving evaluation results to {output_file}")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, indent=2)
        
        return evaluation_results
