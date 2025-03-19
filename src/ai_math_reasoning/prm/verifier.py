"""Solution verification for mathematical reasoning."""

import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_math_reasoning.models import MathReasoningModel
from ai_math_reasoning.data.processing.math_processing import extract_final_answer, compare_answers


class SolutionVerifier:
    """Verifier for mathematical solutions.
    
    This class is used to verify the correctness of mathematical solutions,
    either by checking against a reference answer or by using a language
    model to assess the solution's validity.
    """
    
    def __init__(
        self,
        base_model: Optional[MathReasoningModel] = None,
        verification_template: Optional[str] = None,
        confidence_threshold: float = 0.8,
        enable_step_by_step: bool = True,
        step_verification_template: Optional[str] = None,
        max_tokens: int = 256
    ):
        """Initialize a solution verifier.
        
        Args:
            base_model: Base language model for verification (optional)
            verification_template: Template for verification prompts (optional)
            confidence_threshold: Threshold for verification confidence
            enable_step_by_step: Whether to verify solutions step by step
            step_verification_template: Template for step-by-step verification (optional)
            max_tokens: Maximum tokens for verification generation
        """
        self.model = base_model
        self.confidence_threshold = confidence_threshold
        self.enable_step_by_step = enable_step_by_step
        self.max_tokens = max_tokens
        
        # Default verification template
        self.verification_template = verification_template or (
            "You are tasked with verifying a solution to a mathematical problem. "
            "Carefully check if the solution is correct and leads to the right answer.\n\n"
            "Problem: {problem}\n\n"
            "Solution: {solution}\n\n"
            "{answer_text}"
            "Is this solution correct? Answer only with 'Correct' or 'Incorrect', "
            "followed by a brief explanation of your verification process."
        )
        
        # Default step-by-step verification template
        self.step_verification_template = step_verification_template or (
            "You are tasked with verifying a solution to a mathematical problem step by step. "
            "For each step in the solution, check if it is mathematically valid and leads to the next step correctly.\n\n"
            "Problem: {problem}\n\n"
            "Solution: {solution}\n\n"
            "{answer_text}"
            "Analyze each step and identify any errors or issues. Focus on mathematical correctness, not just the final answer. "
            "First, provide your overall verdict as 'Correct' or 'Incorrect', then explain your analysis. "
            "If the solution is correct, explain why. If there are errors, point them out specifically."
        )
    
    def verify_solution(
        self,
        problem: str,
        solution: str,
        answer: Optional[str] = None,
        reference_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify a solution to a problem.
        
        Args:
            problem: Problem text
            solution: Solution text
            answer: Extracted answer (optional)
            reference_answer: Reference answer (optional)
            
        Returns:
            Dictionary with verification results
        """
        start_time = time.time()
        
        # Initialize results
        verification_results = {
            "verified": False,
            "confidence": 0.0,
            "explanation": "",
            "matches_reference": None,
            "verification_time": 0.0
        }
        
        # Extract answer if not provided
        if answer is None:
            answer = extract_final_answer(solution)
            verification_results["extracted_answer"] = answer
        
        # Check against reference answer if provided
        if reference_answer is not None and answer is not None:
            matches = compare_answers(answer, reference_answer)
            verification_results["matches_reference"] = matches
            
            # If answers match exactly, we can say it's verified with high confidence
            if matches:
                verification_results["verified"] = True
                verification_results["confidence"] = 0.95
                verification_results["explanation"] = f"The answer '{answer}' matches the reference answer '{reference_answer}'."
                verification_results["verification_time"] = time.time() - start_time
                return verification_results
        
        # Use verification model if available
        if self.model is not None:
            # Choose verification mode
            if self.enable_step_by_step:
                template = self.step_verification_template
            else:
                template = self.verification_template
            
            # Add answer text if available
            answer_text = ""
            if answer:
                answer_text = f"Proposed answer: {answer}\n\n"
            if reference_answer:
                answer_text += f"Reference answer: {reference_answer}\n\n"
            
            # Format prompt
            prompt = template.format(
                problem=problem,
                solution=solution,
                answer_text=answer_text
            )
            
            # Generate verification
            try:
                verification_text = self.model.generate(
                    prompt=prompt,
                    max_new_tokens=self.max_tokens,
                    temperature=0.0,  # Use deterministic generation for verification
                    do_sample=False
                )
                
                # Extract verification result
                verification_results["raw_verification"] = verification_text
                
                # Check if solution is verified
                verification_lower = verification_text.lower()
                if "correct" in verification_lower and "incorrect" not in verification_lower:
                    verification_results["verified"] = True
                    verification_results["confidence"] = self._extract_confidence(verification_text)
                    verification_results["explanation"] = verification_text
                else:
                    verification_results["verified"] = False
                    verification_results["confidence"] = 1.0 - self._extract_confidence(verification_text)
                    verification_results["explanation"] = verification_text
                
            except Exception as e:
                logging.error(f"Verification failed: {str(e)}")
                verification_results["error"] = str(e)
        
        # Record verification time
        verification_results["verification_time"] = time.time() - start_time
        
        return verification_results
    
    def _extract_confidence(self, verification_text: str) -> float:
        """Extract confidence from verification text.
        
        Args:
            verification_text: Verification text
            
        Returns:
            Confidence score (0-1)
        """
        # Look for explicit confidence statements
        import re
        confidence_patterns = [
            r"confidence:?\s*(\d+)%",
            r"(\d+)%\s*confiden(t|ce)",
            r"confiden(t|ce)\s*[:|is]\s*(\d+)%"
        ]
        
        for pattern in confidence_patterns:
            matches = re.search(pattern, verification_text.lower())
            if matches:
                confidence_str = matches.group(1)
                try:
                    confidence = float(confidence_str) / 100.0
                    # Clamp to [0, 1]
                    return max(0.0, min(1.0, confidence))
                except (ValueError, IndexError):
                    pass
        
        # Look for confidence keywords
        confidence_keywords = {
            "certain": 0.95,
            "very confident": 0.9,
            "highly confident": 0.9,
            "confident": 0.8,
            "likely": 0.7,
            "probably": 0.6,
            "possibly": 0.5,
            "uncertain": 0.4,
            "doubtful": 0.3,
            "unlikely": 0.2,
            "very unlikely": 0.1,
            "incorrect": 0.0
        }
        
        for keyword, conf_value in confidence_keywords.items():
            if keyword in verification_text.lower():
                return conf_value
        
        # Default confidence based on verification result
        if "correct" in verification_text.lower() and "incorrect" not in verification_text.lower():
            return 0.8  # Default confidence for correct
        else:
            return 0.2  # Default confidence for incorrect
    
    def batch_verify(
        self,
        problems: List[str],
        solutions: List[str],
        answers: Optional[List[str]] = None,
        reference_answers: Optional[List[str]] = None,
        batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """Verify a batch of solutions.
        
        Args:
            problems: List of problem texts
            solutions: List of solution texts
            answers: List of extracted answers (optional)
            reference_answers: List of reference answers (optional)
            batch_size: Batch size for processing
            
        Returns:
            List of verification results
        """
        # Check input lengths
        if len(problems) != len(solutions):
            raise ValueError(f"Number of problems ({len(problems)}) and solutions ({len(solutions)}) must match")
        
        if answers is not None and len(problems) != len(answers):
            raise ValueError(f"Number of problems ({len(problems)}) and answers ({len(answers)}) must match")
        
        if reference_answers is not None and len(problems) != len(reference_answers):
            raise ValueError(f"Number of problems ({len(problems)}) and reference answers ({len(reference_answers)}) must match")
        
        # Prepare empty answers list if not provided
        if answers is None:
            answers = [None] * len(problems)
        
        # Prepare empty reference answers list if not provided
        if reference_answers is None:
            reference_answers = [None] * len(problems)
        
        # Verify in batches
        all_results = []
        total_problems = len(problems)
        num_batches = (total_problems + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            # Get batch data
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_problems)
            
            batch_problems = problems[start_idx:end_idx]
            batch_solutions = solutions[start_idx:end_idx]
            batch_answers = answers[start_idx:end_idx]
            batch_references = reference_answers[start_idx:end_idx]
            
            # Process each problem in the batch
            batch_results = []
            
            for i, (problem, solution, answer, reference) in enumerate(
                zip(batch_problems, batch_solutions, batch_answers, batch_references)
            ):
                result = self.verify_solution(
                    problem=problem,
                    solution=solution,
                    answer=answer,
                    reference_answer=reference
                )
                
                batch_results.append(result)
                all_results.append(result)
                
                # Log progress
                problem_idx = start_idx + i
                logging.debug(
                    f"Verified problem {problem_idx+1}/{total_problems} "
                    f"(Batch {batch_idx+1}/{num_batches}, Item {i+1}/{len(batch_problems)}): "
                    f"{'✓' if result['verified'] else '✗'} ({result['confidence']:.2f})"
                )
            
            # Log batch statistics
            verified_count = sum(1 for r in batch_results if r.get("verified", False))
            batch_accuracy = verified_count / len(batch_results)
            
            logging.info(
                f"Batch {batch_idx+1}/{num_batches}: "
                f"Verified {verified_count}/{len(batch_results)} "
                f"({batch_accuracy:.2%})"
            )
        
        # Calculate overall statistics
        verified_count = sum(1 for r in all_results if r.get("verified", False))
        accuracy = verified_count / total_problems
        
        logging.info(
            f"Overall: Verified {verified_count}/{total_problems} "
            f"({accuracy:.2%})"
        )
        
        return all_results
