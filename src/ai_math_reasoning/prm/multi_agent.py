"""Multi-agent reasoning for mathematical problem solving."""

import os
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from ai_math_reasoning.models import MathReasoningModel
from ai_math_reasoning.prm.reranker import PRMReranker
from ai_math_reasoning.prm.verifier import SolutionVerifier
from ai_math_reasoning.data.processing.math_processing import extract_final_answer, compare_answers


class MultiAgentReasoner:
    """Multi-agent system for mathematical reasoning.
    
    This system uses multiple agents to generate diverse solutions to
    mathematical problems, then uses a reranker to select the best solution.
    """
    
    def __init__(
        self,
        solver_model: MathReasoningModel,
        reranker: Optional[PRMReranker] = None,
        verifier: Optional[SolutionVerifier] = None,
        num_agents: int = 5,
        temperature_range: Tuple[float, float] = (0.3, 0.8),
        top_p_range: Tuple[float, float] = (0.8, 0.98),
        max_tokens_per_agent: int = 1024,
        prompt_template: Optional[str] = None,
        max_parallel_agents: int = 3,
        enable_reflection: bool = True,
        reflection_prompt: Optional[str] = None,
        refinement_rounds: int = 1
    ):
        """Initialize the multi-agent reasoner.
        
        Args:
            solver_model: Base language model for solving
            reranker: PRM reranker for scoring and ranking solutions (optional)
            verifier: Solution verifier for checking answers (optional)
            num_agents: Number of solver agents to use
            temperature_range: Range of temperatures to use for diverse generation
            top_p_range: Range of top_p values to use for diverse generation
            max_tokens_per_agent: Maximum tokens per agent response
            prompt_template: Template for formatting prompts (optional)
            max_parallel_agents: Maximum number of parallel agent executions
            enable_reflection: Whether to enable self-reflection
            reflection_prompt: Prompt template for reflection (optional)
            refinement_rounds: Number of refinement rounds
        """
        self.solver_model = solver_model
        self.reranker = reranker
        self.verifier = verifier
        self.num_agents = num_agents
        self.temperature_range = temperature_range
        self.top_p_range = top_p_range
        self.max_tokens_per_agent = max_tokens_per_agent
        self.max_parallel_agents = max_parallel_agents
        self.enable_reflection = enable_reflection
        self.refinement_rounds = refinement_rounds
        
        # Set default prompt template if not provided
        self.prompt_template = prompt_template or (
            "You are a mathematical problem-solving assistant who is very careful and thorough. "
            "Solve the following problem step by step, showing your reasoning. "
            "End your solution with 'Answer: [your final answer]'.\n\n"
            "Problem: {problem}\n\n"
            "Solution:"
        )
        
        # Set default reflection prompt if not provided
        self.reflection_prompt = reflection_prompt or (
            "You previously provided a solution to this mathematical problem:\n\n"
            "Problem: {problem}\n\n"
            "Your solution was:\n{solution}\n\n"
            "Review your solution for errors, oversights, or improvements. "
            "If you find any issues, provide an improved solution. "
            "If the solution is already correct, you can clarify your reasoning or make it more elegant.\n\n"
            "Reflection and improved solution:"
        )
    
    def _sample_generation_params(self) -> Tuple[float, float]:
        """Sample generation parameters for diverse solutions.
        
        Returns:
            Tuple of (temperature, top_p)
        """
        temperature = random.uniform(self.temperature_range[0], self.temperature_range[1])
        top_p = random.uniform(self.top_p_range[0], self.top_p_range[1])
        return temperature, top_p
    
    def _generate_solution(
        self,
        problem: str,
        agent_id: int
    ) -> Dict[str, Any]:
        """Generate a solution using a solver agent.
        
        Args:
            problem: Mathematical problem
            agent_id: Agent identifier
            
        Returns:
            Dictionary with solution information
        """
        start_time = time.time()
        
        # Format prompt
        prompt = self.prompt_template.format(problem=problem)
        
        # Sample generation parameters
        temperature, top_p = self._sample_generation_params()
        
        # Generate solution
        try:
            solution = self.solver_model.generate(
                prompt=prompt,
                max_new_tokens=self.max_tokens_per_agent,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
            
            # Extract answer
            answer = extract_final_answer(solution)
            
            result = {
                "agent_id": agent_id,
                "solution": solution,
                "answer": answer,
                "temperature": temperature,
                "top_p": top_p,
                "generation_time": time.time() - start_time,
                "success": True
            }
            
        except Exception as e:
            # Log and record failure
            logging.error(f"Agent {agent_id} failed: {str(e)}")
            
            result = {
                "agent_id": agent_id,
                "solution": "",
                "answer": None,
                "temperature": temperature,
                "top_p": top_p,
                "generation_time": time.time() - start_time,
                "success": False,
                "error": str(e)
            }
        
        return result
    
    def _reflect_and_improve(
        self,
        problem: str,
        solution: str,
        agent_id: int
    ) -> Dict[str, Any]:
        """Reflect on and improve a solution.
        
        Args:
            problem: Mathematical problem
            solution: Initial solution
            agent_id: Agent identifier
            
        Returns:
            Dictionary with improved solution information
        """
        start_time = time.time()
        
        # Format reflection prompt
        prompt = self.reflection_prompt.format(problem=problem, solution=solution)
        
        # Use lower temperature for reflection to be more focused
        temperature = min(self.temperature_range) * 0.8
        
        # Generate improved solution
        try:
            improved_solution = self.solver_model.generate(
                prompt=prompt,
                max_new_tokens=self.max_tokens_per_agent,
                temperature=temperature,
                top_p=self.top_p_range[1],  # Use high top_p for reflection
                do_sample=True
            )
            
            # Extract answer from improved solution
            answer = extract_final_answer(improved_solution)
            
            result = {
                "agent_id": agent_id,
                "original_solution": solution,
                "improved_solution": improved_solution,
                "answer": answer,
                "reflection_time": time.time() - start_time,
                "success": True
            }
            
        except Exception as e:
            # Log and record failure
            logging.error(f"Reflection for agent {agent_id} failed: {str(e)}")
            
            result = {
                "agent_id": agent_id,
                "original_solution": solution,
                "improved_solution": None,
                "answer": None,
                "reflection_time": time.time() - start_time,
                "success": False,
                "error": str(e)
            }
        
        return result
    
    def solve_problem(
        self,
        problem: str,
        return_all_solutions: bool = False,
        reference_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Solve a mathematical problem using multi-agent reasoning.
        
        Args:
            problem: Mathematical problem
            return_all_solutions: Whether to return all solutions or just the best
            reference_answer: Reference answer for evaluation (optional)
            
        Returns:
            Dictionary with solution results
        """
        start_time = time.time()
        
        # Track solutions
        all_solutions = []
        
        # Step 1: Generate initial solutions in parallel
        with ThreadPoolExecutor(max_workers=self.max_parallel_agents) as executor:
            # Submit solution generation tasks
            future_to_agent = {
                executor.submit(self._generate_solution, problem, i): i
                for i in range(self.num_agents)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_agent):
                agent_id = future_to_agent[future]
                result = future.result()
                
                if result["success"]:
                    all_solutions.append(result)
        
        # Step 2: Apply reflection and refinement if enabled
        if self.enable_reflection and self.refinement_rounds > 0:
            refined_solutions = []
            
            # Apply multiple rounds of refinement
            for round_idx in range(self.refinement_rounds):
                with ThreadPoolExecutor(max_workers=self.max_parallel_agents) as executor:
                    # Submit refinement tasks
                    future_to_solution = {}
                    
                    for sol_idx, solution_data in enumerate(all_solutions):
                        if solution_data["success"] and solution_data["solution"]:
                            # Determine which solution to refine
                            if round_idx == 0:
                                # First round: refine original solutions
                                solution_to_refine = solution_data["solution"]
                            else:
                                # Subsequent rounds: refine previously refined solutions
                                solution_to_refine = solution_data.get("improved_solution") or solution_data["solution"]
                            
                            # Submit refinement task
                            future = executor.submit(
                                self._reflect_and_improve,
                                problem,
                                solution_to_refine,
                                solution_data["agent_id"]
                            )
                            future_to_solution[future] = sol_idx
                    
                    # Process refinement results
                    for future in as_completed(future_to_solution):
                        sol_idx = future_to_solution[future]
                        refinement_result = future.result()
                        
                        if refinement_result["success"]:
                            # Update solution with refinement results
                            original_data = all_solutions[sol_idx]
                            
                            refined_data = {
                                **original_data,
                                "improved_solution": refinement_result["improved_solution"],
                                "answer": refinement_result["answer"] or original_data["answer"],
                                "reflection_time": refinement_result["reflection_time"],
                                "has_improvement": True
                            }
                            
                            # Use improved solution for the next round
                            all_solutions[sol_idx] = refined_data
                            refined_solutions.append(refined_data)
            
            # Add solutions that weren't refined
            for solution_data in all_solutions:
                if "has_improvement" not in solution_data:
                    refined_solutions.append(solution_data)
            
            # Update all_solutions with refined solutions
            all_solutions = refined_solutions
        
        # Step 3: Score and rank solutions
        if self.reranker is not None:
            solutions_text = []
            for solution_data in all_solutions:
                # Use improved solution if available, otherwise use original
                solution_text = solution_data.get("improved_solution") or solution_data["solution"]
                solutions_text.append(solution_text)
            
            # Perform ranking
            ranked_results = self.reranker.rank_solutions(
                problem=problem,
                solutions=solutions_text,
                normalize=True,
                add_analysis=True
            )
            
            # Merge ranking information with solution data
            for i, (rank_result, solution_data) in enumerate(zip(ranked_results, all_solutions)):
                solution_data["score"] = rank_result["score"]
                solution_data["rank"] = rank_result["rank"]
                solution_data["normalized_score"] = rank_result.get("normalized_score", 0.0)
                solution_data["analysis"] = rank_result.get("analysis")
            
            # Sort solutions by rank
            all_solutions.sort(key=lambda x: x.get("rank", float("inf")))
        else:
            # No reranker, add placeholder ranks based on order
            for i, solution_data in enumerate(all_solutions):
                solution_data["rank"] = i + 1
                solution_data["score"] = 0.0
                solution_data["normalized_score"] = 0.0
        
        # Step 4: Verify best solution if verifier is available
        best_solution = all_solutions[0] if all_solutions else None
        
        if best_solution and self.verifier is not None:
            best_solution_text = best_solution.get("improved_solution") or best_solution["solution"]
            answer = best_solution["answer"]
            
            verification_result = self.verifier.verify_solution(
                problem=problem,
                solution=best_solution_text,
                answer=answer,
                reference_answer=reference_answer
            )
            
            # Add verification results to solution data
            best_solution["verification"] = verification_result
        
        # Step 5: Assemble final result
        total_time = time.time() - start_time
        
        result = {
            "problem": problem,
            "best_solution": best_solution,
            "all_solutions": all_solutions if return_all_solutions else None,
            "num_solutions": len(all_solutions),
            "total_time": total_time,
            "reference_answer": reference_answer
        }
        
        # Add accuracy if reference answer is provided
        if reference_answer is not None:
            # Check if best solution is correct
            correct = False
            
            if best_solution and best_solution["answer"]:
                correct = compare_answers(best_solution["answer"], reference_answer)
            
            result["correct"] = correct
            
            # Check how many solutions got the correct answer
            correct_solutions = sum(
                1 for sol in all_solutions
                if sol["answer"] and compare_answers(sol["answer"], reference_answer)
            )
            
            result["num_correct"] = correct_solutions
            result["accuracy"] = correct_solutions / len(all_solutions) if all_solutions else 0.0
        
        return result
    
    def solve_batch(
        self,
        problems: List[str],
        reference_answers: Optional[List[str]] = None,
        return_all_solutions: bool = False,
        batch_size: int = 8,
        output_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Solve a batch of problems.
        
        Args:
            problems: List of mathematical problems
            reference_answers: List of reference answers (optional)
            return_all_solutions: Whether to return all solutions or just the best
            batch_size: Batch size for processing
            output_path: Path to save results (optional)
            
        Returns:
            List of solution results
        """
        # Create results directory if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Solve problems in batches
        all_results = []
        total_problems = len(problems)
        num_batches = (total_problems + batch_size - 1) // batch_size
        
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            # Get batch problems
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_problems)
            batch_problems = problems[start_idx:end_idx]
            
            # Get batch reference answers if available
            batch_answers = None
            if reference_answers:
                batch_answers = reference_answers[start_idx:end_idx]
            
            # Process each problem in the batch
            batch_results = []
            
            for i, problem in enumerate(batch_problems):
                # Get reference answer if available
                ref_answer = batch_answers[i] if batch_answers else None
                
                # Solve problem
                result = self.solve_problem(
                    problem=problem,
                    return_all_solutions=return_all_solutions,
                    reference_answer=ref_answer
                )
                
                batch_results.append(result)
                all_results.append(result)
                
                # Log progress
                problem_idx = start_idx + i
                logging.info(
                    f"Solved problem {problem_idx+1}/{total_problems} "
                    f"(Batch {batch_idx+1}/{num_batches}, Problem {i+1}/{len(batch_problems)})"
                )
                
                if ref_answer:
                    correct = result.get("correct", False)
                    logging.info(f"  - Correct: {correct}")
            
            # Save batch results if output path is provided
            if output_path:
                interim_path = f"{os.path.splitext(output_path)[0]}_batch{batch_idx+1}.json"
                with open(interim_path, "w") as f:
                    json.dump(batch_results, f, indent=2)
                
                logging.info(f"Saved batch {batch_idx+1} results to {interim_path}")
            
            # Log batch statistics
            if batch_answers:
                batch_correct = sum(1 for r in batch_results if r.get("correct", False))
                batch_accuracy = batch_correct / len(batch_results)
                logging.info(
                    f"Batch {batch_idx+1} accuracy: {batch_correct}/{len(batch_results)} ({batch_accuracy:.2%})"
                )
        
        # Calculate overall statistics
        total_time = time.time() - start_time
        avg_time_per_problem = total_time / total_problems
        
        statistics = {
            "total_problems": total_problems,
            "total_time": total_time,
            "avg_time_per_problem": avg_time_per_problem
        }
        
        # Add accuracy statistics if reference answers are provided
        if reference_answers:
            correct_count = sum(1 for r in all_results if r.get("correct", False))
            accuracy = correct_count / total_problems
            
            statistics["correct_count"] = correct_count
            statistics["accuracy"] = accuracy
            
            logging.info(
                f"Overall accuracy: {correct_count}/{total_problems} ({accuracy:.2%})"
            )
        
        logging.info(
            f"Processed {total_problems} problems in {total_time:.2f}s "
            f"({avg_time_per_problem:.2f}s per problem)"
        )
        
        # Save final results if output path is provided
        if output_path:
            final_results = {
                "results": all_results,
                "statistics": statistics
            }
            
            with open(output_path, "w") as f:
                json.dump(final_results, f, indent=2)
            
            logging.info(f"Saved all results to {output_path}")
        
        return all_results
