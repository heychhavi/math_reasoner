#!/usr/bin/env python
"""
Example script for simple inference with mathematical reasoning models.

This script demonstrates how to load a model and use it to solve
mathematical problems with various inference options.
"""

import os
import argparse
import logging
from typing import List, Optional

from ai_math_reasoning.models import create_model
from ai_math_reasoning.data.processing.math_processing import (
    format_problem_with_template,
    extract_final_answer,
    analyze_solution_steps
)
from ai_math_reasoning.prm.multi_agent import MultiAgentReasoner
from ai_math_reasoning.prm.reranker import PRMReranker


def solve_problem(
    problem: str,
    model_type: str = "auto",
    model_name: str = "deepseek-ai/deepseek-math-7b-instruct",
    use_multi_agent: bool = False,
    num_agents: int = 3,
    prompt_template: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    load_in_8bit: bool = False,
    use_flash_attention: bool = False
):
    """Solve a mathematical problem using a model.
    
    Args:
        problem: Mathematical problem to solve
        model_type: Type of model ("auto", "deepseek", "qwen", "transformer")
        model_name: Model name or path
        use_multi_agent: Whether to use multi-agent reasoning
        num_agents: Number of agents for multi-agent reasoning
        prompt_template: Custom prompt template (optional)
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p sampling parameter
        load_in_8bit: Whether to load the model in 8-bit precision
        use_flash_attention: Whether to use flash attention
        
    Returns:
        Generated solution
    """
    logging.info(f"Loading model: {model_name}")
    
    # Create model
    model = create_model(
        model_type=model_type,
        model_name_or_path=model_name,
        load_in_8bit=load_in_8bit,
        use_flash_attention=use_flash_attention
    )
    
    # Print model info
    logging.info(f"Loaded model: {model.model_name_or_path}")
    
    if use_multi_agent:
        logging.info(f"Using multi-agent reasoning with {num_agents} agents")
        
        # Create reranker
        reranker = PRMReranker(
            base_model=model,
            inference_only=True
        )
        
        # Create multi-agent reasoner
        reasoner = MultiAgentReasoner(
            solver_model=model,
            reranker=reranker,
            num_agents=num_agents,
            prompt_template=prompt_template,
            max_tokens_per_agent=max_tokens
        )
        
        # Solve problem
        result = reasoner.solve_problem(problem)
        
        # Extract best solution
        if result and result.get("best_solution"):
            solution = result["best_solution"].get("improved_solution") or result["best_solution"]["solution"]
            
            # Print additional info
            score = result["best_solution"].get("score", 0.0)
            rank = result["best_solution"].get("rank", 1)
            answer = result["best_solution"].get("answer", "")
            
            logging.info(f"Best solution (rank {rank}, score {score:.2f})")
            if answer:
                logging.info(f"Extracted answer: {answer}")
            
            return solution
        else:
            logging.error("No valid solution found")
            return "No solution found."
    else:
        # Format problem with template
        formatted_problem = format_problem_with_template(
            problem=problem,
            template=prompt_template
        )
        
        # Generate solution
        logging.info("Generating solution...")
        solution = model.generate(
            prompt=formatted_problem,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0)
        )
        
        # Extract answer
        answer = extract_final_answer(solution)
        if answer:
            logging.info(f"Extracted answer: {answer}")
        
        # Analyze solution
        analysis = analyze_solution_steps(solution)
        if analysis:
            logging.info(f"Steps: {analysis['step_count']}, Has reasoning: {analysis['has_reasoning']}")
        
        return solution


def solve_multiple_problems(
    problems: List[str],
    **kwargs
):
    """Solve multiple problems.
    
    Args:
        problems: List of mathematical problems
        **kwargs: Additional arguments for solve_problem
        
    Returns:
        List of solutions
    """
    solutions = []
    
    for i, problem in enumerate(problems):
        logging.info(f"Problem {i+1}/{len(problems)}:")
        logging.info(problem)
        
        solution = solve_problem(problem, **kwargs)
        solutions.append(solution)
        
        logging.info("\nSolution:")
        logging.info(solution)
        logging.info("-" * 80)
    
    return solutions


def main():
    """Main function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Solve mathematical problems")
    parser.add_argument("--problem", type=str, help="Mathematical problem to solve")
    parser.add_argument("--problems_file", type=str, help="File containing problems (one per line)")
    parser.add_argument("--model_type", type=str, default="auto", help="Model type (auto, deepseek, qwen, transformer)")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-math-7b-instruct", help="Model name or path")
    parser.add_argument("--use_multi_agent", action="store_true", help="Use multi-agent reasoning")
    parser.add_argument("--num_agents", type=int, default=3, help="Number of agents for multi-agent reasoning")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--use_flash_attention", action="store_true", help="Use flash attention")
    parser.add_argument("--output_file", type=str, help="Output file for solutions")
    args = parser.parse_args()
    
    # Ensure we have a problem to solve
    problems = []
    
    if args.problem:
        problems = [args.problem]
    elif args.problems_file:
        with open(args.problems_file, "r") as f:
            problems = [line.strip() for line in f if line.strip()]
    else:
        # Example problems if none provided
        problems = [
            "If 3x + 5y = 15 and 2x - 3y = 13, find the value of x + y.",
            "Find all real solutions to the equation x^4 - 5x^2 + 4 = 0.",
            "In a geometric sequence, the first term is 6 and the third term is 24. Find the second term."
        ]
        logging.info("No problem specified, using example problems")
    
    # Solve problems
    solutions = solve_multiple_problems(
        problems=problems,
        model_type=args.model_type,
        model_name=args.model_name,
        use_multi_agent=args.use_multi_agent,
        num_agents=args.num_agents,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        load_in_8bit=args.load_in_8bit,
        use_flash_attention=args.use_flash_attention
    )
    
    # Save solutions if output file is specified
    if args.output_file:
        with open(args.output_file, "w") as f:
            for i, (problem, solution) in enumerate(zip(problems, solutions)):
                f.write(f"Problem {i+1}:\n{problem}\n\nSolution:\n{solution}\n\n")
                f.write("-" * 80 + "\n\n")
        
        logging.info(f"Solutions saved to {args.output_file}")


if __name__ == "__main__":
    main()
