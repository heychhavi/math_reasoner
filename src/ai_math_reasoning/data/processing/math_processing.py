"""Processing utilities for mathematical text and problems."""

import re
import logging
import string
from typing import Dict, List, Optional, Tuple, Union, Any
import difflib
import unicodedata
import fractions


def normalize_text(text: str) -> str:
    """Normalize text for comparison.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Normalize Unicode characters (e.g., convert "−" to "-")
    text = unicodedata.normalize('NFKD', text)
    
    # Replace common mathematical symbols
    replacements = {
        '×': '*',  # multiplication sign
        '÷': '/',  # division sign
        '−': '-',  # minus sign
        '=': ' = ',  # equals sign with spaces
        '+': ' + ',  # plus sign with spaces
        '/': ' / ',  # division with spaces
        '*': ' * ',  # multiplication with spaces
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove punctuation except for digits, operators, and decimal points
    allowed_chars = set(string.digits + '+-*/=.,()')
    text = ''.join(c for c in text if c in allowed_chars or c.isspace())
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def extract_final_answer(solution: str) -> Optional[str]:
    """Extract the final answer from a solution text.
    
    Looks for patterns like "Answer: X" or "The answer is X" or "Therefore, X"
    
    Args:
        solution: Solution text
        
    Returns:
        Extracted answer or None if not found
    """
    if not solution:
        return None
    
    # Try different patterns for answer extraction
    patterns = [
        r"(?:^|\n)(?:answer|final answer|the answer is)[:\s]+([^\n\.]+)",
        r"(?:^|\n)(?:answer|final answer|the answer is)[:\s]+([0-9\.\-\+\/\*\s\(\)]+)",
        r"(?:therefore|thus|hence|so)[,\s]+(the answer is|we get|we have)[:\s]+([^\n\.]+)",
        r"(?:therefore|thus|hence|so)[,\s]+([^\n\.]+) is the (?:final )?answer",
        r"(?:^|\n)([0-9\.\-\+\/\*\s\(\)]+) is the (?:final )?answer",
    ]
    
    # Check for "Answer:" line, which is the most common format
    lines = solution.lower().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('answer:') or line.startswith('final answer:'):
            return line[line.index(':') + 1:].strip()
    
    # Try to match patterns
    for pattern in patterns:
        matches = re.finditer(pattern, solution.lower(), re.IGNORECASE)
        for match in matches:
            groups = match.groups()
            if len(groups) == 1:
                return groups[0].strip()
            elif len(groups) > 1 and groups[-1]:
                return groups[-1].strip()
    
    # If no clear answer pattern, take the last line or last sentence
    lines = [line.strip() for line in solution.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        
        # Check if the last line is very short (likely just the answer)
        if len(last_line) < 30 and any(c.isdigit() for c in last_line):
            return last_line
        
        # Otherwise, try to take the last sentence
        sentences = last_line.split('.')
        if sentences:
            last_sentence = sentences[-1].strip()
            if last_sentence and len(last_sentence) < 50:
                return last_sentence
    
    # No answer found
    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison.
    
    Args:
        answer: Answer string
        
    Returns:
        Normalized answer
    """
    if not answer:
        return ""
    
    # Convert to lowercase and normalize whitespace
    normalized = normalize_text(answer)
    
    # Handle "The answer is X" format
    answer_prefixes = ["the answer is ", "answer is ", "answer: ", "final answer: "]
    for prefix in answer_prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    
    # Remove trailing punctuation
    normalized = normalized.rstrip(".,:;")
    
    # Replace common words in answers
    replacements = {
        "x equals": "",
        "y equals": "",
        "equal to": "",
        "equals": "",
        "equal": "",
        "is": "",
        "approximately": "",
        "approximate": "",
        "about": "",
        "around": "",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    
    # Try to extract numerical values and fractions
    return normalized.strip()


def parse_number(text: str) -> Optional[float]:
    """Parse a number from text.
    
    Args:
        text: Text representation of a number
        
    Returns:
        Parsed number as float or None if parsing fails
    """
    try:
        # Try direct conversion to float
        return float(text)
    except ValueError:
        pass
    
    # Try parsing fractions
    try:
        if '/' in text:
            num, denom = text.split('/', 1)
            return float(num.strip()) / float(denom.strip())
    except (ValueError, ZeroDivisionError):
        pass
    
    # Try handling mixed numbers (e.g., "1 2/3")
    try:
        if ' ' in text and '/' in text:
            parts = text.split()
            if len(parts) == 2 and '/' in parts[1]:
                whole = float(parts[0])
                num, denom = parts[1].split('/', 1)
                frac = float(num) / float(denom)
                return whole + frac if whole >= 0 else whole - frac
    except (ValueError, ZeroDivisionError, IndexError):
        pass
    
    return None


def compare_answers(answer1: Optional[str], answer2: Optional[str], tolerance: float = 1e-6) -> bool:
    """Compare two mathematical answers for equality.
    
    Args:
        answer1: First answer
        answer2: Second answer
        tolerance: Tolerance for numerical equality
        
    Returns:
        True if answers are equivalent, False otherwise
    """
    if answer1 is None or answer2 is None:
        return False
    
    # Normalize answers
    norm1 = normalize_answer(answer1)
    norm2 = normalize_answer(answer2)
    
    # Direct string comparison after normalization
    if norm1 == norm2:
        return True
    
    # Try to parse as numbers for numerical comparison
    num1 = parse_number(norm1)
    num2 = parse_number(norm2)
    
    if num1 is not None and num2 is not None:
        # Check numerical equality with tolerance
        return abs(num1 - num2) < tolerance
    
    # Check for fraction equivalence
    try:
        if '/' in norm1 and '/' in norm2:
            frac1 = fractions.Fraction(norm1)
            frac2 = fractions.Fraction(norm2)
            return frac1 == frac2
    except (ValueError, ZeroDivisionError):
        pass
    
    # Check for close string similarity
    similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()
    if similarity > 0.9:
        return True
    
    return False


def convert_to_latex(expression: str) -> str:
    """Convert a mathematical expression to LaTeX format.
    
    Args:
        expression: Mathematical expression
        
    Returns:
        LaTeX formatted expression
    """
    latex = expression
    
    # Replace fractions
    frac_pattern = r'(\d+)\s*/\s*(\d+)'
    latex = re.sub(frac_pattern, r'\\frac{\1}{\2}', latex)
    
    # Replace exponents
    exp_pattern = r'(\d+)\s*\^\s*(\d+)'
    latex = re.sub(exp_pattern, r'\1^{\2}', latex)
    
    # Replace square roots
    sqrt_pattern = r'sqrt\s*\(\s*([^)]+)\s*\)'
    latex = re.sub(sqrt_pattern, r'\\sqrt{\1}', latex)
    
    return latex


def extract_math_expressions(text: str) -> List[str]:
    """Extract mathematical expressions from text.
    
    Args:
        text: Text containing mathematical expressions
        
    Returns:
        List of extracted expressions
    """
    expressions = []
    
    # Find equations with = sign
    eq_pattern = r'[^=]+=[^=]+'
    for match in re.finditer(eq_pattern, text):
        expressions.append(match.group(0).strip())
    
    # Find expressions with mathematical operators
    op_pattern = r'[\d\.\s\+\-\*\/\(\)]{5,}'
    for match in re.finditer(op_pattern, text):
        expr = match.group(0).strip()
        if any(op in expr for op in "+-*/") and any(c.isdigit() for c in expr):
            expressions.append(expr)
    
    return expressions


def analyze_solution_steps(solution: str) -> Dict[str, Any]:
    """Analyze the steps in a mathematical solution.
    
    Args:
        solution: Solution text
        
    Returns:
        Dictionary with analysis results
    """
    # Split into steps (assuming each step is a paragraph or line)
    lines = [line.strip() for line in solution.split('\n') if line.strip()]
    
    # Get mathematical expressions in each step
    expressions = []
    for line in lines:
        step_expressions = extract_math_expressions(line)
        if step_expressions:
            expressions.extend(step_expressions)
    
    # Extract the final answer
    answer = extract_final_answer(solution)
    
    # Count the number of steps
    step_count = sum(1 for line in lines if re.search(r'^step\s+\d+:', line.lower()))
    if step_count == 0:
        # Alternative counting method if steps aren't explicitly labeled
        step_count = len([line for line in lines if any(marker in line.lower() 
                                              for marker in ['first', 'second', 'third', 'next', 'then', 'finally'])])
    
    # Analyze for presence of reasoning
    has_reasoning = any(marker in solution.lower() for marker in 
                       ['because', 'since', 'therefore', 'thus', 'hence', 'so,', 'implies', 'meaning'])
    
    return {
        "step_count": max(step_count, len(lines)),  # At least as many steps as lines
        "expressions": expressions,
        "answer": answer,
        "has_reasoning": has_reasoning,
        "length": len(solution),
    }


def format_problem_with_template(
    problem: str, 
    template: Optional[str] = None
) -> str:
    """Format a math problem using a template.
    
    Args:
        problem: Math problem text
        template: Template for formatting (with {problem} placeholder)
        
    Returns:
        Formatted problem
    """
    if template is None:
        template = (
            "Solve the following mathematical problem step by step, showing all your work. "
            "End your solution with 'Answer: [your final answer]'.\n\n"
            "Problem: {problem}\n\n"
            "Solution:"
        )
    
    return template.format(problem=problem)


def format_prompt_for_verification(
    problem: str,
    solution: str,
    answer: Optional[str] = None,
    template: Optional[str] = None
) -> str:
    """Format a prompt for solution verification.
    
    Args:
        problem: Math problem text
        solution: Proposed solution text
        answer: Extracted answer (optional)
        template: Template for formatting
        
    Returns:
        Formatted verification prompt
    """
    if template is None:
        template = (
            "You are tasked with verifying a solution to a mathematical problem. "
            "Carefully check if the solution is correct and leads to the right answer.\n\n"
            "Problem: {problem}\n\n"
            "Solution: {solution}\n\n"
            "{answer_text}"
            "Is this solution correct? Answer only with 'Correct' or 'Incorrect', "
            "followed by a brief explanation of your verification process."
        )
    
    answer_text = ""
    if answer:
        answer_text = f"Proposed answer: {answer}\n\n"
    
    return template.format(
        problem=problem,
        solution=solution,
        answer_text=answer_text
    )
