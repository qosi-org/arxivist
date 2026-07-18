"""Countdown task verifier.

STUB NOTE: the paper does not specify its exact verifier implementation — see sir.json
ambiguities. This implementation checks: (1) the completion contains a valid arithmetic
expression using +, -, *, / and parentheses, (2) it uses exactly the given numbers, each once,
(3) it evaluates to the target.

Security note: the model's completion is untrusted text. We parse it with `ast` and only
evaluate a strict whitelist of arithmetic AST nodes — never call eval()/exec() on model output.
"""
from __future__ import annotations

import ast
import operator
import re
from collections import Counter

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


class UnsafeExpressionError(ValueError):
    pass


def _safe_eval(node) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    raise UnsafeExpressionError(f"disallowed expression node: {type(node).__name__}")


def _extract_expression(completion: str) -> str | None:
    match = re.search(r"[-+*/()\d.\s]{3,}", completion)
    return match.group(0).strip() if match else None


def _extract_numbers_used(expr: str) -> list[int]:
    return [int(tok) for tok in re.findall(r"\d+", expr)]


class CountdownVerifier:
    """Score = 1.0 if the completion is a valid expression using exactly the given numbers
    once each and evaluating to the target (within floating-point tolerance); else 0.0."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def score(self, numbers: list[int], target: int, completion: str) -> float:
        expr_str = _extract_expression(completion)
        if expr_str is None:
            return 0.0

        used = _extract_numbers_used(expr_str)
        if Counter(used) != Counter(numbers):
            return 0.0

        try:
            tree = ast.parse(expr_str, mode="eval")
            value = _safe_eval(tree)
        except (SyntaxError, UnsafeExpressionError, ZeroDivisionError):
            return 0.0

        return 1.0 if abs(value - target) < self.tolerance else 0.0
