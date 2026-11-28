"""Calculator tool for ForgeAI."""

import ast
import math
import operator
from typing import Callable

_MAX_RESULT = 10**12
_BINARY_OPERATORS: dict[type[ast.operator], Callable[[int | float, int | float], int | float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _checked_number(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Only real numeric values are allowed")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite numbers are not allowed")
    if abs(value) > _MAX_RESULT:
        raise ValueError("Result magnitude must not exceed 10**12")
    return value


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        return _checked_number(node.value)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _checked_number(-_evaluate(node.operand))

    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Only +, -, *, /, //, %, and ** operators are allowed")
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and right > 100:
            raise ValueError("Exponent must not exceed 100")
        return _checked_number(operation(left, right))

    raise ValueError(
        "Only numbers, parentheses, unary minus, and basic arithmetic are allowed"
    )


def calculator(expression: str) -> dict[str, object]:
    """Safely evaluate an arithmetic expression."""
    try:
        parsed = ast.parse(expression, mode="eval")
        return {"result": _evaluate(parsed.body)}
    except (SyntaxError, ValueError, ArithmeticError, OverflowError) as error:
        return {"error": f"Invalid arithmetic expression: {error}"}
