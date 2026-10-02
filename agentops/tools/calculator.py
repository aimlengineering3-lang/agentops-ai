"""A deterministic arithmetic evaluator: the "Python/Data Analysis Tool" from the
master doc (S5), scoped down to what the Analyst actually needs -- verifying a sum,
a product, or a budget range the model claims in a finding.

This is NOT a general-purpose code execution tool. It never calls Python's eval() or
exec(). It walks a parsed expression's AST and allows only numeric literals and the
arithmetic operators below; anything else (names, calls, attributes, subscripts,
comprehensions, strings) is rejected before it can run. That is the whole safety
story: there is no sandbox to escape because there is no execution, only arithmetic.
"""

import ast
import operator

_MAX_EXPR_LENGTH = 200
_MAX_MAGNITUDE = 1e15  # guards against absurd/DoS-y results (e.g. runaway powers)

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalculatorError(ValueError):
    """The expression was unsafe, malformed, or produced a non-finite/huge result."""


def safe_eval(expression: str) -> float:
    """Evaluate a pure arithmetic expression (+ - * / // % ** and parentheses only).

    Raises CalculatorError for anything else -- a name, a function call, a string, an
    import, an absurd magnitude -- rather than guessing at the person's intent.
    """
    if not expression or len(expression) > _MAX_EXPR_LENGTH:
        raise CalculatorError(f"expression must be 1-{_MAX_EXPR_LENGTH} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"not a valid expression: {exc}") from exc
    result = _eval_node(tree.body)
    if not isinstance(result, int | float) or result != result or abs(result) == float("inf"):
        raise CalculatorError("expression did not produce a finite number")
    if abs(result) > _MAX_MAGNITUDE:
        raise CalculatorError(f"result exceeds the allowed magnitude ({_MAX_MAGNITUDE:g})")
    return float(result)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise CalculatorError(f"only numeric literals are allowed, got {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"operator not allowed: {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if type(node.op) is ast.Pow and abs(right) > 12:
            raise CalculatorError("exponent too large")  # blocks e.g. 9**9**9 blowing up
        try:
            return op(left, right)
        except ZeroDivisionError as exc:
            raise CalculatorError("division by zero") from exc
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"operator not allowed: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise CalculatorError(f"expression contains a disallowed construct: {type(node).__name__}")
