"""Sandboxed tool implementations with strict AST validation and bounded evaluation."""

from __future__ import annotations

import ast
import io
import math
import operator
from typing import Any

from polyagent.tools.base import tool

_MAX_AST_DEPTH = 32
_MAX_EXPONENT = 1000
_MAX_REPL_CHARS = 8000
_FORBIDDEN_CALL_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "hasattr",
        "input",
        "locals",
        "memoryview",
        "open",
        "setattr",
        "vars",
    }
)
_FORBIDDEN_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.AsyncFor,
    ast.Await,
    ast.With,
    ast.ClassDef,
    ast.Try,
)

_SAFE_BUILTINS: dict[str, Any] = {
    "False": False,
    "None": None,
    "True": True,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_CALC_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}


def _check_ast_safety(node: ast.AST, depth: int = 0) -> None:
    """Recursively reject unsafe Python AST constructs and dunder accesses."""
    if depth > _MAX_AST_DEPTH:
        raise ValueError("Code AST nesting exceeds allowed limit")

    if isinstance(node, _FORBIDDEN_AST_NODES):
        raise ValueError(f"Disallowed Python construct: {type(node).__name__}")

    if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
        raise ValueError(f"Access to private/dunder attribute '{node.attr}' is prohibited")

    if isinstance(node, ast.Name) and (
        node.id.startswith("__") or node.id in _FORBIDDEN_CALL_NAMES
    ):
        raise ValueError(f"Access to restricted identifier '{node.id}' is prohibited")

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALL_NAMES:
            raise ValueError(f"Call to '{node.func.id}' is prohibited")

    for child in ast.iter_child_nodes(node):
        _check_ast_safety(child, depth + 1)


def _eval_math_node(node: ast.AST, depth: int = 0) -> float | int:
    """Safely evaluate a parsed mathematical AST without calling eval()."""
    if depth > _MAX_AST_DEPTH:
        raise ValueError("Expression nesting exceeds allowed limit")

    if isinstance(node, ast.Expression):
        return _eval_math_node(node.body, depth + 1)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise ValueError("Boolean literals are not supported in arithmetic expressions")
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_math_node(node.operand, depth + 1))

    if isinstance(node, ast.BinOp):
        left = _eval_math_node(node.left, depth + 1)
        right = _eval_math_node(node.right, depth + 1)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_EXPONENT:
                raise ValueError(f"Exponent {right} exceeds safe limit ({_MAX_EXPONENT})")
            return left**right
        op_fn = _BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        return op_fn(left, right)

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _CALC_FUNCS.get(node.func.id)
        if fn is None or node.keywords:
            raise ValueError(f"Unsupported function call: {node.func.id}")
        args = [_eval_math_node(arg, depth + 1) for arg in node.args]
        return fn(*args)

    raise ValueError(f"Disallowed syntax in expression: {type(node).__name__}")


@tool(name="python_repl", description="Executes isolated Python code and captures standard output.")
def python_repl_tool(code: str) -> str:
    """Execute AST-validated Python code in a restricted namespace without mutating global stdout."""
    if not code or len(code) > _MAX_REPL_CHARS:
        return "Sandbox policy violation: code must be between 1 and 8000 characters."

    try:
        tree = ast.parse(code, mode="exec")
        _check_ast_safety(tree)
    except (SyntaxError, ValueError) as exc:
        return f"Sandbox policy violation: {exc}"

    buffer = io.StringIO()

    def _sandbox_print(*args: Any, sep: str = " ", end: str = "\n", **_kwargs: Any) -> None:
        buffer.write(sep.join(str(a) for a in args) + end)

    restricted_builtins = {**_SAFE_BUILTINS, "print": _sandbox_print}
    namespace: dict[str, Any] = {"__builtins__": restricted_builtins}

    try:
        compiled = compile(tree, "<sandbox>", "exec")
        exec(compiled, namespace, namespace)  # noqa: S102 - AST & builtins constrained above
        output = buffer.getvalue()
        return output if output else "Code executed successfully with no stdout."
    except Exception as exc:
        return f"Error during execution: {type(exc).__name__}: {exc}"


@tool(name="calculator", description="Computes mathematical expressions safely.")
def calculator_tool(expression: str) -> str:
    """Evaluate bounded arithmetic expressions via AST traversal."""
    if not expression or len(expression) > 512:
        return "Calculation error: expression must be between 1 and 512 characters"
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_math_node(tree)
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool(name="text_summarizer", description="Summarizes key information from long text.")
def text_summarizer_tool(text: str, max_sentences: int = 2) -> str:
    """Return the leading sentences from input text."""
    if max_sentences <= 0:
        return ""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return ". ".join(sentences[:max_sentences]) + ("." if sentences else "")
