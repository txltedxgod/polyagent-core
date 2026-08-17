from __future__ import annotations
import sys
import io
import traceback
from polyagent.tools.base import tool


@tool(name="python_repl", description="Executes isolated Python code and captures standard output.")
def python_repl_tool(code: str) -> str:
    """Executes python code in a constrained namespace and returns stdout/result."""
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    namespace = {}
    try:
        exec(code, namespace)
        output = buffer.getvalue()
        return output if output else "Code executed successfully with no stdout."
    except Exception:
        return f"Error during execution:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout


@tool(name="calculator", description="Computes mathematical expressions safely.")
def calculator_tool(expression: str) -> str:
    """Evaluates arithmetic expressions."""
    try:
        # Restricted safe eval
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of '{name}' is not allowed")
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


@tool(name="text_summarizer", description="Summarizes key information from long text.")
def text_summarizer_tool(text: str, max_sentences: int = 2) -> str:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return ". ".join(sentences[:max_sentences]) + ("." if sentences else "")
