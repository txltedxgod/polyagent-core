from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, create_model


class Tool:
    """
    Standard tool interface with argument validation and metadata description.
    """
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        args_schema: type[BaseModel] | None = None,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema or self._generate_schema()

    def _generate_schema(self) -> type[BaseModel]:
        sig = inspect.signature(self.func)
        fields = {}
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            default = ... if param.default == inspect.Parameter.empty else param.default
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            fields[param.name] = (param_type, default)
        return create_model(f"{self.name}_Args", **fields)

    async def execute(self, **kwargs) -> Any:
        # Validate schema
        validated = self.args_schema(**kwargs)
        if inspect.iscoroutinefunction(self.func):
            return await self.func(**validated.model_dump())
        return self.func(**validated.model_dump())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema(),
        }


def tool(name: str | None = None, description: str | None = None):
    """Decorator to define a tool from a python function."""
    def decorator(fn: Callable[..., Any]) -> Tool:
        tool_name = name or fn.__name__
        tool_desc = description or fn.__doc__ or "No description provided."
        return Tool(name=tool_name, description=tool_desc.strip(), func=fn)
    return decorator
