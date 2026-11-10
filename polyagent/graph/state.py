"""Immutable workflow state container passed across WorkflowGraph nodes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Single conversation or execution trace entry."""

    role: str
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class State(BaseModel):
    """State container passed between workflow nodes with immutable delta helpers."""

    task: str
    messages: list[Message] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    current_step: int = 0
    completed: bool = False
    timed_out: bool = False
    error: str | None = None

    def add_message(
        self,
        role: str,
        content: str,
        name: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> State:
        """Return a deep copy of State with a newly appended message."""
        new_state = self.model_copy(deep=True)
        new_state.messages.append(
            Message(role=role, content=content, name=name, tool_calls=tool_calls)
        )
        new_state.current_step += 1
        return new_state

    def set_context(self, key: str, value: Any) -> State:
        """Return a deep copy of State with a single context key updated."""
        new_state = self.model_copy(deep=True)
        new_state.context[key] = value
        return new_state

    def update_context(self, data: dict[str, Any]) -> State:
        """Return a deep copy of State with multiple context keys merged."""
        new_state = self.model_copy(deep=True)
        new_state.context.update(data)
        return new_state

    def last_message(self) -> Message | None:
        """Return the most recent message or None when empty."""
        return self.messages[-1] if self.messages else None
