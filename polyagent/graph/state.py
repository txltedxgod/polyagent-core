from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import copy


class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class State(BaseModel):
    """
    State container passed between workflow nodes. Supports immutable cloning
    and delta updates with history checkpointing.
    """
    task: str
    messages: List[Message] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    current_step: int = 0
    completed: bool = False
    error: Optional[str] = None

    def add_message(self, role: str, content: str, name: Optional[str] = None, tool_calls: Optional[List[Dict[str, Any]]] = None) -> State:
        new_state = self.model_copy(deep=True)
        new_state.messages.append(Message(role=role, content=content, name=name, tool_calls=tool_calls))
        new_state.current_step += 1
        return new_state

    def set_context(self, key: str, value: Any) -> State:
        new_state = self.model_copy(deep=True)
        new_state.context[key] = value
        return new_state

    def update_context(self, data: Dict[str, Any]) -> State:
        new_state = self.model_copy(deep=True)
        new_state.context.update(data)
        return new_state

    def last_message(self) -> Optional[Message]:
        return self.messages[-1] if self.messages else None
