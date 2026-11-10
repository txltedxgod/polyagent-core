from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from polyagent.graph.state import State

NodeFunc = Callable[[State], Awaitable[State]] | Callable[[State], State]
ConditionFunc = Callable[[State], str]


class Node:
    """
    Executable node wrapper in the DAG.
    """
    def __init__(self, name: str, action: NodeFunc):
        self.name = name
        self.action = action

    async def execute(self, state: State) -> State:
        if inspect.iscoroutinefunction(self.action):
            return await self.action(state)
        return self.action(state)


class ConditionalEdge:
    """
    Conditional branch evaluator returning the next node's identifier.
    """
    def __init__(self, source: str, condition: ConditionFunc):
        self.source = source
        self.condition = condition

    def evaluate(self, state: State) -> str:
        return self.condition(state)
