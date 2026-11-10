"""Stateful directed workflow graph engine supporting conditional branching and streaming."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from polyagent.graph.node import ConditionalEdge, ConditionFunc, Node, NodeFunc
from polyagent.graph.state import State

START = "__start__"
END = "__end__"

logger = logging.getLogger("polyagent.graph")


class WorkflowGraph:
    """Stateful directed workflow graph supporting branching, cycles, and event streaming."""

    def __init__(self, name: str = "workflow") -> None:
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, str] = {}
        self.conditional_edges: dict[str, ConditionalEdge] = {}
        self.entry_point: str | None = None

    def add_node(self, name: str, action: NodeFunc) -> WorkflowGraph:
        """Register a named executable node."""
        if name in (START, END):
            raise ValueError(f"Cannot name a node '{name}'")
        self.nodes[name] = Node(name, action)
        return self

    def set_entry_point(self, node_name: str) -> WorkflowGraph:
        """Set the entry node where execution starts."""
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' must be added before setting as entry point")
        self.entry_point = node_name
        return self

    def add_edge(self, source: str, target: str) -> WorkflowGraph:
        """Add a direct transition edge from source node to target or END."""
        if source == START:
            self.set_entry_point(target)
            return self
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' does not exist")
        if target != END and target not in self.nodes:
            raise ValueError(f"Target node '{target}' does not exist")
        self.edges[source] = target
        return self

    def add_conditional_edges(self, source: str, condition: ConditionFunc) -> WorkflowGraph:
        """Attach a conditional router function to source node."""
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' does not exist")
        self.conditional_edges[source] = ConditionalEdge(source, condition)
        return self

    def validate(self) -> list[str]:
        """Validate graph structure and return human-readable diagnostics (empty when valid)."""
        errors: list[str] = []
        if not self.entry_point:
            errors.append("Entry point has not been set")
        elif self.entry_point not in self.nodes:
            errors.append(f"Entry point '{self.entry_point}' is not a registered node")
        for src, tgt in self.edges.items():
            if tgt != END and tgt not in self.nodes:
                errors.append(f"Edge target '{tgt}' from '{src}' does not exist")
        return errors

    async def run(self, initial_state: State, max_steps: int = 50) -> State:
        """Execute the graph to completion or max_steps."""
        final_state = initial_state
        async for state in self.stream(initial_state, max_steps=max_steps):
            final_state = state
        return final_state

    async def stream(
        self, initial_state: State, max_steps: int = 50
    ) -> AsyncGenerator[State, None]:
        """Yield intermediate states after each node step and a terminal summary state."""
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if not self.entry_point:
            raise ValueError("Entry point has not been set for the graph")

        current_node_name = self.entry_point
        state = initial_state
        step = 0

        while current_node_name != END and step < max_steps:
            node = self.nodes.get(current_node_name)
            if not node:
                raise RuntimeError(f"Node '{current_node_name}' not found during execution")

            logger.info("Executing step %d: [%s]", step + 1, current_node_name)
            state = await node.execute(state)
            yield state

            if current_node_name in self.conditional_edges:
                cond = self.conditional_edges[current_node_name]
                current_node_name = cond.evaluate(state)
            elif current_node_name in self.edges:
                current_node_name = self.edges[current_node_name]
            else:
                current_node_name = END

            step += 1

        if current_node_name == END:
            state.completed = True
        else:
            state.completed = False
            state.timed_out = True
            state.error = f"Workflow stopped after reaching max_steps ({max_steps})"
        yield state

    def to_schema(self) -> dict[str, Any]:
        """Export graph topology for frontend canvas visualization."""
        node_list = [{"id": key, "label": key} for key in self.nodes]
        edge_list = [
            {"source": src, "target": tgt, "type": "direct"} for src, tgt in self.edges.items()
        ]
        for src in self.conditional_edges:
            edge_list.append({"source": src, "target": "dynamic", "type": "conditional"})

        return {
            "name": self.name,
            "entry_point": self.entry_point,
            "nodes": node_list,
            "edges": edge_list,
        }
