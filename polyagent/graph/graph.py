from __future__ import annotations
from typing import Dict, List, Optional, Callable, AsyncGenerator, Any
import asyncio
import logging
from polyagent.graph.state import State
from polyagent.graph.node import Node, ConditionalEdge, NodeFunc, ConditionFunc

START = "__start__"
END = "__end__"

logger = logging.getLogger("polyagent.graph")


class WorkflowGraph:
    """
    Stateful directed workflow graph supporting branching, cycles, tool loops,
    and event streaming.
    """
    def __init__(self, name: str = "workflow"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, ConditionalEdge] = {}
        self.entry_point: Optional[str] = None

    def add_node(self, name: str, action: NodeFunc) -> WorkflowGraph:
        if name in [START, END]:
            raise ValueError(f"Cannot name a node '{name}'")
        self.nodes[name] = Node(name, action)
        return self

    def set_entry_point(self, node_name: str) -> WorkflowGraph:
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' must be added before setting as entry point")
        self.entry_point = node_name
        return self

    def add_edge(self, source: str, target: str) -> WorkflowGraph:
        if source == START:
            self.entry_point = target
            return self
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' does not exist")
        if target != END and target not in self.nodes:
            raise ValueError(f"Target node '{target}' does not exist")
        self.edges[source] = target
        return self

    def add_conditional_edges(self, source: str, condition: ConditionFunc) -> WorkflowGraph:
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' does not exist")
        self.conditional_edges[source] = ConditionalEdge(source, condition)
        return self

    async def run(self, initial_state: State, max_steps: int = 50) -> State:
        """
        Executes the graph to completion.
        """
        async for state in self.stream(initial_state, max_steps):
            pass
        return state

    async def stream(self, initial_state: State, max_steps: int = 50) -> AsyncGenerator[State, None]:
        """
        Streams intermediate states after each node execution step.
        """
        if not self.entry_point:
            raise ValueError("Entry point has not been set for the graph")

        current_node_name = self.entry_point
        state = initial_state
        step = 0

        while current_node_name != END and step < max_steps:
            node = self.nodes.get(current_node_name)
            if not node:
                raise RuntimeError(f"Node '{current_node_name}' not found during execution")

            logger.info(f"Executing step {step + 1}: [{current_node_name}]")
            state = await node.execute(state)
            yield state

            # Resolve next node
            if current_node_name in self.conditional_edges:
                cond = self.conditional_edges[current_node_name]
                current_node_name = cond.evaluate(state)
            elif current_node_name in self.edges:
                current_node_name = self.edges[current_node_name]
            else:
                current_node_name = END

            step += 1

        state.completed = True
        yield state

    def to_schema(self) -> Dict[str, Any]:
        """
        Exports graph topology for frontend canvas visualization.
        """
        node_list = [{"id": k, "label": k} for k in self.nodes.keys()]
        edge_list = []
        for src, tgt in self.edges.items():
            edge_list.append({"source": src, "target": tgt, "type": "direct"})
        for src, cond in self.conditional_edges.items():
            edge_list.append({"source": src, "target": "dynamic", "type": "conditional"})

        return {
            "name": self.name,
            "entry_point": self.entry_point,
            "nodes": node_list,
            "edges": edge_list,
        }
