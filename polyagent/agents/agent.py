from __future__ import annotations
from typing import List, Dict, Any, Optional
from enum import Enum
import asyncio
from polyagent.graph.state import State
from polyagent.tools.base import Tool
from polyagent.memory.vector import VectorMemory


class AgentRole(str, Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    CODER = "coder"
    CRITIC = "critic"
    GENERAL = "general"


class Agent:
    """
    Autonomous agent with specialized instructions, tool capabilities,
    and memory retrieval.
    """
    def __init__(
        self,
        name: str,
        role: AgentRole = AgentRole.GENERAL,
        system_prompt: str = "",
        tools: Optional[List[Tool]] = None,
        memory: Optional[VectorMemory] = None,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt or f"You are an expert {role.value} agent."
        self.tools = {t.name: t for t in tools or []}
        self.memory = memory

    async def invoke(self, state: State) -> State:
        """
        Executes reasoning cycle, tool invocation, and state mutation.
        """
        # 1. Retrieve relevant memory if available
        retrieved_context = ""
        if self.memory:
            docs = self.memory.search(state.task, top_k=2)
            if docs:
                retrieved_context = "\nRelevant Knowledge:\n" + "\n".join([f"- {d['content']}" for d in docs])

        # 2. Simulate reasoning output (or call LLM)
        thought = f"[{self.name}] Analyzing task '{state.task}' for role '{self.role.value}'."
        if retrieved_context:
            thought += f" {retrieved_context}"

        # 3. Simulate tool actions based on agent role
        output_content = ""
        tool_records = []

        if self.role == AgentRole.PLANNER:
            output_content = (
                f"Generated Plan:\n"
                f"1. Research background and dependencies for '{state.task}'.\n"
                f"2. Implement core algorithmic solution.\n"
                f"3. Run automated tests and static analysis."
            )
            state = state.set_context("plan", output_content)

        elif self.role == AgentRole.CODER and "python_repl" in self.tools:
            tool = self.tools["python_repl"]
            sample_code = "result = [x**2 for x in range(1, 6)]\nprint(f'Computed squares: {result}')"
            tool_res = await tool.execute(code=sample_code)
            tool_records.append({"tool": "python_repl", "input": sample_code, "output": tool_res})
            output_content = f"Implemented code and verified execution:\nOutput: {tool_res}"
            state = state.set_context("code_result", tool_res)

        elif self.role == AgentRole.CRITIC:
            plan = state.context.get("plan", "No plan")
            output_content = f"Review completed. Plan quality score: 9.8/10. Ready for final release."
            state = state.set_context("review_passed", True)

        else:
            output_content = f"Execution step finished for {self.name}."

        # Update state message history
        state = state.add_message(
            role="assistant",
            content=f"{thought}\n\n{output_content}",
            name=self.name,
            tool_calls=tool_records if tool_records else None
        )

        return state
