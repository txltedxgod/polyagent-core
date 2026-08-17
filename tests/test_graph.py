import pytest
import asyncio
from polyagent.graph.state import State
from polyagent.graph.graph import WorkflowGraph, END, START
from polyagent.agents.agent import Agent, AgentRole
from polyagent.tools.sandbox import python_repl_tool, calculator_tool
from polyagent.memory.vector import VectorMemory


@pytest.mark.asyncio
async def test_workflow_graph_execution():
    state = State(task="Generate Fibonacci sequence")
    
    async def step_one(s: State) -> State:
        return s.add_message("assistant", "Step 1 complete", name="step1").set_context("num", 10)

    async def step_two(s: State) -> State:
        num = s.context.get("num", 0)
        return s.add_message("assistant", f"Step 2 processed {num * 2}", name="step2")

    graph = WorkflowGraph("test_graph")
    graph.add_node("step1", step_one)
    graph.add_node("step2", step_two)
    graph.set_entry_point("step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", END)

    final_state = await graph.run(state)
    assert len(final_state.messages) == 2
    assert final_state.completed is True
    assert final_state.context["num"] == 10


@pytest.mark.asyncio
async def test_vector_memory_and_agent_tools():
    mem = VectorMemory(dim=64)
    mem.add("FastAPI is an async Python web framework")
    mem.add("Docker enables containerized reproducible deployments")

    results = mem.search("What is FastAPI?", top_k=1)
    assert len(results) == 1
    assert "FastAPI" in results[0]["content"]

    # Test tool execution
    calc_res = await calculator_tool.execute(expression="25 * 4 + 10")
    assert calc_res == "110"

    repl_res = await python_repl_tool.execute(code="print('Agent Sandbox OK')")
    assert "Agent Sandbox OK" in repl_res


@pytest.mark.asyncio
async def test_multi_agent_pipeline():
    memory = VectorMemory()
    planner = Agent("planner", role=AgentRole.PLANNER, memory=memory)
    coder = Agent("coder", role=AgentRole.CODER, tools=[python_repl_tool])

    graph = WorkflowGraph("agent_pipe")
    graph.add_node("planner", planner.invoke)
    graph.add_node("coder", coder.invoke)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", END)

    state = State(task="Calculate factorial of 5")
    final_state = await graph.run(state)

    assert len(final_state.messages) == 2
    assert "plan" in final_state.context
    assert "code_result" in final_state.context
