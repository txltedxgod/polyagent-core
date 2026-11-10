import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from polyagent.agents.agent import Agent, AgentRole
from polyagent.api.server import app
from polyagent.graph.graph import END, WorkflowGraph
from polyagent.graph.state import State
from polyagent.memory.vector import VectorMemory
from polyagent.tools.sandbox import calculator_tool, python_repl_tool


@pytest.mark.asyncio
async def test_workflow_graph_execution() -> None:
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
    assert final_state.timed_out is False
    assert final_state.context["num"] == 10


@pytest.mark.asyncio
async def test_workflow_graph_cycle_hits_max_steps_without_marking_completed() -> None:
    async def loop_node(s: State) -> State:
        return s.add_message("assistant", "looping", name="looper")

    graph = WorkflowGraph("cycle_graph")
    graph.add_node("looper", loop_node)
    graph.set_entry_point("looper")
    graph.add_edge("looper", "looper")

    final_state = await graph.run(State(task="Cycle check"), max_steps=3)
    assert final_state.completed is False
    assert final_state.timed_out is True
    assert final_state.error is not None
    assert "max_steps" in final_state.error
    assert len(final_state.messages) == 3


@pytest.mark.asyncio
async def test_vector_memory_and_agent_tools() -> None:
    mem = VectorMemory(dim=64)
    first_id = mem.add("FastAPI is an async Python web framework", metadata={"topic": "web"})
    mem.add("Docker enables containerized reproducible deployments", metadata={"topic": "ops"})

    # Semantic deduplication returns existing id without duplicating storage
    dup_id = mem.add("FastAPI is an async Python web framework", metadata={"topic": "web"})
    assert dup_id == first_id
    assert len(mem.documents) == 2

    # Zero-norm custom vector must not cause NaN division
    zero_id = mem.add("zero placeholder", custom_embedding=np.zeros(64), deduplicate=False)

    results = mem.search("What is FastAPI web framework?", top_k=1)
    assert len(results) == 1
    assert "FastAPI" in results[0]["content"]
    assert results[0]["score"] > 0

    filtered = mem.search("container deployments", top_k=3, filters={"topic": "ops"})
    assert len(filtered) == 1
    assert "Docker" in filtered[0]["content"]
    assert mem.delete(zero_id) is True

    calc_res = await calculator_tool.execute(expression="25 * 4 + sqrt(100)")
    assert calc_res == "110.0"

    repl_res = await python_repl_tool.execute(code="print('Agent Sandbox OK')")
    assert "Agent Sandbox OK" in repl_res


@pytest.mark.asyncio
async def test_sandbox_blocks_rce_dunder_and_math_dos() -> None:
    blocked_import = await python_repl_tool.execute(code="import os\nprint(os.listdir('.'))")
    assert "Sandbox policy violation" in blocked_import

    blocked_dunder = await python_repl_tool.execute(
        code="print((1).__class__.__base__.__subclasses__())"
    )
    assert "Sandbox policy violation" in blocked_dunder

    blocked_open = await python_repl_tool.execute(code="open('/etc/passwd').read()")
    assert "Sandbox policy violation" in blocked_open

    blocked_pow = await calculator_tool.execute(expression="9 ** 5000")
    assert "Calculation error" in blocked_pow

    # Concurrent print calls remain isolated without clobbering global stdout
    outputs = await asyncio.gather(
        python_repl_tool.execute(code="print('worker_alpha')"),
        python_repl_tool.execute(code="print('worker_beta')"),
    )
    assert outputs[0].strip() == "worker_alpha"
    assert outputs[1].strip() == "worker_beta"


@pytest.mark.asyncio
async def test_multi_agent_pipeline() -> None:
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


def test_http_api_endpoints_and_dashboard_xss_hardening() -> None:
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    schema = client.get("/api/graph")
    assert schema.status_code == 200
    assert schema.json()["entry_point"] == "planner"

    run_resp = client.post("/api/run", json={"task": "Verify workflow execution"})
    assert run_resp.status_code == 200
    data = run_resp.json()
    assert data["completed"] is True
    assert len(data["messages"]) == 3

    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "logBox.innerHTML +=" not in dashboard.text
    assert "textContent = msg.content" in dashboard.text
