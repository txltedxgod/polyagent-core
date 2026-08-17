# 🧠 PolyAgent Core

[![CI](https://github.com/txltedxgod/polyagent-core/actions/workflows/ci.yml/badge.svg)](https://github.com/txltedxgod/polyagent-core/actions)
[![Python Version](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**PolyAgent Core** is an enterprise-grade autonomous multi-agent orchestration framework and workflow engine written in modern Python. It features stateful Directed Acyclic Graph (DAG) task execution, hybrid vector episodic memory, secure tool sandboxes, and a real-time visual canvas studio.

---

## 🌟 Architectural Features

```
               ┌────────────────────────────────────────────────────────┐
               │                  WorkflowGraph (DAG)                   │
               │                                                        │
               │   [State] ──>  [ Planner Agent ]                       │
               │                     │                                  │
               │                     ▼                                  │
               │              [ Coder Agent ] ──> (Python REPL / Tools) │
               │                     │                                  │
               │                     ▼                                  │
               │              [ Critic Agent ] ──> [Vector Memory]      │
               │                     │                                  │
               │                     ▼                                  │
               │                 [ END ]                                │
               └────────────────────┬───────────────────────────────────┘
                                    │
                         [SSE Real-Time Stream]
                                    │
                                    ▼
                         [Interactive Web Studio]
```

- **Stateful Graph Execution:** Chain agents and functional nodes with conditional branching, cyclic feedback loops, and immutable state checkpoints.
- **Autonomous Agent Personas:** Built-in Planner, Researcher, Coder, and Critic roles with dynamic tool selection.
- **Sandboxed Tooling:** Secure Python REPL, Safe Mathematical Expression Evaluator, and Extensible Tool Decorators (`@tool`).
- **Episodic & Vector Memory:** Cosine-similarity dense vector indexing for persistent context retention across multi-step conversations.
- **Real-Time Streaming Studio:** FastAPI backend streaming execution events to an interactive dashboard via Server-Sent Events (SSE).

---

## 🚀 Quick Start

### 1. Run with Docker
```bash
git clone https://github.com/txltedxgod/polyagent-core.git
cd polyagent-core
docker compose up --build
```
Open **`http://localhost:8000`** in your browser.

### 2. Python Code Example
```python
import asyncio
from polyagent import WorkflowGraph, State, Agent, AgentRole, END
from polyagent.tools.sandbox import python_repl_tool

async def main():
    # 1. Define Agents
    planner = Agent("planner", role=AgentRole.PLANNER)
    coder = Agent("coder", role=AgentRole.CODER, tools=[python_repl_tool])

    # 2. Construct Workflow DAG
    graph = WorkflowGraph("Code-Synthesis")
    graph.add_node("planner", planner.invoke)
    graph.add_node("coder", coder.invoke)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", END)

    # 3. Execute
    initial_state = State(task="Generate a fast sorting algorithm with benchmark")
    final_state = await graph.run(initial_state)

    for msg in final_state.messages:
        print(f"[{msg.name}]: {msg.content}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🧪 Testing

```bash
pytest -v tests/
```

---

## 📄 License
Released under the [MIT License](LICENSE).
