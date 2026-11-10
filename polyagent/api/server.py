from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from polyagent.agents.agent import Agent, AgentRole
from polyagent.graph.graph import END, WorkflowGraph
from polyagent.graph.state import State
from polyagent.memory.vector import VectorMemory
from polyagent.tools.sandbox import calculator_tool, python_repl_tool

app = FastAPI(
    title="PolyAgent Studio",
    description="Autonomous Multi-Agent Workflow Engine API",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize sample workspace workflow
memory = VectorMemory()
memory.add("NexusMQ is a high throughput commit log message broker written in Go.")
memory.add("CRDTs enable lock-free collaborative editing across distributed network nodes.")

planner = Agent("planner_agent", role=AgentRole.PLANNER, memory=memory)
coder = Agent("coder_agent", role=AgentRole.CODER, tools=[python_repl_tool, calculator_tool], memory=memory)
critic = Agent("critic_agent", role=AgentRole.CRITIC, memory=memory)

graph = WorkflowGraph(name="Autonomous-Software-Pipeline")
graph.add_node("planner", planner.invoke)
graph.add_node("coder", coder.invoke)
graph.add_node("critic", critic.invoke)

graph.set_entry_point("planner")
graph.add_edge("planner", "coder")
graph.add_edge("coder", "critic")
graph.add_edge("critic", END)


class RunTaskRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=5000)
    context: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "workflow": graph.name}


@app.get("/api/graph")
async def get_graph_schema():
    return graph.to_schema()


@app.post("/api/run")
async def run_task(req: RunTaskRequest):
    initial_state = State(task=req.task, context=req.context)
    final_state = await graph.run(initial_state)
    return final_state.model_dump()


@app.get("/api/stream")
async def stream_task(task: str):
    initial_state = State(task=task)

    async def event_generator():
        async for state in graph.stream(initial_state):
            payload = json.dumps(state.model_dump(), default=str)
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PolyAgent — Multi-Agent Studio</title>
    <style>
        :root {
            --bg: #0b0f19;
            --panel: #131b2e;
            --border: #1e293b;
            --accent: #6366f1;
            --accent-light: #818cf8;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --success: #10b981;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
        .logo { font-size: 24px; font-weight: 800; background: linear-gradient(135deg, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .container { display: grid; grid-template-columns: 350px 1fr; gap: 24px; }
        .sidebar { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
        .main { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
        input, button { width: 100%; padding: 12px; border-radius: 8px; font-size: 14px; }
        input { background: #0f172a; border: 1px solid var(--border); color: #fff; margin-bottom: 12px; }
        button { background: var(--accent); color: #fff; border: none; font-weight: 700; cursor: pointer; transition: 0.2s; }
        button:hover { background: #4f46e5; }
        .flow-node { background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 14px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
        .flow-node.active { border-color: var(--accent-light); box-shadow: 0 0 15px rgba(99, 102, 241, 0.3); }
        .log-box { background: #090d16; border: 1px solid var(--border); border-radius: 8px; padding: 16px; min-height: 400px; max-height: 600px; overflow-y: auto; font-family: monospace; font-size: 13px; line-height: 1.6; }
        .msg { margin-bottom: 16px; padding-bottom: 16px; border-bottom: 1px solid #1e293b; }
        .msg-role { color: var(--accent-light); font-weight: bold; margin-bottom: 4px; }
        .tool-chip { display: inline-block; background: rgba(16,185,129,0.2); color: var(--success); padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-top: 4px; }
    </style>
</head>
<body>
    <header>
        <div class="logo">🧠 PolyAgent Core <span style="font-size: 14px; color: var(--text-dim); font-weight: normal;">DAG Multi-Agent Studio</span></div>
        <div style="color: var(--text-dim); font-size: 14px;">Status: <span style="color: var(--success);">Online</span></div>
    </header>

    <div class="container">
        <div class="sidebar">
            <h3 style="margin-bottom: 16px;">Trigger Workflow</h3>
            <input id="taskInput" type="text" value="Build high-concurrency event streamer with tests" />
            <button onclick="runWorkflow()">Stream Agents Execution</button>

            <h3 style="margin-top: 28px; margin-bottom: 16px;">Active DAG Nodes</h3>
            <div class="flow-node" id="node-planner">
                <div><strong>1. Planner</strong><div style="font-size:12px; color:var(--text-dim)">Deconstructs task</div></div>
                <span class="tool-chip">Memory</span>
            </div>
            <div class="flow-node" id="node-coder">
                <div><strong>2. Coder</strong><div style="font-size:12px; color:var(--text-dim)">Executes sandbox</div></div>
                <span class="tool-chip">REPL</span>
            </div>
            <div class="flow-node" id="node-critic">
                <div><strong>3. Critic</strong><div style="font-size:12px; color:var(--text-dim)">Validates quality</div></div>
                <span class="tool-chip">Judge</span>
            </div>
        </div>

        <div class="main">
            <h2>Real-Time Multi-Agent Thought & Execution Stream</h2>
            <div class="log-box" id="logs">
                <div style="color: var(--text-dim)">Waiting for task execution...</div>
            </div>
        </div>
    </div>

    <script>
        function runWorkflow() {
            const task = document.getElementById('taskInput').value;
            const logBox = document.getElementById('logs');
            logBox.innerHTML = '<div style="color: var(--accent-light)">[INIT] Connecting to PolyAgent Event Stream...</div>';

            const evtSource = new EventSource('/api/stream?task=' + encodeURIComponent(task));

            evtSource.onmessage = function(event) {
                const state = JSON.parse(event.data);
                logBox.textContent = '';
                state.messages.forEach(msg => {
                    const card = document.createElement('div'); card.className = 'msg'; const roleEl = document.createElement('div'); roleEl.className = 'msg-role'; roleEl.textContent = (msg.name || msg.role).toUpperCase(); const bodyEl = document.createElement('div'); bodyEl.style.whiteSpace = 'pre-wrap'; bodyEl.textContent = msg.content; card.appendChild(roleEl); card.appendChild(bodyEl); (msg.tool_calls || []).forEach(t => { const chip = document.createElement('div'); chip.className = 'tool-chip'; chip.textContent = 'Executed: ' + t.tool; card.appendChild(chip); });
                    logBox.appendChild(card);
                });
                logBox.scrollTop = logBox.scrollHeight;

                if (state.completed) {
                    const doneEl = document.createElement('div'); doneEl.style.cssText = 'color: var(--success); font-weight:bold; margin-top:12px;'; doneEl.textContent = 'DAG Workflow Completed Successfully!'; logBox.appendChild(doneEl);
                    evtSource.close();
                }
            };

            evtSource.onerror = function() {
                evtSource.close();
            };
        }
    </script>
</body>
</html>
"""
