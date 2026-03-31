<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Context Isolation

Nexus supports per-request and per-task context isolation through hierarchical scope stacks. Each scope stack has its own root UUID, enabling safe concurrent and multi-tenant agent execution in a single process.

## Scope Stack

A `ScopeStack` is a vector of `ScopeHandle`s with an immovable root scope at index 0:

```
┌──────────────────────────────────┐
│  ScopeStack                      │
│  ┌────────────────────────────┐  │
│  │ [0] Root (Agent, auto-UUID)│  │  ← never removed
│  │ [1] Agent: orchestrator    │  │
│  │ [2] Tool: search           │  │  ← current top
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

- Created via `create_scope_stack()` — each stack gets a fresh root with a unique UUID
- Can never be empty; root scope cannot be removed
- Thread-safe: wrapped in `Arc<RwLock<ScopeStack>>`

## Storage: Two-Tier Lookup

Nexus uses a two-tier storage pattern for context isolation:

```
current_scope_stack()
  │
  ├── Try task-local (tokio::task_local!)
  │     └── Found? → return it
  │
  └── Fallback to thread-local (thread_local!)
        └── return default or explicitly-set stack
```

### Task-Local (Async)

Each tokio task can have its own scope stack via `TASK_SCOPE_STACK`:

```rust
let stack = create_scope_stack();
tokio::spawn(async move {
    TASK_SCOPE_STACK.scope(stack, async {
        // All scope operations use this stack
        task_scope_push(handle);
        tokio::task::yield_now().await;  // Yields without losing isolation
        task_scope_top()  // Still sees this stack
    }).await
});
```

### Thread-Local (Sync Fallback)

Each OS thread gets its own default scope stack automatically. Override with `set_thread_scope_stack()`:

```rust
let custom = create_scope_stack();
set_thread_scope_stack(custom);
// All sync scope operations now use custom stack
```

## API

### Core Functions

| Function | Purpose |
|----------|---------|
| `create_scope_stack()` | Create new isolated stack with fresh root |
| `current_scope_stack()` | Get current task/thread's stack |
| `set_thread_scope_stack(stack)` | Bind stack to current OS thread |
| `scope_stack_active()` | Check if an explicit scope stack is set |
| `propagate_scope_to_thread()` | Capture current stack for worker-thread propagation (Python) |

### Per-Language Patterns

**Python** — uses `contextvars.ContextVar` for async-safe isolation:

```python
import nat_nexus

async def handle_request():
    # Each asyncio task gets its own scope stack via get_scope_stack()
    # (lazily created, stored in a contextvars.ContextVar)
    nat_nexus.get_scope_stack()

    handle = nat_nexus.scope.push("agent", nat_nexus.ScopeType.Agent)
    # ... process request ...
    nat_nexus.scope.pop(handle)
```

Lazy initialization: `get_scope_stack()` creates a new stack on first access in a task.

Check whether a scope stack has been explicitly initialized (useful for integrations
that should only activate when the caller has set up Nexus):

```python
if nat_nexus.scope_stack_active():
    # Nexus is active — use the middleware pipeline
    ...
```

**Go** — uses `ScopeStack.Run()` which locks the goroutine to an OS thread:

```go
stack, _ := nat_nexus.NewScopeStack()
defer stack.Close()

go func() {
    stack.Run(func() {
        // Goroutine is pinned to an OS thread
        // All scope operations use this stack
        scope.Push("agent", scope.TypeAgent)
    })
}()
```

**Node.js** — explicit stack management:

```javascript
const stack = createScopeStack();
setThreadScopeStack(stack);
pushScope("agent", ScopeType.Agent, null, null);
// ... operations use this stack ...
```

**WASM** — single-threaded; scope stacks are manually passed:

```javascript
const stack = createScopeStack();
setThreadScopeStack(stack);
```

## Multi-Tenant Isolation

The standard pattern for isolating concurrent agents:

```mermaid
graph TD
    subgraph "Process"
        subgraph "Agent A (Stack A)"
            A_ROOT["Root (uuid-A)"]
            A_ROOT --> A1["Agent: planner"]
            A1 --> A2["Tool: search"]
        end

        subgraph "Agent B (Stack B)"
            B_ROOT["Root (uuid-B)"]
            B_ROOT --> B1["Agent: coder"]
            B1 --> B2["LLM: gpt-4"]
        end

        REG["Global Registries<br/>(guardrails, intercepts, subscribers)"]
    end
```

Each agent gets its own scope stack with a unique root UUID. **Global** middleware registrations are shared across all agents. Use `root_uuid` on events to filter per-agent in subscribers or ATIF export.

### Scope-Local Middleware for Per-Agent Isolation

For middleware that should only apply to a specific agent or session, use **scope-local registration** instead of global registration. Scope-local middleware is stored in the `ScopeStack` and only participates in pipeline execution for that stack:

```mermaid
graph TD
    subgraph "Process"
        subgraph "Agent A (Stack A)"
            A_ROOT["Root (uuid-A)"]
            A_ROOT --> A1["Agent: planner"]
            A_SL["Scope-local:<br/>pii_filter (priority=5)"]
        end

        subgraph "Agent B (Stack B)"
            B_ROOT["Root (uuid-B)"]
            B_ROOT --> B1["Agent: coder"]
            B_SL["Scope-local:<br/>code_validator (priority=5)"]
        end

        REG["Global Registries<br/>compliance_check (priority=1)"]
    end
```

In this setup:
- **Agent A** runs `compliance_check` (global) + `pii_filter` (scope-local) during tool/LLM calls
- **Agent B** runs `compliance_check` (global) + `code_validator` (scope-local) during tool/LLM calls
- Neither agent sees the other's scope-local middleware

```python
async def handle_agent(agent_id: str, guardrail_fn):
    stack = nat_nexus.create_scope_stack()
    nat_nexus._scope_stack_var.set(stack)

    handle = nat_nexus.scope.push(f"agent-{agent_id}", nat_nexus.ScopeType.Agent)

    # Register middleware only for this agent's scope
    nat_nexus.scope_local.register_tool_conditional_execution(
        handle, f"{agent_id}_guard", 10, guardrail_fn,
    )

    response = await nat_nexus.llm.execute("gpt-4", request, llm_func)
    nat_nexus.scope.pop(handle)  # guardrail automatically removed

async def main():
    # Global: applies to all agents
    nat_nexus.guardrails.register_llm_conditional_execution(
        "compliance", 1, compliance_check,
    )

    await asyncio.gather(
        handle_agent("alice", alice_guardrail),
        handle_agent("bob", bob_guardrail),
    )
```

See [Core Concepts: Scope-Local Middleware](concepts.md#scope-local-middleware) for full details.

### Async Example (Python)

```python
async def handle_agent(agent_id: str):
    # Isolated stack for this agent — get_scope_stack() creates one per task
    nat_nexus.get_scope_stack()

    handle = nat_nexus.scope.push(f"agent-{agent_id}", nat_nexus.ScopeType.Agent)
    response = await nat_nexus.llm.execute("gpt-4", request, llm_func)
    nat_nexus.scope.pop(handle)

# Concurrent agents — fully isolated
async def main():
    await asyncio.gather(
        handle_agent("alice"),
        handle_agent("bob"),
    )
```

### Sync Example (Go)

```go
var wg sync.WaitGroup

for _, agentID := range agents {
    wg.Add(1)
    go func(id string) {
        defer wg.Done()
        stack, _ := nat_nexus.NewScopeStack()
        defer stack.Close()

        stack.Run(func() {
            scope.Push("agent-"+id, scope.TypeAgent)
            // Process agent — isolated from other goroutines
        })
    }(agentID)
}
wg.Wait()
```

## ATIF Export with Root UUID Filtering

When exporting trajectories, pass `root_uuid` to isolate a single agent's events:

```python
exporter = AtifExporter()
# ... agents run concurrently ...

# Export only Agent A's trajectory
trajectory_a = exporter.export(root_uuid=agent_a_root_uuid)

# Export everything
trajectory_all = exporter.export(root_uuid=None)
```

See [ATIF Export](atif-export.md) for details.
