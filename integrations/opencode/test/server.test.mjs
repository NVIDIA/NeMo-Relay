// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict"
import { randomUUID } from "node:crypto"
import fs from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { describe, it } from "node:test"

import { createServerPlugin } from "../server.js"

function createFakeRuntime() {
  const events = []
  let counter = 0
  let activeStack = { id: "current" }

  return {
    events,
    ScopeType: { Agent: 0 },
    createScopeStack() {
      return { id: `stack-${++counter}` }
    },
    currentScopeStack() {
      return activeStack
    },
    setThreadScopeStack(stack) {
      activeStack = stack
    },
    pushScope(name, scopeType, _parent, attributes, data, metadata, input) {
      const handle = {
        uuid: randomUUID(),
        name,
        scopeType,
        attributes,
      }
      events.push({
        kind: "scope",
        category: "agent",
        scope_category: "start",
        stack: activeStack.id,
        uuid: handle.uuid,
        name,
        data,
        metadata,
        input,
      })
      return handle
    },
    popScope(handle, output) {
      events.push({
        kind: "scope",
        category: "agent",
        scope_category: "end",
        stack: activeStack.id,
        uuid: handle.uuid,
        name: handle.name,
        data: output,
      })
    },
    event(name, handle, data, metadata) {
      events.push({
        kind: "mark",
        stack: activeStack.id,
        uuid: randomUUID(),
        parent_uuid: handle?.uuid,
        name,
        data,
        metadata,
      })
    },
    toolCall(name, args, handle, _attributes, data, metadata, toolCallID) {
      const tool = {
        uuid: randomUUID(),
        name,
        parentUuid: handle?.uuid,
        toolCallID,
      }
      events.push({
        kind: "scope",
        category: "tool",
        scope_category: "start",
        stack: activeStack.id,
        uuid: tool.uuid,
        name,
        parent_uuid: handle?.uuid,
        data: args,
        metadata,
      })
      return tool
    },
    toolCallEnd(handle, result, data, metadata) {
      events.push({
        kind: "scope",
        category: "tool",
        scope_category: "end",
        stack: activeStack.id,
        uuid: handle.uuid,
        name: handle.name,
        data: result ?? data,
        metadata,
      })
    },
    llmCall(name, request, handle, _attributes, _data, metadata, modelName) {
      const llm = {
        uuid: randomUUID(),
        name,
        parentUuid: handle?.uuid,
        modelName,
      }
      events.push({
        kind: "scope",
        category: "llm",
        scope_category: "start",
        stack: activeStack.id,
        uuid: llm.uuid,
        name,
        parent_uuid: handle?.uuid,
        data: request,
        metadata,
        modelName,
      })
      return llm
    },
    llmCallEnd(handle, response, data, metadata) {
      events.push({
        kind: "scope",
        category: "llm",
        scope_category: "end",
        stack: activeStack.id,
        uuid: handle.uuid,
        name: handle.name,
        data: response ?? data,
        metadata,
        modelName: handle.modelName,
      })
    },
  }
}

function createFakePluginHost({ validateDiagnostics = [], initializeDiagnostics = [], onInitialize } = {}) {
  return {
    validateCalls: [],
    initializeCalls: [],
    clearCalls: 0,
    validate(config) {
      this.validateCalls.push(config)
      return { diagnostics: validateDiagnostics }
    },
    async initialize(config) {
      await onInitialize?.(config)
      this.initializeCalls.push(config)
      return { diagnostics: initializeDiagnostics }
    },
    clear() {
      this.clearCalls += 1
    },
  }
}

function createHarness(params = {}) {
  const runtime = params.runtime ?? createFakeRuntime()
  const pluginHost = params.pluginHost ?? createFakePluginHost(params)
  let cleanup
  const server = createServerPlugin({
    loadModules: async () => {
      if (params.loadError) throw params.loadError
      return { lib: runtime, pluginHost }
    },
    registerCleanup(close) {
      cleanup = close
      return () => {
        if (cleanup === close) cleanup = undefined
      }
    },
  })

  return {
    runtime,
    pluginHost,
    server,
    cleanup: async () => cleanup?.(),
  }
}

function pluginConfig() {
  return {
    version: 1,
    components: [
      {
        kind: "observability",
        enabled: true,
        config: {
          version: 1,
          atof: {
            enabled: true,
            output_directory: "./.nemoflow",
            filename: "opencode.atof.jsonl",
          },
          atif: {
            enabled: true,
            agent_name: "opencode",
            output_directory: "./.nemoflow",
            filename_template: "opencode-{session_id}.atif.json",
          },
        },
      },
    ],
  }
}

async function makeTempDir() {
  return fs.mkdtemp(path.join(os.tmpdir(), "nemo-flow-opencode-"))
}

async function waitForLogMatch(logPath, pattern) {
  let log = ""
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      log = await fs.readFile(logPath, "utf8")
      if (pattern.test(log)) return log
    } catch (error) {
      if (error.code !== "ENOENT") throw error
    }
    await new Promise((resolve) => setTimeout(resolve, 10))
  }
  assert.match(log, pattern)
  return log
}

async function statIsDirectory(targetPath) {
  try {
    return (await fs.stat(targetPath)).isDirectory()
  } catch {
    return false
  }
}

function eventNames(events) {
  return events.map((event) => event.name).filter(Boolean)
}

function assertSessionStackUsed(runtime) {
  const sessionStart = runtime.events.find(
    (event) => event.category === "agent" && event.scope_category === "start",
  )
  assert.ok(sessionStart?.stack?.startsWith("stack-"))
  for (const event of runtime.events) {
    assert.equal(event.stack, sessionStart.stack, `${event.name ?? event.category} should use session scope stack`)
  }
  assert.equal(runtime.currentScopeStack().id, "current")
}

describe("NeMo Flow OpenCode plugin", () => {
  it("initializes generic plugin config and records OpenCode hooks until idle", async () => {
    const dir = await makeTempDir()
    const { runtime, pluginHost, server } = createHarness()
    const hooks = await server(
      { directory: dir },
      {
        enabled: true,
        logPath: "./.nemoflow/opencode-plugin.log",
        plugins: pluginConfig(),
      },
    )

    const expectedOutputDirectory = path.join(dir, ".nemoflow")
    assert.equal(pluginHost.validateCalls.length, 1)
    assert.equal(pluginHost.initializeCalls.length, 1)
    assert.ok(await statIsDirectory(expectedOutputDirectory))
    assert.equal(pluginHost.validateCalls[0].components[0].config.atof.output_directory, expectedOutputDirectory)
    assert.equal(pluginHost.validateCalls[0].components[0].config.atif.output_directory, expectedOutputDirectory)
    assert.deepEqual(pluginHost.initializeCalls[0], pluginHost.validateCalls[0])

    await hooks.config?.({ model: "test-provider/test-model", agent: { build: {} } })
    await hooks["chat.message"]?.(
      {
        sessionID: "ses_1",
        agent: "build",
        model: { providerID: "test-provider", modelID: "test-model" },
        messageID: "msg_1",
      },
      {
        message: { id: "msg_1", role: "user", agent: "build" },
        parts: [{ id: "part_1", type: "text", text: "hello" }],
      },
    )
    await hooks["chat.params"]?.(
      {
        sessionID: "ses_1",
        agent: "build",
        model: { providerID: "test-provider", id: "test-model" },
        provider: { source: "config", options: {} },
        message: { id: "msg_1" },
      },
      { temperature: 0, topP: 1, topK: 0, options: {} },
    )
    await hooks.event?.({
      event: {
        id: "evt_assistant",
        type: "message.updated",
        properties: {
          sessionID: "ses_1",
          info: { id: "msg_assistant_1", role: "assistant", agent: "build", sessionID: "ses_1" },
        },
      },
    })
    await hooks.event?.({
      event: {
        id: "evt_tool_part",
        type: "message.part.updated",
        properties: {
          sessionID: "ses_1",
          part: {
            id: "part_tool_1",
            messageID: "msg_assistant_1",
            sessionID: "ses_1",
            type: "tool",
            tool: "write",
            callID: "call_1",
            state: { input: {}, status: "pending" },
          },
        },
      },
    })
    await hooks["tool.execute.before"]?.(
      { tool: "write", sessionID: "ses_1", callID: "call_1" },
      { args: { path: "phase1-demo.txt" } },
    )
    await hooks["tool.execute.after"]?.(
      { tool: "write", sessionID: "ses_1", callID: "call_1", args: { path: "phase1-demo.txt" } },
      { title: "Wrote file", output: "done", metadata: { ok: true } },
    )
    await hooks.event?.({
      event: {
        id: "evt_1",
        type: "session.status",
        properties: { sessionID: "ses_1", status: { type: "idle" } },
      },
    })
    await hooks.event?.({
      event: {
        id: "evt_1_duplicate",
        type: "session.status",
        properties: { sessionID: "ses_1", status: { type: "idle" } },
      },
    })

    const names = eventNames(runtime.events)
    assert.equal(names.includes("opencode.chat.message"), false)
    assert.equal(names.includes("opencode.llm.request"), false)
    assert.equal(names.filter((name) => name === "opencode.session.flush").length, 0)
    assert.equal(runtime.events.filter((event) => event.category === "llm" && event.scope_category === "start").length, 1)
    assert.equal(runtime.events.filter((event) => event.category === "llm" && event.scope_category === "end").length, 1)
    const llmEnd = runtime.events.find((event) => event.category === "llm" && event.scope_category === "end")
    assert.deepEqual(JSON.parse(llmEnd.data.tool_calls[0].function.arguments), { path: "phase1-demo.txt" })
    assert.equal(runtime.events.filter((event) => event.category === "tool" && event.scope_category === "start").length, 1)
    assert.equal(runtime.events.filter((event) => event.category === "tool" && event.scope_category === "end").length, 1)
    assert.equal(runtime.events.filter((event) => event.category === "agent" && event.scope_category === "end").length, 1)
    assertSessionStackUsed(runtime)
  })

  it("creates observability output directories before plugin host initialization", async () => {
    const dir = await makeTempDir()
    const expectedOutputDirectory = path.join(dir, ".nemoflow")
    const { server } = createHarness({
      async onInitialize() {
        assert.ok(await statIsDirectory(expectedOutputDirectory))
      },
    })

    await server({ directory: dir }, { enabled: true, plugins: pluginConfig() })
  })

  it("registers default observability output when plugins are omitted", async () => {
    const dir = await makeTempDir()
    const { pluginHost, server } = createHarness()
    const hooks = await server({ directory: dir })

    assert.equal(typeof hooks.event, "function")
    assert.equal(pluginHost.validateCalls.length, 1)
    assert.equal(pluginHost.initializeCalls.length, 1)
    const component = pluginHost.validateCalls[0].components[0]
    assert.equal(component.kind, "observability")
    assert.equal(component.config.atof.enabled, true)
    assert.equal(component.config.atof.output_directory, path.join(dir, ".nemoflow"))
    assert.equal(component.config.atof.filename, "opencode.atof.jsonl")
    assert.equal(component.config.atif.enabled, true)
    assert.equal(component.config.atif.output_directory, path.join(dir, ".nemoflow"))
    assert.equal(component.config.atif.filename_template, "opencode-{session_id}.atif.json")
  })

  it("keeps normal bus events internal while recording session errors", async () => {
    const dir = await makeTempDir()
    const { runtime, server } = createHarness()
    const hooks = await server(
      { directory: dir },
      {
        enabled: true,
        logPath: "./.nemoflow/opencode-plugin.log",
        plugins: pluginConfig(),
      },
    )

    const model = { providerID: "anthropic", modelID: "claude-test" }
    await hooks.event?.({
      event: {
        id: "evt_created",
        type: "session.created",
        properties: {
          sessionID: "ses_2",
          info: { id: "ses_2", agent: "review", model },
          apiKey: "secret",
          outputTokens: 8,
        },
      },
    })
    await hooks.event?.({
      event: {
        id: "evt_updated",
        type: "session.updated",
        properties: { sessionID: "ses_2", info: { id: "ses_2", agent: "review", model } },
      },
    })
    await hooks["chat.message"]?.(
      {
        sessionID: "ses_2",
        agent: "review",
        model,
        messageID: "msg_2",
        apiKey: "secret",
        outputTokens: 3,
      },
      {
        message: { id: "msg_2", role: "user", agent: "review" },
        parts: [{ id: "part_2", type: "text", text: "summarize this" }],
      },
    )
    await hooks["chat.params"]?.(
      {
        sessionID: "ses_2",
        agent: "review",
        model,
        provider: { source: "config", options: { apiKey: "secret" } },
        message: { id: "msg_2" },
      },
      { maxOutputTokens: 64, temperature: 0 },
    )
    await hooks.event?.({
      event: {
        id: "evt_error",
        type: "session.error",
        properties: { sessionID: "ses_2", error: { message: "provider failed", apiKey: "secret" } },
      },
    })
    await hooks.event?.({
      event: {
        id: "evt_deleted",
        type: "session.deleted",
        properties: { sessionID: "ses_2" },
      },
    })

    const names = eventNames(runtime.events)
    const error = runtime.events.find((event) => event.name === "opencode.session.error")
    const serialized = JSON.stringify(runtime.events)

    assert.equal(names.includes("opencode.session.created"), false)
    assert.equal(names.includes("opencode.session.updated"), false)
    assert.ok(names.includes("opencode.session.error"))
    assert.equal(names.includes("opencode.session.deleted"), false)
    assert.equal(error.metadata.sessionID, "ses_2")
    assert.equal(error.metadata.agent, "review")
    assert.equal(error.metadata.model, "anthropic/claude-test")
    assert.match(serialized, /"apiKey":"\[Redacted\]"/)
    assert.equal(names.includes("opencode.session.flush"), false)
  })

  it("ignores hooks without an OpenCode session identifier", async () => {
    const dir = await makeTempDir()
    const { runtime, server } = createHarness()
    const hooks = await server({ directory: dir }, { enabled: true, plugins: pluginConfig() })

    await assert.doesNotReject(async () => {
      await hooks["chat.message"]?.({ agent: "build" }, { message: { id: "msg_missing" } })
      await hooks["chat.params"]?.({ agent: "build", message: { id: "msg_missing" } }, {})
      await hooks["tool.execute.before"]?.({ tool: "read", callID: "call_missing" }, { args: { path: "x" } })
      await hooks["tool.execute.after"]?.({ tool: "read", callID: "call_missing" }, { output: "x" })
    })
    assert.equal(runtime.events.length, 0)
  })

  it("stays pass-through when disabled", async () => {
    const dir = await makeTempDir()
    const { pluginHost, server } = createHarness()
    const hooks = await server({ directory: dir }, { enabled: false, plugins: pluginConfig() })

    assert.deepEqual(hooks, {})
    assert.equal(pluginHost.validateCalls.length, 0)
    assert.equal(pluginHost.initializeCalls.length, 0)
  })

  it("logs once and disables hooks when the runtime cannot load", async () => {
    const dir = await makeTempDir()
    const { server } = createHarness({ loadError: new Error("missing native binding") })
    const hooks = await server({ directory: dir }, { enabled: true, plugins: pluginConfig() })

    assert.deepEqual(hooks, {})
    const log = await fs.readFile(path.join(dir, ".nemoflow", "opencode-plugin.log"), "utf8")
    assert.match(log, /pass-through/)
  })

  it("logs and disables hooks when removed exporter options are used", async () => {
    const dir = await makeTempDir()
    const { pluginHost, server } = createHarness()
    const hooks = await server(
      { directory: dir },
      {
        enabled: true,
        atofPath: "./.nemoflow/opencode.atof.jsonl",
        logPath: "./.nemoflow/opencode-plugin.log",
      },
    )

    assert.deepEqual(hooks, {})
    assert.equal(pluginHost.validateCalls.length, 0)
    const log = await fs.readFile(path.join(dir, ".nemoflow", "opencode-plugin.log"), "utf8")
    assert.match(log, /config invalid/)
    assert.match(log, /atofPath was removed/)
  })

  it("logs and disables hooks when generic plugin validation fails", async () => {
    const dir = await makeTempDir()
    const { pluginHost, server } = createHarness({
      validateDiagnostics: [
        {
          level: "error",
          code: "plugin.unknown_component",
          component: "missing",
          message: "unknown component",
        },
      ],
    })
    const hooks = await server({ directory: dir }, { enabled: true, plugins: pluginConfig() })

    assert.deepEqual(hooks, {})
    assert.equal(pluginHost.validateCalls.length, 1)
    assert.equal(pluginHost.initializeCalls.length, 0)
    const log = await fs.readFile(path.join(dir, ".nemoflow", "opencode-plugin.log"), "utf8")
    assert.match(log, /plugin.unknown_component/)
    assert.match(log, /plugin host config validation failed/)
  })

  it("redacts console diagnostics when file logging is disabled", async () => {
    const dir = await makeTempDir()
    const warnings = []
    const originalWarn = console.warn
    console.warn = (...args) => {
      warnings.push(args)
    }

    try {
      const { server } = createHarness({
        validateDiagnostics: [
          {
            level: "error",
            code: "plugin.invalid",
            message: "invalid config",
            apiKey: "secret-value",
          },
        ],
      })
      const hooks = await server({ directory: dir }, { enabled: true, logPath: "", plugins: pluginConfig() })

      assert.deepEqual(hooks, {})
    } finally {
      console.warn = originalWarn
    }

    assert.ok(warnings.length > 0)
    assert.equal(warnings[0][1].apiKey, "[Redacted]")
    assert.equal(JSON.stringify(warnings).includes("secret-value"), false)
  })

  it("keeps hooks non-fatal when NeMo Flow runtime calls throw", async () => {
    async function assertRuntimeFailure(mutator, runHook, pattern) {
      const dir = await makeTempDir()
      const runtime = createFakeRuntime()
      mutator(runtime)
      const { server } = createHarness({ runtime })
      const hooks = await server({ directory: dir }, { enabled: true, plugins: pluginConfig() })

      await assert.doesNotReject(async () => {
        await runHook(hooks)
      })
      await waitForLogMatch(path.join(dir, ".nemoflow", "opencode-plugin.log"), pattern)
    }

    await assertRuntimeFailure(
      (runtime) => {
        runtime.pushScope = () => {
          throw new Error("scope failed")
        }
      },
      (hooks) =>
        hooks["chat.message"]?.(
          { sessionID: "ses_scope", agent: "build" },
          { message: { id: "msg_scope", role: "user" }, parts: [] },
        ),
      /failed to start OpenCode session scope/,
    )

    await assertRuntimeFailure(
      (runtime) => {
        runtime.event = () => {
          throw new Error("event failed")
        }
      },
      (hooks) =>
        hooks.event?.({
          event: {
            id: "evt_error",
            type: "session.error",
            properties: { sessionID: "ses_event", error: { message: "provider failed" } },
          },
        }),
      /failed to record OpenCode mark event/,
    )

    await assertRuntimeFailure(
      (runtime) => {
        runtime.llmCall = () => {
          throw new Error("llm failed")
        }
      },
      (hooks) =>
        hooks["chat.params"]?.(
          { sessionID: "ses_llm", agent: "build", model: { providerID: "test-provider", id: "test-model" } },
          {},
        ),
      /failed to start OpenCode LLM span/,
    )

    await assertRuntimeFailure(
      (runtime) => {
        runtime.llmCallEnd = () => {
          throw new Error("llm end failed")
        }
      },
      async (hooks) => {
        await hooks["chat.params"]?.(
          { sessionID: "ses_llm_end", agent: "build", model: { providerID: "test-provider", id: "test-model" } },
          {},
        )
        await hooks["tool.execute.before"]?.(
          { tool: "write", sessionID: "ses_llm_end", callID: "call_llm_end" },
          { args: { path: "x" } },
        )
      },
      /failed to close OpenCode LLM span/,
    )

    await assertRuntimeFailure(
      (runtime) => {
        runtime.toolCall = () => {
          throw new Error("tool failed")
        }
      },
      (hooks) =>
        hooks["tool.execute.before"]?.(
          { tool: "write", sessionID: "ses_tool", callID: "call_tool" },
          { args: { path: "x" } },
        ),
      /failed to start OpenCode tool span/,
    )

    await assertRuntimeFailure(
      (runtime) => {
        runtime.toolCallEnd = () => {
          throw new Error("tool end failed")
        }
      },
      async (hooks) => {
        await hooks["tool.execute.before"]?.(
          { tool: "write", sessionID: "ses_tool_end", callID: "call_tool_end" },
          { args: { path: "x" } },
        )
        await hooks["tool.execute.after"]?.(
          { tool: "write", sessionID: "ses_tool_end", callID: "call_tool_end" },
          { output: "done" },
        )
      },
      /failed to close OpenCode tool span/,
    )
  })

  it("flushes open sessions and clears the plugin host during cleanup", async () => {
    const dir = await makeTempDir()
    const { runtime, pluginHost, server, cleanup } = createHarness()
    const hooks = await server({ directory: dir }, { enabled: true, plugins: pluginConfig() })

    await hooks["chat.message"]?.(
      {
        sessionID: "ses_3",
        agent: "build",
        model: { providerID: "test-provider", modelID: "test-model" },
      },
      {
        message: { id: "msg_3", role: "user", agent: "build" },
        parts: [],
      },
    )
    await hooks["tool.execute.before"]?.(
      { tool: "write", sessionID: "ses_3", callID: "call_3" },
      { args: { path: "left-open.txt" } },
    )

    await cleanup()

    const names = eventNames(runtime.events)
    const pendingToolEnd = runtime.events.find(
      (event) => event.category === "tool" && event.scope_category === "end" && event.metadata.callID === "call_3",
    )
    assert.equal(pluginHost.clearCalls, 1)
    assert.equal(names.includes("opencode.session.flush"), false)
    assert.equal(pendingToolEnd.data.status, "unknown")
    assert.equal(runtime.events.filter((event) => event.category === "agent" && event.scope_category === "end").length, 1)
    assertSessionStackUsed(runtime)
  })
})
