// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict"
import fs from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { describe, it } from "node:test"

import { createServerPlugin } from "../server.js"

function createFakeRuntime() {
  const subscribers = new Map()
  let counter = 0

  function emit(event) {
    for (const callback of subscribers.values()) callback(event)
  }

  class AtifExporter {
    constructor(sessionID, agentName, agentVersion, modelName) {
      this.sessionID = sessionID
      this.agentName = agentName
      this.agentVersion = agentVersion
      this.modelName = modelName
      this.events = []
      this.callback = (event) => this.events.push(event)
    }

    register(name) {
      subscribers.set(name, this.callback)
    }

    deregister(name) {
      return subscribers.delete(name)
    }

    exportJson() {
      return JSON.stringify({
        session_id: this.sessionID,
        agent: {
          name: this.agentName,
          version: this.agentVersion,
          model_name: this.modelName,
        },
        steps: this.events,
      })
    }
  }

  return {
    ScopeType: { Agent: 0 },
    AtifExporter,
    registerSubscriber(name, callback) {
      subscribers.set(name, callback)
    },
    deregisterSubscriber(name) {
      return subscribers.delete(name)
    },
    createScopeStack() {
      return { id: `stack-${++counter}` }
    },
    currentScopeStack() {
      return { id: "current" }
    },
    setThreadScopeStack(_stack) {},
    pushScope(name, scopeType, _parent, attributes, data, metadata, input) {
      const handle = {
        uuid: `scope-${++counter}`,
        name,
        scopeType,
        attributes,
      }
      emit({
        kind: "scope",
        category: "agent",
        scope_category: "start",
        uuid: handle.uuid,
        name,
        data,
        metadata,
        input,
      })
      return handle
    },
    popScope(handle, output) {
      emit({
        kind: "scope",
        category: "agent",
        scope_category: "end",
        uuid: handle.uuid,
        name: handle.name,
        data: output,
      })
    },
    event(name, handle, data, metadata) {
      emit({
        kind: "mark",
        uuid: `mark-${++counter}`,
        parent_uuid: handle?.uuid,
        name,
        data,
        metadata,
      })
    },
    toolCall(name, args, handle, attributes, data, metadata, toolCallID) {
      const tool = {
        uuid: `tool-${++counter}`,
        name,
        parentUuid: handle?.uuid,
        toolCallID,
      }
      emit({
        kind: "scope",
        category: "tool",
        scope_category: "start",
        uuid: tool.uuid,
        name,
        parent_uuid: handle?.uuid,
        data: args,
        metadata,
      })
      return tool
    },
    toolCallEnd(handle, result, data, metadata) {
      emit({
        kind: "scope",
        category: "tool",
        scope_category: "end",
        uuid: handle.uuid,
        name: handle.name,
        data: result ?? data,
        metadata,
      })
    },
  }
}

async function makeTempDir() {
  return fs.mkdtemp(path.join(os.tmpdir(), "nemo-flow-opencode-plugin-"))
}

async function readJsonl(filePath) {
  const content = await fs.readFile(filePath, "utf8")
  return content
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line))
}

describe("NeMo Flow OpenCode plugin", () => {
  it("records OpenCode hooks to ATOF and flushes ATIF on idle", async () => {
    const dir = await makeTempDir()
    const server = createServerPlugin({ loadRuntime: async () => createFakeRuntime() })
    const hooks = await server(
      { directory: dir },
      {
        enabled: true,
        atofPath: "./.nemoflow/opencode.atof.jsonl",
        atifPath: "./.nemoflow/opencode.atif.json",
        logPath: "./.nemoflow/opencode-plugin.log",
      },
    )

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

    const atofPath = path.join(dir, ".nemoflow", "opencode.atof.jsonl")
    const atifPath = path.join(dir, ".nemoflow", "opencode.atif.json")
    const atof = await fs.readFile(atofPath, "utf8")
    const atif = JSON.parse(await fs.readFile(atifPath, "utf8"))

    assert.match(atof, /opencode\.chat\.message/)
    assert.match(atof, /opencode\.llm\.request/)
    assert.match(atof, /"category":"tool"/)
    assert.equal(atif.session_id, "ses_1")
    assert.ok(atif.steps.some((event) => event.name === "opencode.session.flush"))
  })

  it("records session lifecycle events, message metadata, errors, and deleted flushes", async () => {
    const dir = await makeTempDir()
    const server = createServerPlugin({ loadRuntime: async () => createFakeRuntime() })
    const hooks = await server(
      { directory: dir },
      {
        enabled: true,
        atofPath: "./.nemoflow/opencode.atof.jsonl",
        atifPath: "./.nemoflow/opencode.atif.json",
        logPath: "./.nemoflow/opencode-plugin.log",
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

    const atofPath = path.join(dir, ".nemoflow", "opencode.atof.jsonl")
    const atifPath = path.join(dir, ".nemoflow", "opencode.atif.json")
    const events = await readJsonl(atofPath)
    const names = events.map((event) => event.name).filter(Boolean)
    const message = events.find((event) => event.name === "opencode.chat.message")
    const serialized = JSON.stringify(events)
    const atif = JSON.parse(await fs.readFile(atifPath, "utf8"))

    assert.ok(names.includes("opencode.session.created"))
    assert.ok(names.includes("opencode.session.updated"))
    assert.ok(names.includes("opencode.session.error"))
    assert.ok(names.includes("opencode.session.deleted"))
    assert.equal(message.metadata.sessionID, "ses_2")
    assert.equal(message.metadata.agent, "review")
    assert.equal(message.metadata.model, "anthropic/claude-test")
    assert.match(serialized, /"apiKey":"\[Redacted\]"/)
    assert.match(serialized, /"outputTokens":3/)
    assert.equal(atif.session_id, "ses_2")
    assert.ok(atif.steps.some((event) => event.name === "opencode.session.flush"))
  })

  it("ignores hooks without an OpenCode session identifier", async () => {
    const dir = await makeTempDir()
    const server = createServerPlugin({ loadRuntime: async () => createFakeRuntime() })
    const hooks = await server({ directory: dir }, { enabled: true })

    await assert.doesNotReject(async () => {
      await hooks["chat.message"]?.({ agent: "build" }, { message: { id: "msg_missing" } })
      await hooks["chat.params"]?.({ agent: "build", message: { id: "msg_missing" } }, {})
      await hooks["tool.execute.before"]?.({ tool: "read", callID: "call_missing" }, { args: { path: "x" } })
      await hooks["tool.execute.after"]?.({ tool: "read", callID: "call_missing" }, { output: "x" })
    })
    await assert.rejects(fs.stat(path.join(dir, ".nemoflow", "opencode.atof.jsonl")))
  })

  it("stays pass-through when disabled", async () => {
    const dir = await makeTempDir()
    const server = createServerPlugin({ loadRuntime: async () => createFakeRuntime() })
    const hooks = await server({ directory: dir }, { enabled: false })

    assert.deepEqual(hooks, {})
    await assert.rejects(fs.stat(path.join(dir, ".nemoflow", "opencode.atof.jsonl")))
  })

  it("logs once and disables hooks when the runtime cannot load", async () => {
    const dir = await makeTempDir()
    const server = createServerPlugin({
      loadRuntime: async () => {
        throw new Error("missing native binding")
      },
    })
    const hooks = await server({ directory: dir }, { enabled: true })

    assert.deepEqual(hooks, {})
    const log = await fs.readFile(path.join(dir, ".nemoflow", "opencode-plugin.log"), "utf8")
    assert.match(log, /pass-through/)
  })
})
