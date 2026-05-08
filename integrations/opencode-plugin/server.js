// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fsSync from "node:fs"
import fs from "node:fs/promises"
import path from "node:path"

const PLUGIN_ID = "@nvidia/nemoflow-opencode-plugin"
const AGENT_VERSION = "opencode-plugin-0.2.0"
const RELEVANT_EVENTS = new Set([
  "session.created",
  "session.updated",
  "session.deleted",
  "session.error",
  "session.status",
  "session.idle",
  "message.updated",
  "message.removed",
  "message.part.updated",
  "message.part.delta",
  "message.part.removed",
])

/**
 * Create the plugin logger.
 */
function createLogger(logPath) {
  const seen = new Set()

  /**
   * Write one diagnostic record to the configured log destination.
   */
  async function write(level, message, extra) {
    const record = {
      timestamp: new Date().toISOString(),
      level,
      plugin: PLUGIN_ID,
      message,
      ...(extra === undefined ? {} : { extra: toJsonSafe(extra) }),
    }
    const line = JSON.stringify(record) + "\n"
    if (logPath) {
      await ensureParentDir(logPath)
      await fs.appendFile(logPath, line)
      return
    }
    const text = `[${PLUGIN_ID}] ${message}`
    if (level === "error") console.error(text, extra ?? "")
    else if (level === "warn") console.warn(text, extra ?? "")
    else console.info(text, extra ?? "")
  }

  return {
    info: (message, extra) => write("info", message, extra),
    warn: (message, extra) => write("warn", message, extra),
    error: (message, extra) => write("error", message, extra),
    warnOnce: (key, message, extra) => {
      if (seen.has(key)) return Promise.resolve()
      seen.add(key)
      return write("warn", message, extra)
    },
  }
}

/**
 * Ensure the parent directory for an output file exists.
 */
async function ensureParentDir(filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true })
}

/**
 * Resolve a plugin output path relative to the OpenCode project directory.
 */
function resolveOutputPath(baseDir, value) {
  if (typeof value !== "string" || value.trim() === "") return undefined
  if (path.isAbsolute(value)) return value
  return path.resolve(baseDir, value)
}

/**
 * Normalize OpenCode plugin options into concrete runtime settings.
 */
function normalizeOptions(input, options = {}) {
  const baseDir = input?.directory ?? process.cwd()
  return {
    enabled: options.enabled !== false,
    atofPath: resolveOutputPath(baseDir, options.atofPath ?? "./.nemoflow/opencode.atof.jsonl"),
    atifPath: resolveOutputPath(baseDir, options.atifPath ?? "./.nemoflow/opencode.atif.json"),
    logPath: resolveOutputPath(baseDir, options.logPath ?? "./.nemoflow/opencode-plugin.log"),
  }
}

/**
 * Convert arbitrary OpenCode hook payloads into JSON-safe data.
 */
function toJsonSafe(value) {
  if (value === undefined) return null
  if (value instanceof Error) {
    return {
      name: value.name,
      message: value.message,
      stack: value.stack,
    }
  }

  const seen = new WeakSet()
  try {
    return JSON.parse(
      JSON.stringify(value, (key, nested) => {
        if (/^(api[-_]?key|authorization|password|secret|access[-_]?token|refresh[-_]?token|id[-_]?token|token)$/i.test(key)) {
          return "[Redacted]"
        }
        if (typeof nested === "bigint") return nested.toString()
        if (typeof nested === "function") return `[Function ${nested.name || "anonymous"}]`
        if (nested instanceof Error) return toJsonSafe(nested)
        if (nested && typeof nested === "object") {
          if (seen.has(nested)) return "[Circular]"
          seen.add(nested)
        }
        return nested
      }),
    )
  } catch {
    return null
  }
}

/**
 * Format OpenCode model metadata as a stable provider/model string.
 */
function modelName(model) {
  if (!model) return undefined
  const provider = model.providerID ?? model.provider?.id
  const id = model.modelID ?? model.id
  if (provider && id) return `${provider}/${id}`
  if (id) return String(id)
  return undefined
}

/**
 * Read the OpenCode agent name from hook input or event metadata.
 */
function agentName(input, fallback = "opencode") {
  if (typeof input?.agent === "string" && input.agent) return input.agent
  if (typeof input?.message?.agent === "string" && input.message.agent) return input.message.agent
  if (typeof input?.info?.agent === "string" && input.info.agent) return input.info.agent
  return fallback
}

/**
 * Read the OpenCode session ID from a bus event payload.
 */
function eventSessionID(event) {
  const props = event?.properties
  return props?.sessionID ?? props?.info?.id
}

/**
 * Build metadata attached to the NeMo Flow session scope.
 */
function inputSessionMetadata(sessionID, state) {
  return {
    source: "opencode",
    sessionID,
    agent: state.agent,
    model: state.model,
  }
}

/**
 * Build common metadata for OpenCode-derived NeMo Flow marks.
 */
function eventMetadata(session, extra = {}) {
  return {
    agent: session?.agent,
    model: session?.model,
    ...extra,
  }
}

/**
 * Decide whether an OpenCode event should flush the ATIF trajectory.
 */
function shouldFlushEvent(event) {
  if (!event) return false
  if (event.type === "session.deleted" || event.type === "session.idle") return true
  if (event.type !== "session.status") return false
  return event.properties?.status?.type === "idle"
}

/**
 * Create the NeMo Flow adapter behind the OpenCode plugin hooks.
 */
function createNemoFlowAdapter(lib, options, logger) {
  const sessions = new Map()
  const recentFlushes = new Map()
  const trajectories = []
  let atofSubscriberName
  let atofDeregisterTimer
  let closed = false

  /**
   * Register the process-local ATOF JSONL subscriber on first use.
   */
  function registerAtOfJsonlExporter() {
    if (atofDeregisterTimer) {
      clearTimeout(atofDeregisterTimer)
      atofDeregisterTimer = undefined
    }
    if (atofSubscriberName || !options.atofPath) return
    fsSync.mkdirSync(path.dirname(options.atofPath), { recursive: true })
    atofSubscriberName = `${PLUGIN_ID}:atof:${process.pid}:${Date.now()}`
    lib.registerSubscriber(atofSubscriberName, (event) => {
      fsSync.appendFileSync(options.atofPath, JSON.stringify(event) + "\n")
    })
    void logger.info("registered ATOF JSONL exporter", { path: options.atofPath })
  }

  /**
   * Deregister the ATOF JSONL subscriber after the last session closes.
   */
  function deregisterAtOfJsonlExporter() {
    if (atofDeregisterTimer) {
      clearTimeout(atofDeregisterTimer)
      atofDeregisterTimer = undefined
    }
    if (!atofSubscriberName) return
    try {
      lib.deregisterSubscriber(atofSubscriberName)
    } catch (error) {
      void logger.warnOnce("atof-deregister", "failed to deregister ATOF JSONL exporter", error)
    } finally {
      atofSubscriberName = undefined
    }
  }

  /**
   * Delay ATOF subscriber cleanup so adjacent events can still flush.
   */
  function scheduleAtOfJsonlExporterDeregister() {
    if (!atofSubscriberName || atofDeregisterTimer) return
    atofDeregisterTimer = setTimeout(() => {
      deregisterAtOfJsonlExporter()
    }, 250)
  }

  /**
   * Run a callback with the session scope stack active when supported.
   */
  function withStack(session, callback) {
    if (!session.stack || typeof lib.setThreadScopeStack !== "function") return callback()
    const previous = typeof lib.currentScopeStack === "function" ? lib.currentScopeStack() : undefined
    lib.setThreadScopeStack(session.stack)
    try {
      return callback()
    } finally {
      if (previous) lib.setThreadScopeStack(previous)
    }
  }

  /**
   * Create or update the NeMo Flow session state for an OpenCode session.
   */
  function ensureSession(sessionID, metadata = {}) {
    if (!sessionID) return undefined
    registerAtOfJsonlExporter()

    let session = sessions.get(sessionID)
    if (session) {
      if (metadata.agent) session.agent = metadata.agent
      if (metadata.model) session.model = metadata.model
      return session
    }

    session = {
      id: sessionID,
      agent: metadata.agent ?? "opencode",
      model: metadata.model,
      stack: typeof lib.createScopeStack === "function" ? lib.createScopeStack() : undefined,
      scope: undefined,
      exporter: undefined,
      exporterName: `${PLUGIN_ID}:atif:${sessionID}:${Date.now()}`,
      pendingTools: new Map(),
    }

    session.exporter = new lib.AtifExporter(session.id, session.agent, AGENT_VERSION, session.model ?? null)
    session.exporter.register(session.exporterName)
    session.scope = withStack(session, () =>
      lib.pushScope(
        "opencode.session",
        lib.ScopeType?.Agent ?? 0,
        null,
        null,
        { sessionID },
        inputSessionMetadata(sessionID, session),
        { sessionID, source: "opencode" },
      ),
    )
    sessions.set(sessionID, session)
    emitMark(session, "opencode.session.observed", {
      sessionID,
      agent: session.agent,
      model: session.model,
    })
    return session
  }

  /**
   * Emit an OpenCode milestone as a NeMo Flow mark event.
   */
  function emitMark(session, name, data, metadata = {}) {
    if (!session?.scope) return
    lib.event(
      name,
      session.scope,
      toJsonSafe(data),
      {
        source: "opencode",
        sessionID: session.id,
        ...toJsonSafe(metadata),
      },
      null,
    )
  }

  /**
   * Write all collected ATIF trajectories to the configured file.
   */
  function writeAtifFile() {
    if (!options.atifPath) return
    const payload = trajectories.length === 1 ? trajectories[0] : { trajectories }
    fsSync.mkdirSync(path.dirname(options.atifPath), { recursive: true })
    fsSync.writeFileSync(options.atifPath, JSON.stringify(payload, null, 2))
  }

  /**
   * Close an OpenCode session scope and export its trajectory.
   */
  function flushSession(sessionID, reason) {
    const session = sessions.get(sessionID)
    if (!session) return
    recentFlushes.set(sessionID, Date.now())
    emitMark(session, "opencode.session.flush", { sessionID, reason })
    for (const [key, tool] of session.pendingTools) {
      try {
        lib.toolCallEnd(
          tool.handle,
          { status: "unknown", reason: "session flushed before tool.execute.after" },
          null,
          { source: "opencode", sessionID, callID: tool.callID },
        )
      } catch (error) {
        void logger.warnOnce(`tool-close:${key}`, "failed to close pending OpenCode tool span", error)
      }
    }
    session.pendingTools.clear()

    if (session.scope) {
      try {
        withStack(session, () => lib.popScope(session.scope, { sessionID, reason }, null))
      } catch (error) {
        void logger.warnOnce(`scope-pop:${sessionID}`, "failed to close OpenCode session scope", error)
      }
    }

    try {
      trajectories.push(JSON.parse(session.exporter.exportJson()))
      writeAtifFile()
    } catch (error) {
      void logger.warnOnce(`atif-export:${sessionID}`, "failed to export ATIF trajectory", error)
    }

    try {
      session.exporter.deregister(session.exporterName)
    } catch (error) {
      void logger.warnOnce(`atif-deregister:${sessionID}`, "failed to deregister ATIF exporter", error)
    }
    sessions.delete(sessionID)
    if (sessions.size === 0) scheduleAtOfJsonlExporterDeregister()
  }

  return {
    /**
     * Record OpenCode configuration context for diagnostics.
     */
    async recordConfig(config) {
      if (closed) return
      await logger.info("observed OpenCode config", {
        model: config?.model,
        agents: config?.agent ? Object.keys(config.agent) : undefined,
      })
    },

    /**
     * Record relevant OpenCode bus events as NeMo Flow marks.
     */
    async recordEvent(event) {
      if (closed || !RELEVANT_EVENTS.has(event?.type)) return
      const sessionID = eventSessionID(event)
      if (!sessionID) return
      const recentFlushAt = recentFlushes.get(sessionID)
      if (shouldFlushEvent(event) && recentFlushAt && Date.now() - recentFlushAt < 2000) return
      const props = event.properties ?? {}
      const session = ensureSession(sessionID, {
        agent: agentName(props.info, undefined),
        model: modelName(props.info?.model),
      })
      emitMark(
        session,
        `opencode.${event.type}`,
        {
          id: event.id,
          type: event.type,
          properties: props,
        },
        eventMetadata(session, { eventType: event.type }),
      )
      if (shouldFlushEvent(event)) {
        flushSession(sessionID, event.type)
      }
    },

    /**
     * Record user message metadata for the current OpenCode turn.
     */
    async recordChatMessage(input, output) {
      if (closed) return
      const session = ensureSession(input.sessionID, {
        agent: agentName(input),
        model: modelName(input.model ?? output?.message?.model),
      })
      if (!session) return
      emitMark(
        session,
        "opencode.chat.message",
        {
          input,
          message: output?.message,
          parts: output?.parts,
        },
        eventMetadata(session, { messageID: input.messageID ?? output?.message?.id }),
      )
    },

    /**
     * Record model and provider metadata near the LLM request boundary.
     */
    async recordChatParams(input, output) {
      if (closed) return
      const session = ensureSession(input.sessionID, {
        agent: agentName(input),
        model: modelName(input.model),
      })
      if (!session) return
      emitMark(
        session,
        "opencode.llm.request",
        {
          sessionID: input.sessionID,
          agent: input.agent,
          provider: input.provider,
          model: input.model,
          message: input.message,
          params: output,
          limitation: "OpenCode Phase 1 hooks expose request metadata but not exact stream completion.",
        },
        eventMetadata(session, { messageID: input.message?.id }),
      )
    },

    /**
     * Start a NeMo Flow tool span for an OpenCode tool call.
     */
    async recordToolBefore(input, output) {
      if (closed) return
      const session = ensureSession(input.sessionID)
      if (!session) return
      const args = toJsonSafe(output?.args)
      const handle = lib.toolCall(
        input.tool,
        args,
        session.scope,
        null,
        { sessionID: input.sessionID, callID: input.callID },
        { source: "opencode", sessionID: input.sessionID, callID: input.callID },
        input.callID,
        null,
      )
      session.pendingTools.set(input.callID, { handle, callID: input.callID, tool: input.tool, args })
    },

    /**
     * Finish a successful NeMo Flow tool span for an OpenCode tool call.
     */
    async recordToolAfter(input, output) {
      if (closed) return
      const session = ensureSession(input.sessionID)
      if (!session) return
      let pending = session.pendingTools.get(input.callID)
      if (!pending) {
        const args = toJsonSafe(input.args)
        const handle = lib.toolCall(
          input.tool,
          args,
          session.scope,
          null,
          { sessionID: input.sessionID, callID: input.callID },
          { source: "opencode", sessionID: input.sessionID, callID: input.callID, recovered: true },
          input.callID,
          null,
        )
        pending = { handle, callID: input.callID, tool: input.tool, args }
      }
      lib.toolCallEnd(
        pending.handle,
        toJsonSafe({
          title: output?.title,
          output: output?.output,
          metadata: output?.metadata,
        }),
        null,
        { source: "opencode", sessionID: input.sessionID, callID: input.callID },
        null,
      )
      session.pendingTools.delete(input.callID)
    },

    /**
     * Flush open sessions and unregister exporters during plugin shutdown.
     */
    async close() {
      closed = true
      for (const sessionID of [...sessions.keys()]) {
        flushSession(sessionID, "plugin-close")
      }
      deregisterAtOfJsonlExporter()
    },
  }
}

/**
 * Load the default NeMo Flow Node.js runtime.
 */
async function loadDefaultRuntime() {
  if (process.env.NEMO_FLOW_OPENCODE_FORCE_INIT_FAILURE === "1") {
    throw new Error("forced initialization failure")
  }
  const mod = await import("nemo-flow-node")
  return mod.default ?? mod
}

/**
 * Create the OpenCode server plugin entrypoint.
 */
export function createServerPlugin({ loadRuntime = loadDefaultRuntime } = {}) {
  return async function server(input, options) {
    const normalized = normalizeOptions(input, options)
    const logger = createLogger(normalized.logPath)

    if (!normalized.enabled) {
      await logger.warnOnce("disabled", "NeMo Flow OpenCode plugin disabled by configuration")
      return {}
    }

    let adapter
    try {
      const lib = await loadRuntime()
      adapter = createNemoFlowAdapter(lib, normalized, logger)
      await logger.info("initialized NeMo Flow OpenCode plugin", {
        atofPath: normalized.atofPath,
        atifPath: normalized.atifPath,
      })
    } catch (error) {
      await logger.warnOnce(
        "init-failed",
        "NeMo Flow runtime unavailable; OpenCode plugin is running pass-through",
        error,
      )
      return {}
    }

    return {
      config: async (config) => adapter.recordConfig(config),
      event: async ({ event }) => adapter.recordEvent(event),
      "chat.message": async (hookInput, output) => adapter.recordChatMessage(hookInput, output),
      "chat.params": async (hookInput, output) => adapter.recordChatParams(hookInput, output),
      "tool.execute.before": async (hookInput, output) => adapter.recordToolBefore(hookInput, output),
      "tool.execute.after": async (hookInput, output) => adapter.recordToolAfter(hookInput, output),
    }
  }
}

export const server = createServerPlugin()

export default {
  id: PLUGIN_ID,
  server,
}
