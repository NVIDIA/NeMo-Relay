// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs/promises"
import path from "node:path"

const PLUGIN_ID = "nemo-flow-opencode"
const DEFAULT_PLUGIN_HOST_CONFIG = Object.freeze({
  version: 1,
  components: Object.freeze([]),
})
const RECENT_FLUSH_TTL_MS = 2000
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
 * Resolve an output directory inside generic observability plugin config.
 */
function resolveOutputDirectory(baseDir, value) {
  if (typeof value !== "string" || value.trim() === "") return value
  if (path.isAbsolute(value)) return value
  return path.resolve(baseDir, value)
}

/**
 * Normalize OpenCode plugin options into concrete runtime settings.
 */
function normalizeOptions(input, options = {}) {
  const baseDir = input?.directory ?? process.cwd()
  const rawOptions = options ?? {}
  rejectRemovedOption(rawOptions, "atofPath", "configure plugins.components[].config.atof instead")
  rejectRemovedOption(rawOptions, "atifPath", "configure plugins.components[].config.atif instead")

  return {
    enabled: rawOptions.enabled !== false,
    plugins: normalizePluginHostConfig(baseDir, rawOptions.plugins),
    logPath: resolveOutputPath(baseDir, rawOptions.logPath ?? "./.nemoflow/opencode-plugin.log"),
  }
}

/**
 * Normalize the embedded generic NeMo Flow plugin-host configuration.
 */
function normalizePluginHostConfig(baseDir, value) {
  if (value === undefined) {
    return clonePluginHostConfig(DEFAULT_PLUGIN_HOST_CONFIG)
  }

  const raw = asRecord(value, "plugins", false)
  const version = optionalNumber(raw.version, "plugins.version") ?? 1
  const components = raw.components === undefined ? [] : raw.components

  if (!Array.isArray(components)) {
    throw new Error("plugins.components must be an array")
  }

  return {
    ...raw,
    version,
    components: components.map((component, index) =>
      normalizePluginComponent(baseDir, component, `plugins.components[${index}]`),
    ),
  }
}

/**
 * Clone the mutable generic plugin config before giving it to the runtime.
 */
function clonePluginHostConfig(config) {
  return {
    ...config,
    components: [...config.components],
  }
}

/**
 * Normalize path-bearing sections on the built-in observability component.
 */
function normalizePluginComponent(baseDir, value, fieldPath) {
  const component = asRecord(value, fieldPath, false)
  const normalized = { ...component }

  if (component.kind !== "observability" || component.config === undefined) {
    return normalized
  }

  normalized.config = normalizeObservabilityConfig(baseDir, asRecord(component.config, `${fieldPath}.config`, false))
  return normalized
}

/**
 * Keep OpenCode project-relative paths ergonomic while preserving NeMo Flow's
 * generic plugin config shape.
 */
function normalizeObservabilityConfig(baseDir, config) {
  const normalized = { ...config }
  for (const sectionName of ["atof", "atif"]) {
    if (normalized[sectionName] === undefined) continue
    const section = asRecord(normalized[sectionName], `observability.${sectionName}`, false)
    normalized[sectionName] = {
      ...section,
      output_directory: resolveOutputDirectory(baseDir, section.output_directory),
    }
  }
  return normalized
}

/**
 * Reject exporter options that were replaced by the generic plugin config.
 */
function rejectRemovedOption(options, name, hint) {
  if (Object.prototype.hasOwnProperty.call(options, name)) {
    throw new Error(`${name} was removed; ${hint}`)
  }
}

/**
 * Require an object config section, optionally treating undefined as empty.
 */
function asRecord(value, fieldPath, optional) {
  if (value === undefined && optional) return {}
  if (value !== null && typeof value === "object" && !Array.isArray(value)) return value
  throw new Error(`${fieldPath} must be an object`)
}

/**
 * Parse an optional finite number.
 */
function optionalNumber(value, fieldPath) {
  if (value === undefined) return undefined
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${fieldPath} must be a finite number`)
  }
  return value
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
 * Log plugin-host validation or activation diagnostics.
 */
async function logDiagnostics(logger, diagnostics = []) {
  for (const diagnostic of diagnostics) {
    const prefix = diagnostic.component ? `${diagnostic.component}: ` : ""
    const message = `${prefix}${diagnostic.code}: ${diagnostic.message}`
    if (diagnostic.level === "error") {
      await logger.warn(message, diagnostic)
    } else {
      await logger.info(message, diagnostic)
    }
  }
}

/**
 * Return true when a plugin-host report contains error diagnostics.
 */
function hasErrorDiagnostics(report) {
  return report?.diagnostics?.some((diagnostic) => diagnostic.level === "error") === true
}

/**
 * Validate and activate NeMo Flow's generic plugin-host config.
 */
async function initializePluginHost(pluginHost, config, logger) {
  const validationReport = pluginHost.validate(config)
  await logDiagnostics(logger, validationReport.diagnostics)
  if (hasErrorDiagnostics(validationReport)) {
    throw new Error("NeMo Flow plugin host config validation failed")
  }

  const activationReport = await pluginHost.initialize(config)
  await logDiagnostics(logger, activationReport.diagnostics)
  if (hasErrorDiagnostics(activationReport)) {
    await logger.warn("NeMo Flow plugin host initialized with error diagnostics")
  }
}

/**
 * Summarize the active generic plugin config for diagnostics.
 */
function pluginConfigSummary(config) {
  const components = Array.isArray(config?.components) ? config.components : []
  return {
    version: config?.version,
    components: components.map((component) => ({
      kind: component?.kind,
      enabled: component?.enabled !== false,
    })),
  }
}

/**
 * Convert thrown values into stable log records.
 */
function toMessage(error) {
  return error instanceof Error ? error.message : String(error)
}

/**
 * Create the NeMo Flow adapter behind the OpenCode plugin hooks.
 */
function createNemoFlowAdapter(lib, pluginHost, logger) {
  const sessions = new Map()
  const recentFlushes = new Map()
  let closed = false

  /**
   * Prune duplicate-flush suppression state so it cannot grow indefinitely.
   */
  function pruneRecentFlushes(now = Date.now()) {
    for (const [sessionID, timestamp] of recentFlushes) {
      if (now - timestamp > RECENT_FLUSH_TTL_MS) recentFlushes.delete(sessionID)
    }
  }

  /**
   * Return true when a just-closed session receives a duplicate idle/delete event.
   */
  function wasRecentlyFlushed(sessionID) {
    pruneRecentFlushes()
    const recentFlushAt = recentFlushes.get(sessionID)
    return recentFlushAt !== undefined && Date.now() - recentFlushAt <= RECENT_FLUSH_TTL_MS
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
      if (previous !== undefined) lib.setThreadScopeStack(previous)
    }
  }

  /**
   * Create or update the NeMo Flow session state for an OpenCode session.
   */
  function ensureSession(sessionID, metadata = {}) {
    if (!sessionID) return undefined

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
      pendingTools: new Map(),
    }

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
   * Close an OpenCode session scope so generic observability plugins can flush.
   */
  function flushSession(sessionID, reason) {
    const session = sessions.get(sessionID)
    if (!session) return
    recentFlushes.set(sessionID, Date.now())
    pruneRecentFlushes()
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

    sessions.delete(sessionID)
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
      if (shouldFlushEvent(event) && wasRecentlyFlushed(sessionID)) return
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
      try {
        pluginHost.clear()
      } catch (error) {
        await logger.warnOnce("plugin-host-clear", "failed to clear NeMo Flow plugin host", error)
      }
    },
  }
}

/**
 * Load the default NeMo Flow Node.js runtime and plugin host.
 */
async function loadDefaultModules() {
  if (process.env.NEMO_FLOW_OPENCODE_FORCE_INIT_FAILURE === "1") {
    throw new Error("forced initialization failure")
  }
  const [runtimeModule, pluginHostModule] = await Promise.all([
    import("nemo-flow-node"),
    import("nemo-flow-node/plugin"),
  ])
  return {
    lib: runtimeModule.default ?? runtimeModule,
    pluginHost: pluginHostModule.default ?? pluginHostModule,
  }
}

/**
 * Register process cleanup for OpenCode runs without an explicit close hook.
 */
function registerBeforeExitCleanup(close, logger) {
  const listener = () => {
    void close().catch((error) => {
      void logger.warnOnce("before-exit-cleanup", "failed to clean up NeMo Flow OpenCode plugin", error)
    })
  }
  process.on("beforeExit", listener)
  return () => process.removeListener("beforeExit", listener)
}

/**
 * Create the OpenCode server plugin entrypoint.
 */
export function createServerPlugin({
  loadModules = loadDefaultModules,
  registerCleanup = registerBeforeExitCleanup,
} = {}) {
  return async function server(input, options) {
    let normalized
    let logger

    try {
      normalized = normalizeOptions(input, options)
      logger = createLogger(normalized.logPath)
    } catch (error) {
      const baseDir = input?.directory ?? process.cwd()
      logger = createLogger(resolveOutputPath(baseDir, options?.logPath ?? "./.nemoflow/opencode-plugin.log"))
      await logger.warnOnce("config-invalid", "NeMo Flow OpenCode plugin config invalid; running pass-through", error)
      return {}
    }

    if (!normalized.enabled) {
      await logger.warnOnce("disabled", "NeMo Flow OpenCode plugin disabled by configuration")
      return {}
    }

    let adapter
    try {
      const { lib, pluginHost } = await loadModules()
      await initializePluginHost(pluginHost, normalized.plugins, logger)
      adapter = createNemoFlowAdapter(lib, pluginHost, logger)
      let unregisterCleanup
      unregisterCleanup = registerCleanup(async () => {
        unregisterCleanup?.()
        await adapter.close()
      }, logger)
      await logger.info("initialized NeMo Flow OpenCode plugin", {
        plugins: pluginConfigSummary(normalized.plugins),
      })
    } catch (error) {
      await logger.warnOnce(
        "init-failed",
        `NeMo Flow runtime unavailable or misconfigured; OpenCode plugin is running pass-through: ${toMessage(error)}`,
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
