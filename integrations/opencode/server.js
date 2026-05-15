// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs/promises"
import path from "node:path"

const PLUGIN_ID = "nemo-flow-opencode"
const RECENT_FLUSH_TTL_MS = 2000
const OBSERVED_EVENTS = new Set([
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
const INTERNAL_LLM_AGENTS = new Set(["title"])

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
    const loggedExtra = record.extra ?? ""
    if (level === "error") console.error(text, loggedExtra)
    else if (level === "warn") console.warn(text, loggedExtra)
    else console.info(text, loggedExtra)
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
    return defaultPluginHostConfig(baseDir)
  }

  const raw = asRecord(value, "plugins", false)
  const version = optionalNumber(raw.version, "plugins.version") ?? 1
  const components = raw.components === undefined ? [] : raw.components

  if (!Array.isArray(components)) {
    throw new Error("plugins.components must be an array")
  }

  const normalizedComponents = components.map((component, index) =>
    normalizePluginComponent(baseDir, component, `plugins.components[${index}]`),
  )

  return {
    ...raw,
    version,
    components: withDefaultObservabilityComponent(baseDir, normalizedComponents),
  }
}

/**
 * Build the default generic plugin config used by the OpenCode wrapper.
 */
function defaultPluginHostConfig(baseDir) {
  return {
    version: 1,
    components: [defaultObservabilityComponent(baseDir)],
  }
}

/**
 * Add a default observability sink unless the caller configured one explicitly.
 */
function withDefaultObservabilityComponent(baseDir, components) {
  if (components.some((component) => component?.kind === "observability")) return components
  return [...components, defaultObservabilityComponent(baseDir)]
}

/**
 * Create the OpenCode filesystem observability defaults.
 */
function defaultObservabilityComponent(baseDir) {
  return normalizePluginComponent(
    baseDir,
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
    "plugins.components[0]",
  )
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
 * Keep provider config out of telemetry while preserving useful identity.
 */
function compactProvider(provider) {
  if (!provider || typeof provider !== "object") return undefined
  return toJsonSafe({
    id: provider.id,
    source: provider.source,
    env: provider.env,
  })
}

/**
 * Keep model config out of telemetry while preserving useful identity.
 */
function compactModel(model) {
  if (!model || typeof model !== "object") return undefined
  return toJsonSafe({
    id: model.id ?? model.modelID,
    modelID: model.modelID,
    providerID: model.providerID ?? model.provider?.id,
    name: model.name,
    family: model.family,
  })
}

/**
 * Keep only LLM parameter fields that describe the current call.
 */
function compactParams(params) {
  if (!params || typeof params !== "object") return {}
  return toJsonSafe({
    temperature: params.temperature,
    topP: params.topP,
    topK: params.topK,
    maxOutputTokens: params.maxOutputTokens,
    options: params.options,
  })
}

/**
 * Read the OpenCode session ID from a bus event payload.
 */
function eventSessionID(event) {
  const props = event?.properties
  return props?.sessionID ?? props?.info?.id
}

/**
 * Return true for OpenCode helper calls that should not appear in the agent trajectory.
 */
function shouldSkipLlm(input) {
  return INTERNAL_LLM_AGENTS.has(input?.agent)
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

  await ensureObservabilityOutputDirectories(config)
  const activationReport = await pluginHost.initialize(config)
  await logDiagnostics(logger, activationReport.diagnostics)
  if (hasErrorDiagnostics(activationReport)) {
    await logger.warn("NeMo Flow plugin host initialized with error diagnostics")
  }
}

/**
 * Create filesystem output directories before exporter registration opens files.
 */
async function ensureObservabilityOutputDirectories(config) {
  const components = Array.isArray(config?.components) ? config.components : []
  for (const component of components) {
    if (component?.kind !== "observability" || component.enabled === false) continue
    const observabilityConfig = component.config
    for (const sectionName of ["atof", "atif"]) {
      const section = observabilityConfig?.[sectionName]
      if (section?.enabled !== true) continue
      const outputDirectory = section.output_directory
      if (typeof outputDirectory === "string" && outputDirectory.trim() !== "") {
        await fs.mkdir(outputDirectory, { recursive: true })
      }
    }
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
   * Keep NeMo Flow observability failures from changing OpenCode hook behavior.
   */
  function runtimeCall(warnKey, message, callback) {
    try {
      return callback()
    } catch (error) {
      void logger.warnOnce(warnKey, message, error).catch(() => {})
      return undefined
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
      toolCalls: new Map(),
      pendingLlm: undefined,
      messages: new Map(),
      lastUserMessage: undefined,
      sequence: 0,
    }

    const scope = runtimeCall(`scope-push:${sessionID}`, "failed to start OpenCode session scope", () =>
      withStack(session, () =>
        lib.pushScope(
          "opencode.session",
          lib.ScopeType?.Agent ?? 0,
          null,
          null,
          { sessionID },
          inputSessionMetadata(sessionID, session),
          { sessionID, source: "opencode" },
        ),
      ),
    )
    if (!scope) return undefined
    session.scope = scope
    sessions.set(sessionID, session)
    return session
  }

  /**
   * Emit an OpenCode milestone as a NeMo Flow mark event.
   */
  function emitMark(session, name, data, metadata = {}) {
    if (!session?.scope) return
    runtimeCall(`mark:${name}`, "failed to record OpenCode mark event", () =>
      withStack(session, () =>
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
        ),
      ),
    )
  }

  /**
   * Record message and part bus events for later LLM response reconstruction.
   */
  function recordMessageEvent(session, event) {
    const props = event?.properties ?? {}
    session.sequence += 1

    if (event.type === "message.updated" && props.info) {
      const info = toJsonSafe(props.info)
      const messageID = info?.id
      if (!messageID) return
      const message = session.messages.get(messageID) ?? { id: messageID, parts: new Map(), firstSequence: session.sequence }
      message.info = info
      message.role = info.role
      message.agent = info.agent
      message.model = info.model
      message.tokens = info.tokens
      message.cost = info.cost
      session.messages.set(messageID, message)
      return
    }

    if (event.type === "message.part.updated" && props.part) {
      const part = toJsonSafe(props.part)
      const messageID = part?.messageID
      const partID = part?.id
      if (!messageID || !partID) return
      const message = session.messages.get(messageID) ?? { id: messageID, parts: new Map(), firstSequence: session.sequence }
      const existing = message.parts.get(partID) ?? { id: partID, firstSequence: session.sequence }
      message.parts.set(partID, {
        ...existing,
        ...part,
        firstSequence: existing.firstSequence,
        updatedSequence: session.sequence,
      })
      session.messages.set(messageID, message)
      return
    }

    if (event.type === "message.part.delta") {
      const partID = props.partID
      const messageID = props.messageID
      if (!partID || !messageID || props.field !== "text") return
      const message = session.messages.get(messageID) ?? { id: messageID, parts: new Map(), firstSequence: session.sequence }
      const existing = message.parts.get(partID) ?? {
        id: partID,
        messageID,
        sessionID: session.id,
        type: "text",
        firstSequence: session.sequence,
      }
      message.parts.set(partID, {
        ...existing,
        text: `${existing.text ?? ""}${props.delta ?? ""}`,
        updatedSequence: session.sequence,
      })
      session.messages.set(messageID, message)
      return
    }

    if (event.type === "message.part.removed") {
      const partID = props.partID ?? props.part?.id
      const messageID = props.messageID ?? props.part?.messageID
      if (partID && messageID) session.messages.get(messageID)?.parts.delete(partID)
    }
  }

  /**
   * Build a compact NeMo Flow LLM request from OpenCode chat params.
   */
  function buildLlmRequest(session, input, output) {
    const userMessage = session.lastUserMessage
    const promptText = textFromParts(userMessage?.parts)
    return toJsonSafe({
      headers: {},
      content: {
        source: "opencode.chat.params",
        agent: input.agent,
        messageID: input.message?.id ?? userMessage?.message?.id ?? userMessage?.input?.messageID,
        provider: compactProvider(input.provider),
        model: compactModel(input.model),
        params: compactParams(output),
        messages: promptText
          ? [
              {
                role: userMessage?.message?.role ?? input.message?.role ?? "user",
                content: promptText,
              },
            ]
          : [],
      },
    })
  }

  /**
   * Build a compact NeMo Flow LLM response from OpenCode message bus state.
   */
  function buildLlmResponse(session, pending, reason) {
    const messages = assistantMessagesSince(session, pending.sequenceStart)
    const content = messages.flatMap((message) => messageParts(message, "text")).map((part) => part.text).join("\n")
    const toolCalls = messages.flatMap((message) => toolCallsFromMessage(session, message, pending.sequenceStart))
    const usage = usageFromMessages(messages)

    return toJsonSafe({
      role: "assistant",
      content: content || undefined,
      tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
      usage,
      opencode: {
        close_reason: reason,
        agent: pending.agent,
        messageID: pending.messageID,
      },
    })
  }

  /**
   * Start a semantic LLM span for a user-visible OpenCode model call.
   */
  function startLlm(session, input, output) {
    if (typeof lib.llmCall !== "function" || shouldSkipLlm(input)) return
    closeActiveLlm(session, "next_llm_request")

    const model = modelName(input.model) ?? session.model
    const metadata = toJsonSafe({
      source: "opencode.chat.params",
      sessionID: session.id,
      agent: input.agent,
      messageID: input.message?.id,
      model,
    })
    const handle = runtimeCall("llm-start", "failed to start OpenCode LLM span", () =>
      withStack(session, () =>
        lib.llmCall(
          input.provider?.id ?? input.model?.providerID ?? "opencode",
          buildLlmRequest(session, input, output),
          session.scope,
          null,
          null,
          metadata,
          model ?? null,
          null,
        ),
      ),
    )
    if (!handle) return
    session.pendingLlm = {
      handle,
      agent: input.agent,
      messageID: input.message?.id,
      model,
      sequenceStart: session.sequence + 1,
      metadata,
    }
  }

  /**
   * Finish the active LLM span before a tool starts, another LLM starts, or the session closes.
   */
  function closeActiveLlm(session, reason) {
    const pending = session?.pendingLlm
    if (!pending || typeof lib.llmCallEnd !== "function") return
    const response = buildLlmResponse(session, pending, reason)
    runtimeCall("llm-end", "failed to close OpenCode LLM span", () =>
      withStack(session, () => lib.llmCallEnd(pending.handle, response, null, pending.metadata, null)),
    )
    session.pendingLlm = undefined
  }

  /**
   * Extract text from OpenCode message parts.
   */
  function textFromParts(parts) {
    if (!Array.isArray(parts)) return ""
    return parts
      .filter((part) => part?.type === "text" && typeof part.text === "string")
      .map((part) => part.text)
      .join("\n")
  }

  function assistantMessagesSince(session, sequenceStart) {
    return [...session.messages.values()]
      .filter((message) => message.role === "assistant")
      .filter((message) => [...message.parts.values()].some((part) => part.firstSequence >= sequenceStart))
  }

  function messageParts(message, type) {
    return [...message.parts.values()].filter((part) => part.type === type && typeof part.text === "string")
  }

  function toolCallsFromMessage(session, message, sequenceStart) {
    return [...message.parts.values()]
      .filter((part) => part.type === "tool" && part.firstSequence >= sequenceStart)
      .filter((part) => part.callID || part.tool)
      .map((part) => {
        const observed = part.callID ? session.toolCalls.get(part.callID) : undefined
        return {
          id: part.callID ?? "",
          type: "function",
          function: {
            name: observed?.tool ?? part.tool ?? "",
            arguments: JSON.stringify(observed?.args ?? part.state?.input ?? {}),
          },
        }
      })
  }

  function usageFromMessages(messages) {
    const message = [...messages].reverse().find((item) => item.tokens || item.cost !== undefined)
    if (!message) return undefined
    const input = Number.isFinite(message.tokens?.input) ? message.tokens.input : undefined
    const output = Number.isFinite(message.tokens?.output) ? message.tokens.output : undefined
    const cacheRead = Number.isFinite(message.tokens?.cache?.read) ? message.tokens.cache.read : 0
    const cacheWrite = Number.isFinite(message.tokens?.cache?.write) ? message.tokens.cache.write : 0
    return toJsonSafe({
      input_tokens: input,
      output_tokens: output,
      cached_tokens: cacheRead + cacheWrite,
      reasoning_tokens: message.tokens?.reasoning,
      cost_usd: message.cost,
    })
  }

  /**
   * Close an OpenCode session scope so generic observability plugins can flush.
   */
  function flushSession(sessionID, reason) {
    const session = sessions.get(sessionID)
    if (!session) return
    recentFlushes.set(sessionID, Date.now())
    pruneRecentFlushes()
    closeActiveLlm(session, reason)
    for (const [key, tool] of session.pendingTools) {
      runtimeCall(`tool-close:${key}`, "failed to close pending OpenCode tool span", () =>
        withStack(session, () =>
          lib.toolCallEnd(
            tool.handle,
            { status: "unknown", reason: "session flushed before tool.execute.after" },
            null,
            { source: "opencode", sessionID, callID: tool.callID },
          ),
        ),
      )
    }
    session.pendingTools.clear()

    if (session.scope) {
      runtimeCall(`scope-pop:${sessionID}`, "failed to close OpenCode session scope", () =>
        withStack(session, () => lib.popScope(session.scope, { sessionID, reason }, null)),
      )
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
     * Observe OpenCode bus events for response reconstruction and session closure.
     */
    async recordEvent(event) {
      if (closed || !OBSERVED_EVENTS.has(event?.type)) return
      const sessionID = eventSessionID(event)
      if (!sessionID) return
      if (shouldFlushEvent(event) && wasRecentlyFlushed(sessionID)) return
      const props = event.properties ?? {}
      const session = ensureSession(sessionID, {
        agent: typeof props.info?.agent === "string" ? props.info.agent : undefined,
        model: modelName(props.info?.model),
      })
      if (!session) return

      if (event.type.startsWith("message.")) {
        recordMessageEvent(session, event)
      } else if (event.type === "session.error") {
        emitMark(
          session,
          "opencode.session.error",
          {
            id: event.id,
            error: props.error,
          },
          eventMetadata(session, { eventType: event.type }),
        )
      }

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
        agent: agentName(input, agentName(output)),
        model: modelName(input.model ?? output?.message?.model),
      })
      if (!session) return
      session.lastUserMessage = toJsonSafe({
        input,
        message: output?.message,
        parts: output?.parts,
      })
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
      startLlm(session, input, output)
    },

    /**
     * Start a NeMo Flow tool span for an OpenCode tool call.
     */
    async recordToolBefore(input, output) {
      if (closed) return
      const session = ensureSession(input.sessionID)
      if (!session) return
      const args = toJsonSafe(output?.args)
      if (input.callID) {
        session.toolCalls.set(input.callID, { tool: input.tool, args })
      }
      closeActiveLlm(session, "tool_start")
      const handle = runtimeCall(`tool-start:${input.callID ?? input.tool}`, "failed to start OpenCode tool span", () =>
        withStack(session, () =>
          lib.toolCall(
            input.tool,
            args,
            session.scope,
            null,
            { sessionID: input.sessionID, callID: input.callID },
            { source: "opencode", sessionID: input.sessionID, callID: input.callID },
            input.callID,
            null,
          ),
        ),
      )
      if (!handle) return
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
        if (input.callID) {
          session.toolCalls.set(input.callID, { tool: input.tool, args })
        }
        const handle = runtimeCall(`tool-start:${input.callID ?? input.tool}`, "failed to start OpenCode tool span", () =>
          withStack(session, () =>
            lib.toolCall(
              input.tool,
              args,
              session.scope,
              null,
              { sessionID: input.sessionID, callID: input.callID },
              { source: "opencode", sessionID: input.sessionID, callID: input.callID, recovered: true },
              input.callID,
              null,
            ),
          ),
        )
        if (!handle) return
        pending = { handle, callID: input.callID, tool: input.tool, args }
      }
      runtimeCall(`tool-end:${input.callID ?? input.tool}`, "failed to close OpenCode tool span", () =>
        withStack(session, () =>
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
          ),
        ),
      )
      session.pendingTools.delete(input.callID)
    },

    /**
     * Flush open sessions and unregister exporters during plugin shutdown.
     */
    async close() {
      if (closed) return
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
  let started = false
  const listener = () => {
    if (started) return
    started = true
    void close().catch((error) => {
      void logger.warnOnce("before-exit-cleanup", "failed to clean up NeMo Flow OpenCode plugin", error)
    })
  }
  process.on("beforeExit", listener)
  process.on("exit", listener)
  return () => {
    process.removeListener("beforeExit", listener)
    process.removeListener("exit", listener)
  }
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
