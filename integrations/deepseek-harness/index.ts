// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * DeepSeek Harness plugin entry point.
 *
 * This file stays small: it parses the Loader row config, creates the runtime,
 * the rules, and Phoenix, and connects DeepSeek Harness plugin events, the
 * `/relay` command, the agent's `nemo_relay` tool and `nemo-relay` skill, and
 * the conversation-view projection to them. Lifecycle, scope, and middleware
 * behavior live in `src/runtime.ts`.
 */
import path from 'node:path';

import {
  agentToolPreExecute,
  registerAgentTool,
  type AgentToolHost,
  type AgentToolOperations,
} from './src/agent-tool.js';
import { RELAY_COMMAND_HINT, runRelayCommand, type RelayCommandContext } from './src/commands.js';
import { parseConfig, type NemoRelayDeepSeekHarnessConfig } from './src/config.js';
import { loadConfigSchema } from './src/config-schema.js';
import type {
  Agent,
  CommandDefinition,
  GenerateOptions,
  HarnessContext,
  HarnessToolResult,
  PostToolDecision,
  PreStepDecision,
  PreStepPayload,
  PreToolDecision,
  ProjectionDefinition,
  Session,
  SessionEvent,
  SessionHeader,
  StreamChunk,
  ToolExecution,
} from './src/dsh-types.js';
import { PhoenixManager, type PhoenixDeps } from './src/phoenix.js';
import { atifSessionDirectory } from './src/plugin-doc.js';
import {
  ConfigRulesStore,
  FileRulesStore,
  GuardrailPolicy,
  RULES_GUARDRAIL_NAME,
  RULES_GUARDRAIL_PRIORITY,
  type ConfigEditorLike,
  type RulesStore,
} from './src/policy.js';
import { ATIF_TRAJECTORY_FILES, pruneOldFiles, VIEW_TRACE_FILES } from './src/retention.js';
import { NemoRelayDeepSeekHarnessRuntime, toMessage, type RuntimeOptions } from './src/runtime.js';
import { registerRelaySkill, type RelaySkillHost } from './src/skill.js';
import { ActionLog, buildStatusSnapshot, type PhoenixSnapshot, type StatusSnapshot } from './src/status.js';
import { pruneStatusFiles, StatusFile } from './src/status-file.js';
import { SessionTraceWriter } from './src/view.js';

export const name = 'nemo-relay';

/**
 * Schemastery schema DeepSeek Harness uses to validate the Loader row and render
 * its settings; `undefined` when `@deepseek-ai/schemastery` cannot be resolved,
 * in which case `parseConfig` alone validates the row.
 */
export const Config: unknown = await loadConfigSchema();

/** Session projection read by the conversation view to locate a session's trace. */
export const VIEW_PROJECTION_KEY = 'nemoRelayTrace';
/** Version of the projection's wire view and of the status file (Host ↔ Client contract). */
export const CONTRACT_VERSION = 4;

const PASS_SCHEMA = { parse: (value: unknown) => value };
/** How often a running plugin removes files past the retention limit; it also does so at start. */
const RETENTION_INTERVAL_MS = 24 * 60 * 60 * 1000;

export type ApplyOptions = Pick<RuntimeOptions, 'moduleLoader'> & {
  /** Process and network seams of the Phoenix manager (tests). */
  phoenixDeps?: Partial<PhoenixDeps>;
  /** Loads `@deepseek-ai/dsh-tools` for the agent tool; without it the tool definition is built locally (tests replace it). */
  importTools?: () => Promise<unknown>;
};

/**
 * Activate the NeMo Relay integration in one DeepSeek Harness Host composition.
 *
 * DeepSeek Harness treats an unhandled rejection in the Host as fatal, so no
 * failure of this plugin, including a failing logger, may escape to the Host.
 */
export function apply(ctx: HarnessContext, rawConfig: unknown, options: ApplyOptions = {}): void {
  const logger = createLogger(ctx);
  let config: NemoRelayDeepSeekHarnessConfig;
  try {
    config = parseConfig(rawConfig);
  } catch (error) {
    logger.warn(`NeMo Relay is disabled: invalid plugin configuration: ${toMessage(error)}`);
    return;
  }
  // Conversation view traces stay apart from exporter outputs configured under `plugins`.
  const viewDir = path.join(config.dataDir, 'view');
  const writer = config.view.enabled ? new SessionTraceWriter(viewDir, logger.warn) : undefined;
  // One status file per process: DeepSeek Harness processes can share a data directory.
  const statusFile = new StatusFile(config.dataDir);
  // Structured results of `/relay` actions; the view reads its own by request id from the status file.
  const actions = new ActionLog();

  let publish = (): void => undefined;
  const policy = new GuardrailPolicy({
    enabled: config.guardrails.enabled,
    failMode: () => config.guardrails.failMode(),
    store: createRulesStore(ctx, config, logger.warn),
    warn: logger.warn,
    onChange: () => publish(),
  });
  const runtime = new NemoRelayDeepSeekHarnessRuntime({
    config,
    logger,
    sessionLookup: (sessionId) => sessionsService(ctx)?.get(sessionId),
    defaultModel: () => defaultModel(ctx),
    ...(writer === undefined ? {} : { onEvent: (event: unknown) => writer.accept(event) }),
    ...(config.guardrails.enabled
      ? {
          guardrail: {
            name: RULES_GUARDRAIL_NAME,
            priority: RULES_GUARDRAIL_PRIORITY,
            evaluate: (toolName: string, args: unknown) => policy.evaluate(toolName, args),
          },
        }
      : {}),
    ...(options.moduleLoader === undefined ? {} : { moduleLoader: options.moduleLoader }),
  });
  // Phoenix is offered but nothing runs until the user starts it, or it was connected before.
  const phoenix: PhoenixManager | undefined = config.phoenix.enabled
    ? new PhoenixManager({
        config: config.phoenix,
        dataDir: config.dataDir,
        exporter: runtime.phoenixExporter(),
        log: logger.warn,
        onChange: () => publish(),
        ...(options.phoenixDeps === undefined ? {} : { deps: options.phoenixDeps }),
      })
    : undefined;

  const snapshot = (): StatusSnapshot =>
    buildStatusSnapshot({
      health: runtime.health(),
      config,
      plan: runtime.pluginPlan(),
      resolved: runtime.resolvedPluginConfig(),
      relayVersion: runtime.relayVersion(),
      guardrails: policy.snapshot(),
      phoenix: phoenix?.snapshot() ?? phoenixOff(config),
      viewDirectory: viewDir,
      statusFile: statusFile.path,
      actions: actions.list(),
      configSchema: Config !== undefined,
    });
  const writeStatus = (): void => {
    try {
      statusFile.write(`${JSON.stringify(snapshot(), null, 2)}\n`);
    } catch (error) {
      logger.warn(`NeMo Relay status write failed: ${toMessage(error)}`);
    }
  };
  publish = writeStatus;

  // Per-conversation files accumulate for as long as DeepSeek Harness is used: drop the old ones.
  const applyRetention = (): void => {
    const atif = runtime.pluginPlan()?.atif;
    void pruneOldFiles(
      [
        { directory: viewDir, match: VIEW_TRACE_FILES },
        // Only the trajectories of this plugin's default ATIF output; a configured exporter owns its own files.
        ...(atif?.applied === true && atif.directory !== undefined
          ? [{ directory: atif.directory, match: ATIF_TRAJECTORY_FILES, nested: true }]
          : []),
      ],
      config.retention.days,
    )
      .then((result) => {
        if (result.removed > 0) {
          logger.info(
            `NeMo Relay removed ${result.removed} trace file(s) older than ${config.retention.days} days (${Math.round(result.bytes / 1e6)} MB)`,
          );
        }
      })
      .catch(() => undefined);
  };

  ctx.effect(() => {
    pruneStatusFiles(config.dataDir);
    writeStatus();
    const retentionTimer = setInterval(applyRetention, RETENTION_INTERVAL_MS);
    retentionTimer.unref?.();
    void runtime
      .start()
      .then(
        async () => {
          writeStatus();
          applyRetention();
          // Reconnect Phoenix the way the user last connected; never rejects.
          await phoenix?.autoConnect();
        },
        (error: unknown) => logger.warn(`NeMo Relay failed to start: ${toMessage(error)}`),
      )
      .catch(() => undefined);
    // Disposers run concurrently, so one disposer orders the teardown: Phoenix flushes before Relay stops.
    return async () => {
      clearInterval(retentionTimer);
      try {
        await phoenix?.dispose();
        await runtime.stop();
        await writer?.flush();
      } catch (error) {
        logger.warn(`NeMo Relay shutdown failed: ${toMessage(error)}`);
      }
      // A stopped process has no status: remove its file; `/relay json` answers from memory.
      try {
        statusFile.close();
      } catch (error) {
        logger.warn(`NeMo Relay could not remove its status file: ${toMessage(error)}`);
      }
    };
  }, 'nemo-relay: runtime');

  // Global listeners see every agent, including agents of presets with isolated services.
  const global = { global: true };
  ctx.on(
    'session/event',
    (session: Session, event: SessionEvent) => {
      runtime.onSessionEvent(session, event);
      if (event.type === 'turn/end') {
        writeStatus();
      }
    },
    global,
  );
  ctx.on(
    'agent/pre-step',
    async (payload: PreStepPayload, next: () => Promise<PreStepDecision>) => {
      // This listener only observes: the decision belongs to other plugins and passes through unchanged.
      const decision = await next();
      try {
        runtime.noteStep(payload?.agent, payload?.turn, payload?.step);
      } catch (error) {
        logger.warn(`NeMo Relay could not record the model step: ${toMessage(error)}`);
      }
      return decision;
    },
    global,
  );
  ctx.on(
    'agent/disposed',
    (payload: { agent: Agent }) => {
      try {
        runtime.onAgentDisposed(payload.agent);
      } catch (error) {
        logger.warn(`NeMo Relay could not close the agent's turn: ${toMessage(error)}`);
      }
    },
    global,
  );
  ctx.on(
    'llm/stream',
    (options: GenerateOptions, next: () => AsyncIterable<StreamChunk>) => runtime.streamLlm(options, next),
    global,
  );
  ctx.on(
    'tools/pre-execute',
    (exec: ToolExecution, next: () => Promise<PreToolDecision>) =>
      // The agent tool asks the user before loosening changes; Relay's own check then also sees that ask.
      runtime.preExecute(exec, config.agent.enabled ? () => agentToolPreExecute(exec, next) : next),
    global,
  );
  ctx.on(
    'tools/execute',
    (exec: ToolExecution, next: () => Promise<HarnessToolResult>) => runtime.executeTool(exec, next),
    global,
  );
  ctx.on(
    'tools/post-execute',
    (exec: ToolExecution, result: HarnessToolResult, next: () => Promise<PostToolDecision>) =>
      runtime.postExecute(exec, result, next),
    global,
  );

  // One set of operations behind the user's `/relay` command and the agent's `nemo_relay` tool.
  const commandContext = (sessionId: string | undefined): RelayCommandContext => ({
    snapshot,
    publish: writeStatus,
    recordAction: (entry) => actions.record(entry),
    policy,
    phoenix,
    sessionId,
    sessionTrace: writer === undefined || sessionId === undefined ? undefined : writer.pathFor(sessionId),
  });

  // `/relay` is the user's command; the agent cannot run commands.
  ctx.inject(['commands'], (scope) => {
    const command: CommandDefinition = {
      definitionId: 'nemo-relay-deepseek-harness',
      name: 'relay',
      description: 'NeMo Relay: status, guardrail rules, and Arize Phoenix',
      input: { hint: RELAY_COMMAND_HINT },
      // Rule edits carry JSON payloads; keep them out of the session's command log.
      recordInput: false,
      handler: ({ agent, rawInput }) =>
        runRelayCommand(
          rawInput,
          commandContext(typeof agent?.session?.id === 'string' ? agent.session.id : undefined),
        ),
    };
    scope.effect(() => {
      const unregister = scope.commands.register(command);
      policy.editable = true;
      writeStatus();
      return () => {
        policy.editable = false;
        unregister();
      };
    }, 'nemo-relay: /relay command');
  });

  // The agent can set up and change NeMo Relay for the user through the same operations;
  // `tools/pre-execute` above asks the user before any change that loosens protection.
  if (config.agent.enabled) {
    const operations: AgentToolOperations = {
      context: commandContext,
      setFailMode: (mode) => policy.setFailMode(mode),
    };
    ctx.inject(['tools'], (scope) => {
      scope.effect(
        () =>
          registerAgentTool(scope as AgentToolHost, {
            operations,
            warn: logger.warn,
            ...(options.importTools === undefined ? {} : { importTools: options.importTools }),
          }),
        'nemo-relay: nemo_relay tool',
      );
    });
    ctx.inject(['skills'], (scope) => {
      scope.effect(
        () => registerRelaySkill(scope as RelaySkillHost, { snapshot, warn: logger.warn }),
        'nemo-relay: nemo-relay skill',
      );
    });
  }

  if (writer !== undefined) {
    ctx.inject(['sessionProjections'], (scope) => {
      const definition: ProjectionDefinition = {
        key: VIEW_PROJECTION_KEY,
        stateVersion: 1,
        stateSchema: PASS_SCHEMA,
        init: (header: SessionHeader) => ({ id: String(header?.id ?? ''), seq: -1 }),
        // The sequence number changes the view on every committed event, so the view refreshes.
        apply: (state: unknown, event: SessionEvent) => {
          const current = state as { id: string; seq: number };
          return typeof event?.seq === 'number' && event.seq > current.seq
            ? { id: current.id, seq: event.seq }
            : current;
        },
        wire: {
          viewSchema: PASS_SCHEMA,
          view: (state: unknown) => {
            const current = state as { id: string; seq: number };
            const atif = runtime.pluginPlan()?.atif;
            return {
              v: CONTRACT_VERSION,
              sessionId: current.id,
              seq: current.seq,
              file: writer.pathFor(current.id),
              // The status file of the process serving this view.
              status: statusFile.path,
              ...(atif?.applied === true && atif.directory !== undefined
                ? { atifDir: atifSessionDirectory(atif.directory, current.id) }
                : {}),
            };
          },
        },
      };
      scope.sessionProjections.register(definition);
    });
  }
}

/**
 * Rules and the fail mode live in the plugin's own Loader row when DeepSeek
 * Harness offers a config editor for it, and in `<dataDir>/rules.json` otherwise.
 * Either store reads the other place while its own holds no rules, so rules
 * written in the plugin configuration apply without a config editor, and rules
 * saved to the file survive a later start that has one.
 */
function createRulesStore(
  ctx: HarnessContext,
  config: NemoRelayDeepSeekHarnessConfig,
  warn: (message: string) => void,
): RulesStore {
  const file = path.join(config.dataDir, 'rules.json');
  const configured = (): readonly unknown[] | undefined => config.guardrails.rules();
  let reason = 'DeepSeek Harness offers no config editor for this plugin entry';
  try {
    const editor = ctx.get('configEditor') as (Partial<ConfigEditorLike> & { entries?: () => unknown[] }) | undefined;
    const entry = ctx.fiber?.entry;
    if (
      editor !== undefined &&
      typeof editor.edit === 'function' &&
      entry !== undefined &&
      (typeof editor.entries !== 'function' || editor.entries().includes(entry))
    ) {
      return new ConfigRulesStore(
        configured,
        editor as ConfigEditorLike,
        entry,
        () => config.guardrails.failMode(),
        new FileRulesStore(file),
      );
    }
  } catch (error) {
    reason = `the config editor is unavailable (${toMessage(error)})`;
  }
  warn(
    `NeMo Relay saves rule edits in ${file}: ${reason}. Rules saved there replace guardrails.rules of the plugin configuration.`,
  );
  return new FileRulesStore(file, configured);
}

/** Phoenix section of the status when Phoenix support is turned off. */
function phoenixOff(config: NemoRelayDeepSeekHarnessConfig): PhoenixSnapshot {
  return {
    enabled: config.phoenix.enabled,
    state: config.phoenix.enabled ? 'stopped' : 'disabled',
    connected: false,
    url: config.phoenix.url,
    project: config.phoenix.project,
  };
}

/**
 * Log through the Cordis logger without ever throwing. Cordis logger methods
 * read `this`, so they must be called as methods of `ctx.logger`.
 */
function createLogger(ctx: HarnessContext): { info(message: string): void; warn(message: string): void } {
  const write = (level: 'info' | 'warn', message: string): void => {
    try {
      const logger = ctx.logger;
      if (logger !== undefined && typeof logger[level] === 'function') {
        logger[level](message);
        return;
      }
    } catch {
      // Fall back to the console below.
    }
    try {
      if (level === 'warn') {
        console.warn(`[nemo-relay] ${message}`);
      } else {
        console.info(`[nemo-relay] ${message}`);
      }
    } catch {
      // Nothing else can report this message.
    }
  };
  return { info: (message) => write('info', message), warn: (message) => write('warn', message) };
}

/** DeepSeek Harness's default model (`agentDefaultModel` service), when the profile has one. */
function defaultModel(ctx: HarnessContext): string | undefined {
  const service = ctx.get('agentDefaultModel') as { currentSelection?: () => { model?: unknown } } | undefined;
  const model = typeof service?.currentSelection === 'function' ? service.currentSelection().model : undefined;
  return typeof model === 'string' ? model : undefined;
}

function sessionsService(ctx: HarnessContext): { get(sessionId: string): Session | undefined } | undefined {
  const service = ctx.get('sessions') as { get?: (sessionId: string) => Session | undefined } | undefined;
  return typeof service?.get === 'function' ? { get: (sessionId) => service.get?.(sessionId) } : undefined;
}
