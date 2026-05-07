// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as path from "node:path";

import { parseConfig } from "./config.js";
import type { NemoFlowHookBackendConfig } from "./config.js";
import { createHealthSnapshot, type HookReplayBackendStatus } from "./health.js";
import { HookReplayBackend } from "./hooks-backend.js";
import {
  defaultNemoFlowModuleLoader,
  type ConfigDiagnostic,
  type NemoFlowModules,
  type NemoFlowModuleLoader,
  type NemoFlowSubscriber,
} from "./modules.js";
import type {
  PluginHookAfterToolCallEvent,
  PluginHookAgentContext,
  PluginHookAgentEndEvent,
  PluginHookBeforeAgentFinalizeEvent,
  PluginHookGatewayContext,
  PluginHookGatewayStartEvent,
  PluginHookGatewayStopEvent,
  PluginHookLlmInputEvent,
  PluginHookLlmOutputEvent,
  PluginHookModelCallEndedEvent,
  PluginHookModelCallStartedEvent,
  PluginHookSessionContext,
  PluginHookSessionEndEvent,
  PluginHookSessionStartEvent,
  PluginHookSubagentContext,
  PluginHookSubagentEndedEvent,
  PluginHookSubagentSpawnedEvent,
  PluginHookToolContext,
} from "./openclaw-hook-types.js";
import type {
  OpenClawPluginApiLike,
  OpenClawHookHandlerLike,
  OpenClawPluginServiceContextLike,
  PluginLoggerLike,
  RuntimeStateOptions,
  StartContext,
} from "./types.js";

const PLUGIN_ID = "nemo-flow";
const SERVICE_ID = "nemo-flow-observability";
const LIFECYCLE_ID = "nemo-flow-observability-cleanup";
const STATUS_METHOD = "nemoFlow.status";

export class NemoFlowRuntimeState {
  private readonly api: OpenClawPluginApiLike;
  private readonly config: NemoFlowHookBackendConfig;
  private readonly moduleLoader: NemoFlowModuleLoader;
  private loadPromise: Promise<NemoFlowModules> | undefined;
  private statusValue: HookReplayBackendStatus = { state: "not_initialized" };
  private modulesValue?: NemoFlowModules;
  private backendValue: HookReplayBackend | undefined;
  private initializedPluginHost = false;
  private started = false;
  private beforeExitListener?: () => void;
  private unavailableLogged = false;
  private telemetrySubscribers: Array<{
    output: "otel" | "openInference";
    name: string;
    subscriber: NemoFlowSubscriber;
  }> = [];
  private readonly degradedOutputs = new Set<"atif" | "otel" | "openInference">();

  constructor(options: RuntimeStateOptions) {
    this.api = options.api;
    this.config = options.config;
    this.moduleLoader = options.moduleLoader ?? defaultNemoFlowModuleLoader;
  }

  status(): HookReplayBackendStatus {
    return this.statusValue;
  }

  health() {
    return createHealthSnapshot({
      status: this.statusValue,
      initializedPluginHost: this.initializedPluginHost,
    });
  }

  async start(ctx: StartContext): Promise<void> {
    if (this.started || this.statusValue.state === "ready" || this.statusValue.state === "degraded") {
      return;
    }

    let modules: NemoFlowModules;
    try {
      this.loadPromise ??= this.moduleLoader();
      modules = await this.loadPromise;
      this.modulesValue = modules;
    } catch (error) {
      this.loadPromise = undefined;
      this.statusValue = { state: "degraded", reason: `failed to load nemo-flow-node: ${toMessage(error)}` };
      if (!this.unavailableLogged) {
        ctx.logger.warn?.(this.statusValue.reason);
        this.unavailableLogged = true;
      }
      return;
    }

    const { hostConfig, degradedReason: configuredDegradedReason } = this.resolvePluginHostConfig(
      modules,
      ctx.logger,
    );
    let degradedReason = configuredDegradedReason;

    const validationReport = validatePluginHostConfig(modules, hostConfig, ctx.logger);

    if (validationReport.diagnostics.some((diagnostic) => diagnostic.level === "error")) {
      degradedReason = "NeMo Flow plugin host config validation failed";
    } else {
      if (
        validationReport.diagnostics.some((diagnostic) => diagnostic.level === "warning") &&
        degradedReason === undefined
      ) {
        degradedReason = "NeMo Flow plugin host config validation produced warnings";
      }

      try {
        const activationReport = await modules.pluginHost.initialize(hostConfig);
        logDiagnostics(ctx.logger, activationReport.diagnostics);
        this.initializedPluginHost = true;
        if (
          activationReport.diagnostics.some((diagnostic) => diagnostic.level === "error") &&
          degradedReason === undefined
        ) {
          degradedReason = "NeMo Flow plugin host initialization reported errors";
        }
      } catch (error) {
        degradedReason = `failed to initialize NeMo Flow plugin host: ${toMessage(error)}`;
        ctx.logger.warn?.(degradedReason);
      }
    }

    this.backendValue = new HookReplayBackend({
      nf: modules.nf,
      config: this.config,
      logger: ctx.logger,
      agentVersion: ctx.agentVersion,
      resolvedAtifOutputDir: resolveAtifOutputDir(this.config, ctx),
      markOutputDegraded: (output) => this.markOutputDegraded(output),
    });
    this.started = true;
    this.statusValue = degradedReason === undefined ? { state: "ready" } : { state: "degraded", reason: degradedReason };
  }

  async stop(reason: string, logger?: PluginLoggerLike): Promise<void> {
    if (
      this.statusValue.state === "stopped" ||
      this.statusValue.state === "disabled" ||
      this.statusValue.state === "stopping"
    ) {
      return;
    }

    this.statusValue = { state: "stopping" };
    const log = logger ?? this.api.logger;

    try {
      await this.backendValue?.drainForGatewayStop(reason);
    } catch (error) {
      log.warn?.(`failed to stop NeMo Flow hook backend: ${toMessage(error)}`);
    }
    this.backendValue = undefined;

    if (this.initializedPluginHost && this.modulesValue) {
      try {
        this.modulesValue.pluginHost.clear();
      } catch (error) {
        log.warn?.(`failed to clear NeMo Flow plugin host: ${toMessage(error)}`);
      }
      this.initializedPluginHost = false;
    }

    this.started = false;
    this.statusValue = { state: "stopped", reason };
  }

  cleanup(reason: string): Promise<void> {
    return this.stop(reason, this.api.logger);
  }

  registerHooks(): void {
    const dispatch = (
      hookName: string,
      handler: (backend: HookReplayBackend, event: unknown, ctx: unknown) => void,
    ): void => {
      this.api.on(hookName, ((event: unknown, ctx: unknown) => {
        const backend = this.backendValue;
        if (!backend) {
          return;
        }
        backend.safeReplay(hookName, undefined, () => handler(backend, event, ctx));
      }) as OpenClawHookHandlerLike);
    };
    const dispatchAsync = (
      hookName: string,
      handler: (backend: HookReplayBackend, event: unknown, ctx: unknown) => Promise<void>,
    ): void => {
      this.api.on(hookName, (async (event: unknown, ctx: unknown) => {
        const backend = this.backendValue;
        if (!backend) {
          return;
        }
        await backend.safeReplayAsync(hookName, undefined, () => handler(backend, event, ctx));
      }) as OpenClawHookHandlerLike);
    };

    dispatch("gateway_start", (backend, event, ctx) =>
      backend.onGatewayStart(event as PluginHookGatewayStartEvent, ctx as PluginHookGatewayContext),
    );
    this.api.on("gateway_stop", (async (event: unknown) => {
      const stopEvent = event as PluginHookGatewayStopEvent;
      await this.stop(stopEvent.reason ?? "gateway_stop", this.api.logger);
    }) as OpenClawHookHandlerLike);
    dispatch("session_start", (backend, event, ctx) =>
      backend.onSessionStart(event as PluginHookSessionStartEvent, ctx as PluginHookSessionContext),
    );
    dispatchAsync("session_end", (backend, event, ctx) =>
      backend.onSessionEnd(event as PluginHookSessionEndEvent, ctx as PluginHookSessionContext),
    );
    dispatch("llm_input", (backend, event, ctx) =>
      backend.onLlmInput(event as PluginHookLlmInputEvent, ctx as PluginHookAgentContext),
    );
    dispatch("llm_output", (backend, event, ctx) =>
      backend.onLlmOutput(event as PluginHookLlmOutputEvent, ctx as PluginHookAgentContext),
    );
    dispatch("model_call_started", (backend, event, ctx) =>
      backend.onModelCallStarted(event as PluginHookModelCallStartedEvent, ctx as PluginHookAgentContext),
    );
    dispatch("model_call_ended", (backend, event, ctx) =>
      backend.onModelCallEnded(event as PluginHookModelCallEndedEvent, ctx as PluginHookAgentContext),
    );
    dispatch("after_tool_call", (backend, event, ctx) =>
      backend.onAfterToolCall(event as PluginHookAfterToolCallEvent, ctx as PluginHookToolContext),
    );
    dispatch("agent_end", (backend, event, ctx) =>
      backend.onAgentEnd(event as PluginHookAgentEndEvent, ctx as PluginHookAgentContext),
    );
    dispatch("before_agent_finalize", (backend, event, ctx) =>
      backend.onBeforeAgentFinalize(
        event as PluginHookBeforeAgentFinalizeEvent,
        ctx as PluginHookAgentContext,
      ),
    );
    dispatch("subagent_spawned", (backend, event, ctx) =>
      backend.onSubagentSpawned(event as PluginHookSubagentSpawnedEvent, ctx as PluginHookSubagentContext),
    );
    dispatch("subagent_ended", (backend, event, ctx) =>
      backend.onSubagentEnded(event as PluginHookSubagentEndedEvent, ctx as PluginHookSubagentContext),
    );
  }

  private resolvePluginHostConfig(
    modules: NemoFlowModules,
    logger: PluginLoggerLike,
  ): {
    hostConfig: { version: number; components: unknown[]; [key: string]: unknown };
    degradedReason?: string;
  } {
    const configured = this.config.nemoFlow.pluginConfig;

    if (configured.components.length === 0) {
      return { hostConfig: modules.pluginHost.defaultConfig() };
    }

    const validationReport = validatePluginHostConfig(modules, configured, logger);
    const degradedReason =
      "nemoFlow.pluginConfig.components is not supported by the hook backend; using default NeMo Flow plugin host config";
    logger.warn?.(degradedReason);
    logDiagnostics(logger, validationReport.diagnostics);
    return {
      hostConfig: modules.pluginHost.defaultConfig(),
      degradedReason,
    };
  }

  private markOutputDegraded(output: "atif" | "otel" | "openInference"): void {
    this.degradedOutputs.add(output);
  }
}

export function registerNemoFlowPlugin(
  api: OpenClawPluginApiLike,
  moduleLoader?: NemoFlowModuleLoader,
): void {
  if (api.registrationMode !== "full") {
    return;
  }

  let config;
  try {
    config = parseConfig(api.pluginConfig);
  } catch (error) {
    api.logger.warn?.(
      `nemo-flow observability disabled because plugin config is invalid: ${toMessage(error)}`,
    );
    return;
  }

  if (!config.enabled) {
    api.logger.info?.("nemo-flow observability disabled by plugin config");
    return;
  }

  const runtime = new NemoFlowRuntimeState(
    moduleLoader === undefined ? { api, config } : { api, config, moduleLoader },
  );

  api.registerService({
    id: SERVICE_ID,
    start: (ctx: OpenClawPluginServiceContextLike) =>
      runtime.start({
        stateDir: ctx.stateDir,
        logger: ctx.logger,
        resolvePath: api.resolvePath,
        agentVersion: config.atif.agentVersion ?? api.version ?? "unknown",
        ...(ctx.workspaceDir === undefined ? {} : { workspaceDir: ctx.workspaceDir }),
      }),
    stop: (ctx: OpenClawPluginServiceContextLike) => runtime.stop("service_stop", ctx.logger),
  });

  api.registerRuntimeLifecycle({
    id: LIFECYCLE_ID,
    description: "Clean up NeMo Flow OpenClaw observability plugin state",
    cleanup: (ctx) => runtime.cleanup(ctx.reason),
  });

  api.registerGatewayMethod?.(STATUS_METHOD, () => runtime.health(), {
    scope: "operator.admin",
  });

  runtime.registerHooks();
}

function validatePluginHostConfig(
  modules: NemoFlowModules,
  config: { version: number; components: unknown[]; [key: string]: unknown },
  logger: PluginLoggerLike,
) {
  const report = modules.pluginHost.validate(config);
  logDiagnostics(logger, report.diagnostics);
  return report;
}

function logDiagnostics(logger: PluginLoggerLike, diagnostics: ConfigDiagnostic[]): void {
  for (const diagnostic of diagnostics) {
    const prefix = diagnostic.component ? `${diagnostic.component}: ` : "";
    const message = `${prefix}${diagnostic.code}: ${diagnostic.message}`;
    if (diagnostic.level === "error") {
      logger.warn?.(message);
    } else {
      logger.info?.(message);
    }
  }
}

function resolveAtifOutputDir(config: NemoFlowHookBackendConfig, ctx: StartContext): string {
  const configured = config.atif.outputDir;
  if (!configured) {
    return path.join(ctx.stateDir, "plugins", "nemo-flow", "atif");
  }
  return path.isAbsolute(configured) ? configured : ctx.resolvePath(configured);
}

function toMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
