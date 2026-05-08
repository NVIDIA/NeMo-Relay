// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as path from "node:path";

import type {
  OpenClawPluginApi,
  OpenClawPluginServiceContext,
  PluginLogger,
  PluginRuntimeLifecycleRegistration,
} from "openclaw/plugin-sdk/plugin-entry";

import { parseConfig } from "./config.js";
import type { NemoFlowHookBackendConfig } from "./config.js";
import { createHealthSnapshot, type HookReplayBackendStatus } from "./health.js";
import type { HookReplayCounters } from "./hook-replay/session.js";
import { HookReplayBackend } from "./hooks-backend.js";
import {
  defaultNemoFlowModuleLoader,
  type ConfigDiagnostic,
  type NemoFlowModules,
  type NemoFlowModuleLoader,
} from "./modules.js";
import {
  registerTelemetrySubscribers,
  shutdownTelemetrySubscribers,
  type TelemetrySubscriberEntry,
} from "./telemetry.js";
import type { RuntimeStateOptions, StartContext } from "./types.js";

const SERVICE_ID = "nemo-flow-observability";
const LIFECYCLE_ID = "nemo-flow-observability-cleanup";
const STATUS_METHOD = "nemoFlow.status";
type RuntimeCleanupContext = Parameters<NonNullable<PluginRuntimeLifecycleRegistration["cleanup"]>>[0];

export class NemoFlowRuntimeState {
  private readonly api: OpenClawPluginApi;
  private readonly config: NemoFlowHookBackendConfig;
  private readonly moduleLoader: NemoFlowModuleLoader;
  private loadPromise: Promise<NemoFlowModules> | undefined;
  private startPromise: Promise<void> | undefined;
  private statusValue: HookReplayBackendStatus = { state: "not_initialized" };
  private modulesValue?: NemoFlowModules;
  private backendValue: HookReplayBackend | undefined;
  private initializedPluginHost = false;
  private started = false;
  private beforeExitListener?: () => void;
  private unavailableLogged = false;
  private missingStartContextLogged = false;
  private telemetrySubscribers: TelemetrySubscriberEntry[] = [];
  private lastStartContext?: StartContext;
  private lastCounters?: HookReplayCounters;
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
    const backendState = this.backendValue?.state();
    return createHealthSnapshot({
      status: this.statusValue,
      initializedPluginHost: this.initializedPluginHost,
      config: this.config,
      degradedOutputs: this.degradedOutputs,
      ...(backendState === undefined
        ? this.lastCounters === undefined
          ? {}
          : { counters: this.lastCounters }
        : {
            counters: backendState.counters,
            sessions: backendState.sessions.values(),
          }),
    });
  }

  async start(ctx: StartContext): Promise<void> {
    this.lastStartContext = copyStartContext(ctx);
    this.missingStartContextLogged = false;

    if (this.started || this.statusValue.state === "ready" || this.statusValue.state === "degraded") {
      return;
    }

    if (this.startPromise) {
      await this.startPromise;
      return;
    }

    this.startPromise = this.startInternal(ctx);
    try {
      await this.startPromise;
    } finally {
      this.startPromise = undefined;
    }
  }

  private async startInternal(ctx: StartContext): Promise<void> {
    delete this.lastCounters;
    this.degradedOutputs.clear();

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

    const degradedOutputCount = this.degradedOutputs.size;
    this.telemetrySubscribers = registerTelemetrySubscribers({
      nf: modules.nf,
      config: this.config,
      logger: ctx.logger,
      markOutputDegraded: (output) => this.markOutputDegraded(output),
    });
    if (this.degradedOutputs.size > degradedOutputCount && degradedReason === undefined) {
      degradedReason = "one or more NeMo Flow telemetry outputs failed to initialize";
    }

    this.backendValue = new HookReplayBackend({
      nf: modules.nf,
      config: this.config,
      logger: ctx.logger,
      agentVersion: ctx.agentVersion,
      resolvedAtifOutputDir: resolveAtifOutputDir(this.config, ctx),
      markOutputDegraded: (output) => this.markOutputDegraded(output),
    });
    this.registerBeforeExit(ctx.logger);
    this.started = true;
    this.statusValue = degradedReason === undefined ? { state: "ready" } : { state: "degraded", reason: degradedReason };
  }

  async stop(reason: string, logger?: PluginLogger): Promise<void> {
    await this.stopWithStatus(reason, logger, { state: "stopped", reason });
  }

  private async stopWithStatus(
    reason: string,
    logger: PluginLogger | undefined,
    finalStatus: HookReplayBackendStatus,
  ): Promise<void> {
    if (
      this.statusValue.state === "stopped" ||
      this.statusValue.state === "disabled" ||
      this.statusValue.state === "stopping"
    ) {
      return;
    }

    if (this.startPromise) {
      await this.startPromise.catch((error) => {
        const log = logger ?? this.api.logger;
        log.warn?.(`failed to finish NeMo Flow startup before stop: ${toMessage(error)}`);
      });
    }

    this.statusValue = { state: "stopping" };
    const log = logger ?? this.api.logger;
    this.removeBeforeExitListener();

    try {
      await this.backendValue?.drainForGatewayStop(reason);
    } catch (error) {
      log.warn?.(`failed to stop NeMo Flow hook backend: ${toMessage(error)}`);
    }
    const backendState = this.backendValue?.state();
    if (backendState) {
      this.lastCounters = { ...backendState.counters };
    }
    this.backendValue = undefined;

    shutdownTelemetrySubscribers({
      subscribers: this.telemetrySubscribers,
      logger: log,
      markOutputDegraded: (output) => this.markOutputDegraded(output),
    });
    this.telemetrySubscribers = [];

    if (this.initializedPluginHost && this.modulesValue) {
      try {
        this.modulesValue.pluginHost.clear();
      } catch (error) {
        log.warn?.(`failed to clear NeMo Flow plugin host: ${toMessage(error)}`);
      }
      this.initializedPluginHost = false;
    }

    this.started = false;
    this.statusValue = finalStatus;
  }

  async cleanup(ctx: RuntimeCleanupContext): Promise<void> {
    if (ctx.sessionKey !== undefined || ctx.runId !== undefined) {
      await this.backendValue?.cleanupSession({
        reason: ctx.reason,
        ...(ctx.sessionKey === undefined ? {} : { sessionKey: ctx.sessionKey }),
        ...(ctx.runId === undefined ? {} : { runId: ctx.runId }),
      });
      return;
    }

    await this.stopWithStatus(
      ctx.reason,
      this.api.logger,
      ctx.reason === "restart" ? { state: "not_initialized", reason: "restart" } : { state: "stopped", reason: ctx.reason },
    );
  }

  private async backendForHook(workspaceDir?: string): Promise<HookReplayBackend | undefined> {
    if (this.backendValue) {
      return this.backendValue;
    }

    if (this.statusValue.state === "disabled" || this.statusValue.state === "stopping") {
      return undefined;
    }

    const startContext = this.lastStartContext ?? this.startContextFromRuntime(workspaceDir);
    if (!startContext) {
      if (!this.missingStartContextLogged) {
        this.api.logger.warn?.("nemo-flow skipped hook replay because OpenClaw service start context is unavailable");
        this.missingStartContextLogged = true;
      }
      return undefined;
    }

    await this.start(startContext);
    return this.backendValue;
  }

  private async replayWithBackend(
    label: string,
    workspaceDir: string | undefined,
    emit: (backend: HookReplayBackend) => void,
  ): Promise<void> {
    const backend = await this.backendForHook(workspaceDir);
    if (!backend) {
      return;
    }

    backend.safeReplay(label, undefined, () => emit(backend));
  }

  private async replayWithBackendAsync(
    label: string,
    workspaceDir: string | undefined,
    emit: (backend: HookReplayBackend) => Promise<void>,
  ): Promise<void> {
    const backend = await this.backendForHook(workspaceDir);
    if (!backend) {
      return;
    }

    await backend.safeReplayAsync(label, undefined, () => emit(backend));
  }

  registerHooks(): void {
    this.api.on("gateway_start", async (event, ctx) => {
      await this.replayWithBackend("gateway_start", ctx.workspaceDir, (backend) =>
        backend.onGatewayStart(event, ctx),
      );
    });

    this.api.on("gateway_stop", async (event) => {
      await this.stop(event.reason ?? "gateway_stop", this.api.logger);
    });

    this.api.on("session_start", async (event, ctx) => {
      await this.replayWithBackend("session_start", undefined, (backend) => backend.onSessionStart(event, ctx));
    });

    this.api.on("session_end", async (event, ctx) => {
      await this.replayWithBackendAsync("session_end", undefined, (backend) => backend.onSessionEnd(event, ctx));
    });

    this.api.on("llm_input", async (event, ctx) => {
      await this.replayWithBackend("llm_input", ctx.workspaceDir, (backend) => backend.onLlmInput(event, ctx));
    });

    this.api.on("llm_output", async (event, ctx) => {
      await this.replayWithBackend("llm_output", ctx.workspaceDir, (backend) => backend.onLlmOutput(event, ctx));
    });

    this.api.on("model_call_started", async (event, ctx) => {
      await this.replayWithBackend("model_call_started", ctx.workspaceDir, (backend) =>
        backend.onModelCallStarted(event, ctx),
      );
    });

    this.api.on("model_call_ended", async (event, ctx) => {
      await this.replayWithBackend("model_call_ended", ctx.workspaceDir, (backend) =>
        backend.onModelCallEnded(event, ctx),
      );
    });

    this.api.on("after_tool_call", async (event, ctx) => {
      await this.replayWithBackend("after_tool_call", undefined, (backend) =>
        backend.onAfterToolCall(event, ctx),
      );
    });

    this.api.on("agent_end", async (event, ctx) => {
      await this.replayWithBackend("agent_end", ctx.workspaceDir, (backend) => backend.onAgentEnd(event, ctx));
    });

    this.api.on("before_agent_finalize", async (event, ctx) => {
      await this.replayWithBackend("before_agent_finalize", ctx.workspaceDir, (backend) =>
        backend.onBeforeAgentFinalize(event, ctx),
      );
    });

    this.api.on("subagent_spawned", async (event, ctx) => {
      await this.replayWithBackend("subagent_spawned", undefined, (backend) =>
        backend.onSubagentSpawned(event, ctx),
      );
    });

    this.api.on("subagent_ended", async (event, ctx) => {
      await this.replayWithBackend("subagent_ended", undefined, (backend) =>
        backend.onSubagentEnded(event, ctx),
      );
    });
  }

  private resolvePluginHostConfig(
    modules: NemoFlowModules,
    logger: PluginLogger,
  ): {
    hostConfig: Parameters<NemoFlowModules["pluginHost"]["validate"]>[0];
    degradedReason?: string;
  } {
    const configured = this.config.nemoFlow.pluginConfig;

    if (configured.components.length === 0) {
      return { hostConfig: modules.pluginHost.defaultConfig() };
    }

    const validationReport = validatePluginHostConfig(
      modules,
      configured as Parameters<NemoFlowModules["pluginHost"]["validate"]>[0],
      logger,
    );
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

  private startContextFromRuntime(workspaceDir?: string): StartContext | undefined {
    try {
      const stateDir = this.api.runtime.state.resolveStateDir();
      return {
        stateDir,
        logger: this.api.logger,
        resolvePath: this.api.resolvePath,
        agentVersion: this.config.atif.agentVersion ?? this.api.version ?? "unknown",
        ...(workspaceDir === undefined ? {} : { workspaceDir }),
      };
    } catch (error) {
      this.api.logger.warn?.(`nemo-flow could not resolve OpenClaw runtime state dir: ${toMessage(error)}`);
      return undefined;
    }
  }

  private registerBeforeExit(logger: PluginLogger): void {
    if (this.beforeExitListener) {
      return;
    }
    const listener = () => {
      void this.stop("beforeExit", logger).catch((error) => {
        logger.warn?.(`nemo-flow beforeExit cleanup failed: ${toMessage(error)}`);
      });
    };
    process.on("beforeExit", listener);
    this.beforeExitListener = listener;
  }

  private removeBeforeExitListener(): void {
    if (!this.beforeExitListener) {
      return;
    }
    process.removeListener("beforeExit", this.beforeExitListener);
    delete this.beforeExitListener;
  }
}

export function registerNemoFlowPlugin(
  api: OpenClawPluginApi,
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
    start: (ctx: OpenClawPluginServiceContext) =>
      runtime.start({
        stateDir: ctx.stateDir,
        logger: ctx.logger,
        resolvePath: api.resolvePath,
        agentVersion: config.atif.agentVersion ?? api.version ?? "unknown",
        ...(ctx.workspaceDir === undefined ? {} : { workspaceDir: ctx.workspaceDir }),
      }),
    stop: (ctx: OpenClawPluginServiceContext) => runtime.stop("service_stop", ctx.logger),
  });

  api.registerRuntimeLifecycle({
    id: LIFECYCLE_ID,
    description: "Clean up NeMo Flow OpenClaw observability plugin state",
    cleanup: (ctx) => runtime.cleanup(ctx),
  });

  api.registerGatewayMethod?.(
    STATUS_METHOD,
    ({ respond }) => {
      respond(true, runtime.health());
    },
    {
      scope: "operator.admin",
    },
  );

  runtime.registerHooks();
}

function validatePluginHostConfig(
  modules: NemoFlowModules,
  config: Parameters<NemoFlowModules["pluginHost"]["validate"]>[0],
  logger: PluginLogger,
) {
  const report = modules.pluginHost.validate(config);
  logDiagnostics(logger, report.diagnostics);
  return report;
}

function logDiagnostics(logger: PluginLogger, diagnostics: ConfigDiagnostic[]): void {
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

function copyStartContext(ctx: StartContext): StartContext {
  return {
    stateDir: ctx.stateDir,
    logger: ctx.logger,
    resolvePath: ctx.resolvePath,
    agentVersion: ctx.agentVersion,
    ...(ctx.workspaceDir === undefined ? {} : { workspaceDir: ctx.workspaceDir }),
  };
}

function toMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
