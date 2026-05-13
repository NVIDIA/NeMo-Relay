// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Runtime lifecycle coordinator for the OpenClaw plugin.
 *
 * This module validates config, lazy-loads NeMo Flow Node bindings, registers
 * OpenClaw service/lifecycle/gateway surfaces, and forwards hooks to the replay
 * backend once runtime state is ready.
 */
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
import type { RuntimeStateOptions, StartContext } from "./types.js";

const SERVICE_ID = "nemo-flow-observability";
const LIFECYCLE_ID = "nemo-flow-observability-cleanup";
const STATUS_METHOD = "nemoFlow.status";
type RuntimeCleanupContext = Parameters<NonNullable<PluginRuntimeLifecycleRegistration["cleanup"]>>[0];

/** Owns one plugin runtime instance across OpenClaw service start/stop cycles. */
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
  private pluginHostOutputsHealthy = false;
  private started = false;
  private beforeExitListener?: () => void;
  private unavailableLogged = false;
  private missingStartContextLogged = false;
  private lastStartContext?: StartContext;
  private lastCounters?: HookReplayCounters;

  constructor(options: RuntimeStateOptions) {
    this.api = options.api;
    this.config = options.config;
    this.moduleLoader = options.moduleLoader ?? defaultNemoFlowModuleLoader;
  }

  /** Return the current coarse backend status. */
  status(): HookReplayBackendStatus {
    return this.statusValue;
  }

  /** Build the operator-facing health payload served through the gateway method. */
  health() {
    const backendState = this.backendValue?.state();
    return createHealthSnapshot({
      status: this.statusValue,
      initializedPluginHost: this.initializedPluginHost,
      pluginHostOutputsHealthy: this.pluginHostOutputsHealthy,
      config: this.config,
      ...(backendState === undefined
        ? this.lastCounters === undefined
          ? {}
          : { counters: this.lastCounters }
        : {
            counters: backendState.counters,
          }),
    });
  }

  /** Start NeMo Flow modules, generic plugins, and the hook replay backend. */
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

  /** Do the startup work behind a single-flight guard. */
  private async startInternal(ctx: StartContext): Promise<void> {
    delete this.lastCounters;
    this.initializedPluginHost = false;
    this.pluginHostOutputsHealthy = false;

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

    const hostConfig = this.config.plugins as Parameters<NemoFlowModules["pluginHost"]["validate"]>[0];
    let degradedReason;

    const validationReport = validatePluginHostConfig(modules, hostConfig, ctx.logger);

    if (!validationReport.ok) {
      degradedReason = validationReport.reason;
      ctx.logger.warn?.(degradedReason);
    } else if (validationReport.report.diagnostics.some((diagnostic) => diagnostic.level === "error")) {
      degradedReason = "NeMo Flow plugin host config validation failed";
    } else {
      if (
        validationReport.report.diagnostics.some((diagnostic) => diagnostic.level === "warning") &&
        degradedReason === undefined
      ) {
        degradedReason = "NeMo Flow plugin host config validation produced warnings";
      }

      try {
        const activationReport = await modules.pluginHost.initialize(hostConfig);
        logDiagnostics(ctx.logger, activationReport.diagnostics);
        this.initializedPluginHost = true;
        const hasInitializationErrors = activationReport.diagnostics.some((diagnostic) => diagnostic.level === "error");
        this.pluginHostOutputsHealthy = !hasInitializationErrors;
        if (hasInitializationErrors) {
          degradedReason ??= "NeMo Flow plugin host initialization reported errors";
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
    });
    this.registerBeforeExit(ctx.logger);
    this.started = true;
    this.statusValue = degradedReason === undefined ? { state: "ready" } : { state: "degraded", reason: degradedReason };
  }

  /** Stop the runtime because OpenClaw service or gateway shutdown is happening. */
  async stop(reason: string, logger?: PluginLogger): Promise<void> {
    await this.stopWithStatus(reason, logger, { state: "stopped", reason });
  }

  /** Shared stop implementation that controls the final health status. */
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

    if (this.initializedPluginHost && this.modulesValue) {
      try {
        this.modulesValue.pluginHost.clear();
      } catch (error) {
        log.warn?.(`failed to clear NeMo Flow plugin host: ${toMessage(error)}`);
      }
      this.initializedPluginHost = false;
      this.pluginHostOutputsHealthy = false;
    }

    this.started = false;
    this.statusValue = finalStatus;
  }

  /** Handle OpenClaw runtime lifecycle cleanup for either a session or the backend. */
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

  /** Return a backend for a hook, lazily starting from runtime context if needed. */
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

  /** Run a synchronous hook against the backend with fail-open replay handling. */
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

  /** Run an asynchronous hook against the backend with fail-open replay handling. */
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

  /** Register every OpenClaw hook used by the observability backend. */
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

    this.api.on("before_message_write", (event, ctx) => {
      const backend = this.backendValue;
      if (!backend) {
        return;
      }
      backend.safeReplay("before_message_write", undefined, () => backend.onBeforeMessageWrite(event, ctx));
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

  /** Reconstruct enough service-start context for hooks that arrive before service start. */
  private startContextFromRuntime(workspaceDir?: string): StartContext | undefined {
    try {
      const stateDir = this.api.runtime.state.resolveStateDir();
      return {
        stateDir,
        logger: this.api.logger,
        resolvePath: this.api.resolvePath,
        agentVersion: this.api.version ?? "unknown",
        ...(workspaceDir === undefined ? {} : { workspaceDir }),
      };
    } catch (error) {
      this.api.logger.warn?.(`nemo-flow could not resolve OpenClaw runtime state dir: ${toMessage(error)}`);
      return undefined;
    }
  }

  /** Register a process beforeExit cleanup guard for local OpenClaw shutdown paths. */
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

  /** Remove the beforeExit listener once normal shutdown begins. */
  private removeBeforeExitListener(): void {
    if (!this.beforeExitListener) {
      return;
    }
    process.removeListener("beforeExit", this.beforeExitListener);
    delete this.beforeExitListener;
  }
}

/** Register the NeMo Flow observability plugin with the OpenClaw plugin API. */
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
        agentVersion: api.version ?? "unknown",
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

/** Validate the NeMo Flow plugin-host config and log diagnostics. */
function validatePluginHostConfig(
  modules: NemoFlowModules,
  config: Parameters<NemoFlowModules["pluginHost"]["validate"]>[0],
  logger: PluginLogger,
):
  | { ok: true; report: ReturnType<NemoFlowModules["pluginHost"]["validate"]> }
  | { ok: false; reason: string } {
  try {
    const report = modules.pluginHost.validate(config);
    logDiagnostics(logger, report.diagnostics);
    return { ok: true, report };
  } catch (error) {
    return {
      ok: false,
      reason: `failed to validate NeMo Flow plugin host config: ${toMessage(error)}`,
    };
  }
}

/** Log plugin-host diagnostics at warning or info level based on severity. */
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

/** Copy service-start context so later lazy hook startup cannot mutate it. */
function copyStartContext(ctx: StartContext): StartContext {
  return {
    stateDir: ctx.stateDir,
    logger: ctx.logger,
    resolvePath: ctx.resolvePath,
    agentVersion: ctx.agentVersion,
    ...(ctx.workspaceDir === undefined ? {} : { workspaceDir: ctx.workspaceDir }),
  };
}

/** Convert thrown values into stable log strings. */
function toMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
