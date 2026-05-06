// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseConfig } from "./config.js";
import { createHealthSnapshot, type HookReplayBackendStatus } from "./health.js";
import {
  defaultNemoFlowModuleLoader,
  type ConfigDiagnostic,
  type NemoFlowModules,
  type NemoFlowModuleLoader,
} from "./modules.js";
import type {
  OpenClawPluginApiLike,
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
  private readonly moduleLoader: NemoFlowModuleLoader;
  private statusValue: HookReplayBackendStatus = { state: "not_initialized" };
  private modulesValue?: NemoFlowModules;
  private initializedPluginHost = false;

  constructor(options: RuntimeStateOptions) {
    this.api = options.api;
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
    if (this.statusValue.state === "ready" || this.statusValue.state === "degraded") {
      return;
    }

    let modules: NemoFlowModules;
    try {
      modules = await this.moduleLoader();
      this.modulesValue = modules;
    } catch (error) {
      this.statusValue = { state: "degraded", reason: `failed to load nemo-flow-node: ${toMessage(error)}` };
      ctx.logger.warn?.(this.statusValue.reason);
      return;
    }

    const { hostConfig, degradedReason } = this.resolvePluginHostConfig(modules, ctx.logger);
    if (degradedReason) {
      this.statusValue = { state: "degraded", reason: degradedReason };
    }

    const validationReport = validatePluginHostConfig(modules, hostConfig, ctx.logger);

    if (validationReport.diagnostics.some((diagnostic) => diagnostic.level === "error")) {
      this.statusValue = {
        state: "degraded",
        reason: "NeMo Flow plugin host config validation failed",
      };
      return;
    }

    try {
      const activationReport = await modules.pluginHost.initialize(hostConfig);
      logDiagnostics(ctx.logger, activationReport.diagnostics);
      this.initializedPluginHost = true;
    } catch (error) {
      this.statusValue = {
        state: "degraded",
        reason: `failed to initialize NeMo Flow plugin host: ${toMessage(error)}`,
      };
      ctx.logger.warn?.(this.statusValue.reason);
      return;
    }

    if (!degradedReason) {
      this.statusValue = { state: "ready" };
    }
  }

  async stop(reason: string, logger?: PluginLoggerLike): Promise<void> {
    if (this.statusValue.state === "stopped" || this.statusValue.state === "disabled") {
      return;
    }

    this.statusValue = { state: "stopping" };

    if (this.initializedPluginHost && this.modulesValue) {
      try {
        this.modulesValue.pluginHost.clear();
      } catch (error) {
        logger?.warn?.(`failed to clear NeMo Flow plugin host: ${toMessage(error)}`);
      }
      this.initializedPluginHost = false;
    }

    this.statusValue = { state: "stopped", reason };
  }

  cleanup(reason: string): Promise<void> {
    return this.stop(reason, this.api.logger);
  }

  private resolvePluginHostConfig(
    modules: NemoFlowModules,
    logger: PluginLoggerLike,
  ): {
    hostConfig: { version: number; components: unknown[]; [key: string]: unknown };
    degradedReason?: string;
  } {
    const configured = parseConfig(this.api.pluginConfig).nemoFlow.pluginConfig;

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

function toMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
