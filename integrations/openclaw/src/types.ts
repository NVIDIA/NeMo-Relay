// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NemoFlowHookBackendConfig } from "./config.js";
import type { HookReplayBackendStatus, NemoFlowHealthSnapshot } from "./health.js";
import type { NemoFlowModuleLoader } from "./modules.js";

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type JsonRecord = { [key: string]: JsonValue };

export type PluginConfigValidation =
  | { ok: true; value?: unknown }
  | { ok: false; errors: string[] };

export type OpenClawPluginConfigSchemaLike = {
  safeParse?: (value: unknown) => {
    success: boolean;
    data?: unknown;
    error?: {
      issues?: Array<{ path: Array<string | number>; message: string }>;
    };
  };
  parse?: (value: unknown) => unknown;
  validate?: (value: unknown) => PluginConfigValidation;
  uiHints?: Record<string, unknown>;
  jsonSchema?: JsonRecord;
};

export type PluginLoggerLike = {
  debug?: (message: string) => void;
  info?: (message: string) => void;
  warn?: (message: string) => void;
  error?: (message: string) => void;
};

export type OpenClawPluginServiceContextLike = {
  stateDir: string;
  workspaceDir?: string;
  logger: PluginLoggerLike;
  config?: unknown;
};

export type OpenClawPluginServiceLike = {
  id: string;
  start: (ctx: OpenClawPluginServiceContextLike) => void | Promise<void>;
  stop?: (ctx: OpenClawPluginServiceContextLike) => void | Promise<void>;
};

export type OpenClawHookHandlerLike = (event: unknown, ctx: unknown) => void | Promise<void>;

export type OpenClawRuntimeCleanupContextLike = {
  reason: string;
  sessionKey?: string;
  runId?: string;
};

export type OpenClawPluginApiLike = {
  id: string;
  name?: string;
  version?: string;
  registrationMode: string;
  pluginConfig?: Record<string, unknown>;
  logger: PluginLoggerLike;
  resolvePath: (input: string) => string;
  registerService: (service: OpenClawPluginServiceLike) => void;
  registerRuntimeLifecycle: (lifecycle: {
    id: string;
    description?: string;
    cleanup: (ctx: OpenClawRuntimeCleanupContextLike) => void | Promise<void>;
  }) => void;
  on: (hookName: string, handler: OpenClawHookHandlerLike, opts?: { priority?: number; timeoutMs?: number }) => void;
  registerGatewayMethod?: (
    method: string,
    handler: () => NemoFlowHealthSnapshot | Promise<NemoFlowHealthSnapshot>,
    opts?: { scope?: string },
  ) => void;
};

export type RuntimeStateOptions = {
  api: OpenClawPluginApiLike;
  config: NemoFlowHookBackendConfig;
  moduleLoader?: NemoFlowModuleLoader;
};

export type StartContext = {
  stateDir: string;
  workspaceDir?: string;
  logger: PluginLoggerLike;
  resolvePath: (input: string) => string;
  agentVersion: string;
};

export type RuntimeStateSnapshot = {
  status: HookReplayBackendStatus;
  initializedPluginHost: boolean;
  unavailableReason?: string;
};
