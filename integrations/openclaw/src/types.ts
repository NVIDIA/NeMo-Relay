// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared runtime option and context types.
 *
 * These types avoid importing OpenClaw implementation modules outside the public
 * plugin SDK surface and keep runtime-state constructor signatures explicit.
 */
import type { OpenClawPluginApi, OpenClawPluginServiceContext } from "openclaw/plugin-sdk/plugin-entry";

import type { NemoFlowHookBackendConfig } from "./config.js";
import type { HookReplayBackendStatus } from "./health.js";
import type { NemoFlowModuleLoader } from "./modules.js";

export type RuntimeStateOptions = {
  api: OpenClawPluginApi;
  config: NemoFlowHookBackendConfig;
  moduleLoader?: NemoFlowModuleLoader;
};

export type StartContext = {
  stateDir: string;
  workspaceDir?: string;
  logger: OpenClawPluginServiceContext["logger"];
  resolvePath: OpenClawPluginApi["resolvePath"];
  agentVersion: string;
};

export type RuntimeStateSnapshot = {
  status: HookReplayBackendStatus;
  initializedPluginHost: boolean;
  unavailableReason?: string;
};
