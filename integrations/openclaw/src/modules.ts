// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type ConfigDiagnostic = {
  level: "warning" | "error";
  code: string;
  component?: string;
  field?: string;
  message: string;
};

export type ConfigReport = {
  diagnostics: ConfigDiagnostic[];
};

export type NemoFlowPluginHostModule = {
  defaultConfig: () => { version: number; components: unknown[]; [key: string]: unknown };
  validate: (config: { version: number; components: unknown[]; [key: string]: unknown }) => ConfigReport;
  initialize: (
    config: { version: number; components: unknown[]; [key: string]: unknown },
  ) => Promise<ConfigReport>;
  clear: () => void;
};

export type NemoFlowRuntimeModule = {
  ScopeType?: {
    Agent?: number;
  };
  createScopeStack: () => unknown;
  currentScopeStack: () => unknown;
  setThreadScopeStack: (stack: unknown) => void;
  pushScope: (
    name: string,
    scopeType: number,
    handle?: unknown | null,
    attributes?: number | null,
    data?: unknown,
    metadata?: unknown,
    input?: unknown,
    timestamp?: number | null,
  ) => unknown;
  popScope: (handle: unknown, output?: unknown, timestamp?: number | null) => void;
  event: (
    name: string,
    handle?: unknown | null,
    data?: unknown,
    metadata?: unknown,
    timestamp?: number | null,
  ) => void;
};

export type NemoFlowModules = {
  nf: NemoFlowRuntimeModule;
  pluginHost: NemoFlowPluginHostModule;
};

export type NemoFlowModuleLoader = () => Promise<NemoFlowModules>;

export const defaultNemoFlowModuleLoader: NemoFlowModuleLoader = async () => {
  const [nf, pluginHost] = await Promise.all([
    import("nemo-flow-node"),
    import("nemo-flow-node/plugin"),
  ]);

  return {
    nf: nf as unknown as NemoFlowRuntimeModule,
    pluginHost: pluginHost as NemoFlowPluginHostModule,
  };
};
