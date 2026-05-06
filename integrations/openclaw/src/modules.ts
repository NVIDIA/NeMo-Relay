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

export type NemoFlowRuntimeModule = Record<string, unknown>;

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
    nf,
    pluginHost: pluginHost as NemoFlowPluginHostModule,
  };
};
