// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

declare module "openclaw/plugin-sdk/plugin-entry" {
  export type OpenClawPluginApi =
    import("./types.js").OpenClawPluginApiLike;

  export type OpenClawPluginConfigSchema =
    import("./types.js").OpenClawPluginConfigSchemaLike;

  export function definePluginEntry(params: {
    id: string;
    name: string;
    description: string;
    configSchema?: OpenClawPluginConfigSchema;
    register: (api: OpenClawPluginApi) => void;
  }): unknown;
}

declare module "nemo-flow-node" {
  const runtimeModule: Record<string, unknown>;
  export default runtimeModule;
}

declare module "nemo-flow-node/plugin" {
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

  export function defaultConfig(): { version: number; components: unknown[]; [key: string]: unknown };

  export function validate(config: {
    version: number;
    components: unknown[];
    [key: string]: unknown;
  }): ConfigReport;

  export function initialize(config: {
    version: number;
    components: unknown[];
    [key: string]: unknown;
  }): Promise<ConfigReport>;

  export function clear(): void;
}
