// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Dynamic module loading boundary for NeMo Flow Node bindings.
 *
 * Keeping imports behind this loader lets the plugin register in OpenClaw even
 * when the native binding is unavailable, then degrade only at runtime start.
 */
import type * as NemoFlowRuntime from "nemo-flow-node";
import type * as NemoFlowPluginHost from "nemo-flow-node/plugin";

type NemoFlowRuntimeKeys =
  | "ScopeType"
  | "createScopeStack"
  | "currentScopeStack"
  | "setThreadScopeStack"
  | "pushScope"
  | "popScope"
  | "event"
  | "llmCall"
  | "llmCallEnd"
  | "toolCall"
  | "toolCallEnd"
  | "AtifExporter"
  | "OpenTelemetrySubscriber"
  | "OpenInferenceSubscriber";

type NemoFlowPluginHostKeys = "defaultConfig" | "validate" | "initialize" | "clear";

export type ConfigDiagnostic = NemoFlowPluginHost.ConfigDiagnostic;
export type ConfigReport = NemoFlowPluginHost.ConfigReport;

/**
 * @internal Package-owned subset of the dynamically imported `nemo-flow-node`
 * namespace used by this integration.
 */
export type NemoFlowRuntimeModule = Omit<Pick<typeof NemoFlowRuntime, NemoFlowRuntimeKeys>, "ScopeType"> & {
  ScopeType: {
    Agent?: Parameters<typeof NemoFlowRuntime.pushScope>[1];
  } | undefined;
};

/**
 * @internal Package-owned subset of the dynamically imported
 * `nemo-flow-node/plugin` namespace used by this integration.
 */
export type NemoFlowPluginHostModule = Pick<typeof NemoFlowPluginHost, NemoFlowPluginHostKeys>;

/** @internal Subscriber surface used by runtime shutdown and health tracking. */
export type NemoFlowSubscriber = Pick<
  InstanceType<typeof NemoFlowRuntime.OpenTelemetrySubscriber>,
  "register" | "deregister" | "forceFlush" | "shutdown"
>;

/** @internal ATIF exporter surface used by per-session capture/export. */
export type AtifExporterLike = Pick<
  InstanceType<typeof NemoFlowRuntime.AtifExporter>,
  "register" | "deregister" | "exportJson" | "clear"
>;

export type NemoFlowModules = {
  nf: NemoFlowRuntimeModule;
  pluginHost: NemoFlowPluginHostModule;
};

export type NemoFlowModuleLoader = () => Promise<NemoFlowModules>;

/** Load the runtime and plugin-host modules used by the OpenClaw integration. */
export const defaultNemoFlowModuleLoader: NemoFlowModuleLoader = async () => {
  const [nf, pluginHost] = await Promise.all([
    import("nemo-flow-node"),
    import("nemo-flow-node/plugin"),
  ]);

  return {
    nf: nf as NemoFlowRuntimeModule,
    pluginHost: pluginHost as NemoFlowPluginHostModule,
  };
};
