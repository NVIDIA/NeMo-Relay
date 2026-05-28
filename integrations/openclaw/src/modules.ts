// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Dynamic module loading boundary for NeMo Relay Node bindings.
 *
 * Keeping imports behind this loader lets the plugin register in OpenClaw even
 * when the native binding is unavailable, then degrade only at runtime start.
 */
import type * as NemoRelayRuntime from 'nemo-relay-node';
import type * as NemoRelayAdaptive from 'nemo-relay-node/adaptive';
import type * as NemoRelayPluginHost from 'nemo-relay-node/plugin';

type NemoRelayRuntimeKeys =
  | 'ScopeType'
  | 'createScopeStack'
  | 'currentScopeStack'
  | 'setThreadScopeStack'
  | 'pushScope'
  | 'popScope'
  | 'event'
  | 'llmCall'
  | 'llmCallEnd'
  | 'toolCall'
  | 'toolCallEnd'
  | 'toolConditionalExecution';

type NemoRelayPluginHostKeys = 'defaultConfig' | 'validate' | 'initialize' | 'clear';
type NemoRelayAdaptiveKeys = 'ADAPTIVE_PLUGIN_KIND' | 'ComponentSpec';

export type ConfigDiagnostic = NemoRelayPluginHost.ConfigDiagnostic;
export type ConfigReport = NemoRelayPluginHost.ConfigReport;

/**
 * @internal Package-owned subset of the dynamically imported `nemo-relay-node`
 * namespace used by this integration.
 */
export type NemoRelayRuntimeModule = Omit<Pick<typeof NemoRelayRuntime, NemoRelayRuntimeKeys>, 'ScopeType'> & {
  ScopeType:
    | {
        Agent?: Parameters<typeof NemoRelayRuntime.pushScope>[1];
      }
    | undefined;
  flushSubscribers?: typeof NemoRelayRuntime.flushSubscribers;
};

/**
 * @internal Package-owned subset of the dynamically imported
 * `nemo-relay-node/plugin` namespace used by this integration.
 */
export type NemoRelayPluginHostModule = Pick<typeof NemoRelayPluginHost, NemoRelayPluginHostKeys>;

/**
 * @internal Adaptive helper subset loaded so the package verifies the built-in
 * adaptive plugin path is available alongside the generic plugin host.
 */
export type NemoRelayAdaptiveModule = Pick<typeof NemoRelayAdaptive, NemoRelayAdaptiveKeys>;

export type NemoRelayModules = {
  nf: NemoRelayRuntimeModule;
  pluginHost: NemoRelayPluginHostModule;
  adaptive: NemoRelayAdaptiveModule;
};

export type NemoRelayModuleLoader = () => Promise<NemoRelayModules>;

/** Load the runtime and plugin-host modules used by the OpenClaw integration. */
export const defaultNemoRelayModuleLoader: NemoRelayModuleLoader = async () => {
  const [nf, pluginHost, adaptive] = await Promise.all([
    import('nemo-relay-node'),
    import('nemo-relay-node/plugin'),
    import('nemo-relay-node/adaptive'),
  ]);

  return {
    nf: nf as NemoRelayRuntimeModule,
    pluginHost: pluginHost as NemoRelayPluginHostModule,
    adaptive: adaptive as NemoRelayAdaptiveModule,
  };
};
