// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Dynamic module loading boundary for NeMo Relay Node bindings.
 *
 * Keeping imports behind this loader lets the plugin load in DeepSeek Harness
 * even when the native binding is unavailable, then degrade only at runtime
 * start. Tests inject fake modules through the same boundary.
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import type * as NemoRelayRuntime from 'nemo-relay-node';
import type * as NemoRelayPluginHost from 'nemo-relay-node/plugin';

type NemoRelayRuntimeKeys =
  | 'createScopeStack'
  | 'withScopeStack'
  | 'pushScope'
  | 'popScope'
  | 'event'
  | 'llmStreamCallExecute'
  | 'toolCallExecuteAsync'
  | 'toolConditionalExecution'
  | 'registerSubscriber'
  | 'deregisterSubscriber'
  | 'flushSubscribers'
  | 'registerLlmSanitizeRequestGuardrail'
  | 'deregisterLlmSanitizeRequestGuardrail'
  | 'registerLlmSanitizeResponseGuardrail'
  | 'deregisterLlmSanitizeResponseGuardrail'
  | 'registerToolSanitizeRequestGuardrail'
  | 'deregisterToolSanitizeRequestGuardrail'
  | 'registerToolSanitizeResponseGuardrail'
  | 'deregisterToolSanitizeResponseGuardrail'
  | 'registerMarkSanitizeGuardrail'
  | 'deregisterMarkSanitizeGuardrail'
  | 'registerToolConditionalExecutionGuardrail'
  | 'deregisterToolConditionalExecutionGuardrail'
  | 'OpenTelemetrySubscriber'
  | 'OpenAIChatCodec';

type NemoRelayPluginHostKeys = 'validate' | 'initialize';

export type ConfigDiagnostic = NemoRelayPluginHost.ConfigDiagnostic;
export type ConfigReport = NemoRelayPluginHost.ConfigReport;
export type PluginHostActivation = NemoRelayPluginHost.PluginHostActivation;
export type PluginHostReport = NemoRelayPluginHost.PluginHostReport;
export type PluginConfig = NemoRelayPluginHost.PluginConfig;
export type ScopeHandle = ReturnType<typeof NemoRelayRuntime.pushScope>;
export type ScopeStack = ReturnType<typeof NemoRelayRuntime.createScopeStack>;
export type LlmStream = Awaited<ReturnType<typeof NemoRelayRuntime.llmStreamCallExecute>>;
export type ToolExecutionResult = Awaited<ReturnType<typeof NemoRelayRuntime.toolCallExecuteAsync>>;
/** JSON value as the binding declares it. */
export type RelayJson = NemoRelayRuntime.Json;
/** LLM request (`headers` + `content`) as the binding passes it to codec callbacks. */
export type LlmRequestShape = NemoRelayRuntime.LlmRequestShape;

/**
 * @internal Package-owned subset of the dynamically imported `nemo-relay-node`
 * namespace used by this integration.
 */
export type NemoRelayRuntimeModule = Pick<typeof NemoRelayRuntime, NemoRelayRuntimeKeys> & {
  ScopeType:
    | {
        Custom?: Parameters<typeof NemoRelayRuntime.pushScope>[1];
      }
    | undefined;
};

/**
 * @internal Package-owned subset of the dynamically imported
 * `nemo-relay-node/plugin` namespace used by this integration.
 */
export type NemoRelayPluginHostModule = Pick<typeof NemoRelayPluginHost, NemoRelayPluginHostKeys>;

export type NemoRelayModules = {
  nf: NemoRelayRuntimeModule;
  pluginHost: NemoRelayPluginHostModule;
  /** `nemo-relay-node` package version, when it can be read. */
  version?: string;
};

export type NemoRelayModuleLoader = () => Promise<NemoRelayModules>;

/** Load the runtime and plugin-host modules used by the DeepSeek Harness integration. */
export const defaultNemoRelayModuleLoader: NemoRelayModuleLoader = async () => {
  const version = await bindingVersion();
  // Checked before the native library loads: a conflicting binding aborts the process while it initializes.
  const conflict = runtimeOwnerConflict(process.env[RUNTIME_OWNER_ENV], process.pid, version);
  if (conflict !== undefined) {
    throw new RuntimeOwnerConflictError(conflict);
  }
  const [nf, pluginHost] = await Promise.all([import('nemo-relay-node'), import('nemo-relay-node/plugin')]);

  return {
    nf: nf as NemoRelayRuntimeModule,
    pluginHost: pluginHost as NemoRelayPluginHostModule,
    ...(version === undefined ? {} : { version }),
  };
};

/**
 * Process-wide owner token the NeMo Relay runtime publishes when a binding loads
 * (`crates/runtime/src/shared_runtime.rs`): `pid=<pid>;binding=<kind>;version=<major>`.
 */
export const RUNTIME_OWNER_ENV = 'NEMO_RELAY_RUNTIME_OWNER';

/** Another NeMo Relay runtime that this plugin's binding cannot coexist with already owns the process. */
export class RuntimeOwnerConflictError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RuntimeOwnerConflictError';
  }
}

/**
 * Why this plugin's binding must not load because another NeMo Relay runtime
 * already owns this process; `undefined` when it may load.
 *
 * NeMo Relay allows one runtime per process. A later binding of another kind or
 * major version fails while its native library initializes, and the Node.js
 * binding then aborts the whole process. A NeMo Relay plugin that was disabled or
 * replaced keeps its binding loaded until the process exits, so swapping another
 * NeMo Relay plugin for this one while DeepSeek Harness runs would stop it.
 *
 * @param token - This process's `NEMO_RELAY_RUNTIME_OWNER` value.
 * @param pid - This process's id; Relay reclaims a token another process left in the inherited environment.
 * @param bindingVersion - Version of the `nemo-relay-node` package this plugin loads, when known.
 * @returns An actionable explanation of the conflict, or `undefined`.
 */
export function runtimeOwnerConflict(
  token: string | undefined,
  pid: number,
  bindingVersion: string | undefined,
): string | undefined {
  const fields = new Map<string, string>();
  for (const field of (token ?? '').split(';')) {
    const separator = field.indexOf('=');
    if (separator > 0) {
      fields.set(field.slice(0, separator), field.slice(separator + 1));
    }
  }
  const kind = fields.get('binding');
  const major = fields.get('version');
  // Relay replaces a malformed token, and one another process left behind, without failing.
  if (Number(fields.get('pid')) !== pid || !kind || !major) {
    return undefined;
  }
  const ours = bindingVersion?.split('.')[0];
  if (kind === 'node' && (ours === undefined || major === ours)) {
    return undefined;
  }
  const loading = ours === undefined ? "this plugin's binding" : `this plugin's version ${ours} binding`;
  return `Another NeMo Relay runtime (${kind} binding, version ${major}) already runs in this DeepSeek Harness process, for example from a NeMo Relay plugin that was disabled or replaced after DeepSeek Harness started. NeMo Relay allows one runtime per process, and loading ${loading} now would stop DeepSeek Harness, so NeMo Relay stays off. Restart DeepSeek Harness to finish the switch.`;
}

/** Oldest Node.js release `nemo-relay-node` supports (its `engines.node`). */
export const MINIMUM_NODE_VERSION = '24.13.1';

/** Native binding targets published for `nemo-relay-node`, by `<platform>-<arch>`. */
const BINDING_TARGETS: Record<string, string[]> = {
  'darwin-arm64': ['darwin-arm64'],
  'linux-x64': ['linux-x64-gnu', 'linux-x64-musl'],
  'linux-arm64': ['linux-arm64-gnu', 'linux-arm64-musl'],
  'win32-x64': ['win32-x64-msvc'],
  'win32-arm64': ['win32-arm64-msvc'],
};

export type HostRuntime = { node: string; platform: string; arch: string };

const CURRENT_RUNTIME: HostRuntime = { node: process.versions.node, platform: process.platform, arch: process.arch };

/**
 * Why the binding cannot run here, before loading it; `undefined` when it can.
 * A too-old Node.js is refused without importing the native addon.
 */
export function bindingPrecondition(runtime: HostRuntime = CURRENT_RUNTIME): string | undefined {
  if (compareVersions(runtime.node, MINIMUM_NODE_VERSION) < 0) {
    return `NeMo Relay needs Node.js ${MINIMUM_NODE_VERSION} or newer, but this DeepSeek Harness runs Node.js ${runtime.node}. Update DeepSeek Harness, then restart it.`;
  }
  if (BINDING_TARGETS[`${runtime.platform}-${runtime.arch}`] === undefined) {
    return `NeMo Relay has no native binding for ${runtime.platform}-${runtime.arch}. It supports macOS on Apple silicon, Linux x64 and arm64, and Windows x64 and arm64.`;
  }
  return undefined;
}

/** An actionable explanation of a binding that failed to load. */
export function describeBindingFailure(error: unknown, runtime: HostRuntime = CURRENT_RUNTIME): string {
  if (error instanceof RuntimeOwnerConflictError) {
    return error.message;
  }
  const precondition = bindingPrecondition(runtime);
  if (precondition !== undefined) {
    return precondition;
  }
  const message = error instanceof Error ? error.message : String(error);
  const host = `${runtime.platform}-${runtime.arch}`;
  if (/Cannot find native binding|Cannot find module|MODULE_NOT_FOUND|ERR_MODULE_NOT_FOUND/.test(message)) {
    const packages = (BINDING_TARGETS[host] ?? []).map((target) => `nemo-relay-node-${target}`).join(' or ');
    return `The NeMo Relay native binding for ${host} is not installed (npm package ${packages}). Reinstall the plugin so npm installs it, then restart DeepSeek Harness. Cause: ${firstLine(message)}`;
  }
  return `nemo-relay-node could not load on ${host} with Node.js ${runtime.node}: ${firstLine(message)}`;
}

function compareVersions(left: string, right: string): number {
  const parse = (version: string) =>
    version
      .replace(/^v/, '')
      .split(/[.+-]/)
      .slice(0, 3)
      .map((part) => Number.parseInt(part, 10) || 0);
  const [a, b] = [parse(left), parse(right)];
  for (let index = 0; index < 3; index += 1) {
    const difference = (a[index] ?? 0) - (b[index] ?? 0);
    if (difference !== 0) {
      return difference;
    }
  }
  return 0;
}

function firstLine(text: string): string {
  return text.split('\n')[0]?.trim() ?? text;
}

/** How far above the entry point the package manifest is searched for. */
const MANIFEST_SEARCH_DEPTH = 4;

/**
 * Version of the `nemo-relay-node` package this plugin loads, without loading it.
 *
 * The package does not export `./package.json`, so the manifest is found above
 * the resolved entry point, wherever the package keeps that file. When it
 * cannot be read, the version this plugin declares as its dependency stands in:
 * the runtime owner check needs the major version before the binding loads.
 */
async function bindingVersion(): Promise<string | undefined> {
  try {
    let directory = path.dirname(fileURLToPath(import.meta.resolve('nemo-relay-node')));
    for (let depth = 0; depth < MANIFEST_SEARCH_DEPTH; depth += 1) {
      const manifest = await readManifest(path.join(directory, 'package.json'));
      if (manifest?.name === 'nemo-relay-node') {
        return typeof manifest.version === 'string' ? manifest.version : undefined;
      }
      const parent = path.dirname(directory);
      if (parent === directory) {
        break;
      }
      directory = parent;
    }
  } catch {
    // Not resolvable: fall through to the declared dependency.
  }
  return declaredBindingVersion();
}

/** The `nemo-relay-node` version in this plugin's own manifest, such as `1.0.0` from `^1.0.0`. */
async function declaredBindingVersion(): Promise<string | undefined> {
  // This module is `<package>/dist/src/modules.js`.
  const manifest = await readManifest(fileURLToPath(new URL('../../package.json', import.meta.url)));
  const dependencies = manifest?.dependencies as Record<string, unknown> | undefined;
  const declared = dependencies?.['nemo-relay-node'];
  return typeof declared === 'string' ? /\d+(?:\.\d+)*/.exec(declared)?.[0] : undefined;
}

async function readManifest(file: string): Promise<Record<string, unknown> | undefined> {
  try {
    const manifest: unknown = JSON.parse(await readFile(file, 'utf8'));
    return manifest !== null && typeof manifest === 'object' ? (manifest as Record<string, unknown>) : undefined;
  } catch {
    return undefined;
  }
}
