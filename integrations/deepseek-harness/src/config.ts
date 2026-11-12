// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * User-facing configuration parsing for the DeepSeek Harness plugin.
 *
 * DeepSeek Harness passes the Loader row `config` object to the plugin, already
 * validated by the exported `Config` schema when DeepSeek Harness loads it.
 * `parseConfig` is the single normalizer: it applies defaults, rejects unknown
 * fields, and accepts both the raw row and the schema output, whose volatile
 * `guardrails.rules` and `guardrails.failMode` fields are references read with `get()`.
 */
import { homedir } from 'node:os';
import path from 'node:path';

export type CaptureConfig = {
  includePrompts: boolean;
  includeResponses: boolean;
  stripToolArgs: boolean;
  stripToolResults: boolean;
};

export type ViewConfig = {
  enabled: boolean;
};

export type RedactionConfig = {
  /** Add Relay's `pii_redaction` component with credential detectors when no layer configures redaction. */
  secrets: boolean;
};

export type FailMode = 'open' | 'closed';

export type AgentConfig = {
  /**
   * Offer the agent the `nemo_relay` tool and the `nemo-relay` skill, so a user
   * can ask it to inspect or change NeMo Relay. Changes that loosen protection
   * still ask the user first.
   */
  enabled: boolean;
};

export type RetentionConfig = {
  /**
   * Remove this plugin's per-conversation files (view traces and default ATIF
   * trajectories) that no turn has touched for this many days; `0` keeps them.
   */
  days: number;
};

export type GuardrailsConfig = {
  enabled: boolean;
  /**
   * Read the fail mode the Loader row currently configures (`open` by default).
   * In DeepSeek Harness the field is volatile, so a saved change is visible
   * here without restarting the plugin.
   */
  failMode(): FailMode;
  /**
   * Read the rules the Loader row currently configures, or `undefined` when it
   * configures none. In DeepSeek Harness the field is volatile, so a saved rule
   * change is visible here without restarting the plugin.
   */
  rules(): readonly unknown[] | undefined;
  /**
   * True when DeepSeek Harness validated the row with this plugin's `Config`
   * schema: the rules and fail mode then arrive as volatile references, and a
   * saved change applies in place. False for a plain row, where a save made
   * through the config editor restarts the plugin.
   */
  live: boolean;
};

export type PhoenixLaunchConfig = {
  enabled: boolean;
  version: string;
  python: string;
  /** Explicit command that serves Phoenix instead of `uv tool run`. */
  command?: string[];
  startupTimeoutMs: number;
};

export type PhoenixConfig = {
  enabled: boolean;
  /** Phoenix base URL without a trailing slash. */
  url: string;
  project: string;
  autoConnect: boolean;
  launch: PhoenixLaunchConfig;
};

export type NemoRelayPluginHostConfig = {
  version: number;
  components: unknown[];
  [key: string]: unknown;
};

export type NemoRelayDeepSeekHarnessConfig = {
  enabled: boolean;
  plugins: NemoRelayPluginHostConfig;
  capture: CaptureConfig;
  dataDir: string;
  view: ViewConfig;
  redaction: RedactionConfig;
  guardrails: GuardrailsConfig;
  phoenix: PhoenixConfig;
  agent: AgentConfig;
  retention: RetentionConfig;
};

export const DEFAULT_CAPTURE: CaptureConfig = {
  includePrompts: true,
  includeResponses: true,
  stripToolArgs: false,
  stripToolResults: false,
};

export const DEFAULT_RETENTION_DAYS = 30;
export const MAX_RETENTION_DAYS = 3650;
export const DEFAULT_PHOENIX_URL = 'http://127.0.0.1:6006';
export const DEFAULT_PHOENIX_PROJECT = 'deepseek-harness';
export const DEFAULT_PHOENIX_VERSION = '21.0.0';
export const DEFAULT_PHOENIX_PYTHON = '3.13';
export const DEFAULT_PHOENIX_STARTUP_TIMEOUT_MS = 240_000;
export const PHOENIX_STARTUP_TIMEOUT_RANGE = { min: 10_000, max: 1_800_000 } as const;

const PHOENIX_PROJECT = /^[A-Za-z0-9._ -]{1,80}$/;
const PHOENIX_VERSION = /^[0-9][0-9A-Za-z.+-]{0,39}$/;
const PYTHON_VERSION = /^[0-9]+(\.[0-9]+){0,2}$/;

/** Parse the DeepSeek Harness Loader row config into the normalized plugin config. */
export function parseConfig(value: unknown, env: NodeJS.ProcessEnv = process.env): NemoRelayDeepSeekHarnessConfig {
  const raw = asRecord(value, 'config', true);
  rejectUnknownFields(raw, 'config', [
    'enabled',
    'plugins',
    'capture',
    'dataDir',
    'view',
    'redaction',
    'guardrails',
    'phoenix',
    'agent',
    'retention',
  ]);
  const capture = asRecord(raw.capture, 'capture', true);
  rejectUnknownFields(capture, 'capture', ['includePrompts', 'includeResponses', 'stripToolArgs', 'stripToolResults']);
  const view = asRecord(raw.view, 'view', true);
  rejectUnknownFields(view, 'view', ['enabled']);
  const redaction = asRecord(raw.redaction, 'redaction', true);
  rejectUnknownFields(redaction, 'redaction', ['secrets']);
  const agent = asRecord(raw.agent, 'agent', true);
  rejectUnknownFields(agent, 'agent', ['enabled']);
  const retention = asRecord(raw.retention, 'retention', true);
  rejectUnknownFields(retention, 'retention', ['days']);
  const retentionDays = optionalNumber(retention.days, 'retention.days') ?? DEFAULT_RETENTION_DAYS;
  if (!Number.isInteger(retentionDays) || retentionDays < 0 || retentionDays > MAX_RETENTION_DAYS) {
    throw new Error(`retention.days must be a whole number from 0 to ${MAX_RETENTION_DAYS}`);
  }
  const dataDir = resolveDataDir(optionalString(raw.dataDir, 'dataDir'), env);

  return {
    enabled: optionalBoolean(raw.enabled, 'enabled') ?? true,
    plugins: resolveOutputDirectories(parsePluginHostConfig(raw.plugins), dataDir),
    capture: {
      includePrompts:
        optionalBoolean(capture.includePrompts, 'capture.includePrompts') ?? DEFAULT_CAPTURE.includePrompts,
      includeResponses:
        optionalBoolean(capture.includeResponses, 'capture.includeResponses') ?? DEFAULT_CAPTURE.includeResponses,
      stripToolArgs: optionalBoolean(capture.stripToolArgs, 'capture.stripToolArgs') ?? DEFAULT_CAPTURE.stripToolArgs,
      stripToolResults:
        optionalBoolean(capture.stripToolResults, 'capture.stripToolResults') ?? DEFAULT_CAPTURE.stripToolResults,
    },
    dataDir,
    view: {
      enabled: optionalBoolean(view.enabled, 'view.enabled') ?? true,
    },
    redaction: {
      secrets: optionalBoolean(redaction.secrets, 'redaction.secrets') ?? true,
    },
    guardrails: parseGuardrails(raw.guardrails),
    phoenix: parsePhoenix(raw.phoenix),
    agent: {
      enabled: optionalBoolean(agent.enabled, 'agent.enabled') ?? true,
    },
    retention: { days: retentionDays },
  };
}

/**
 * Resolve the plugin data directory: an explicit `dataDir` (with `~` expansion), otherwise
 * `<DSH_HOME>/nemo-relay`, where `DSH_HOME` defaults to `~/.dsh`.
 */
export function resolveDataDir(explicit: string | undefined, env: NodeJS.ProcessEnv): string {
  if (explicit !== undefined && explicit.length > 0) {
    return path.resolve(expandHome(explicit));
  }
  const dshHome = env.DSH_HOME !== undefined && env.DSH_HOME.length > 0 ? env.DSH_HOME : path.join(homedir(), '.dsh');
  return path.resolve(dshHome, 'nemo-relay');
}

/**
 * Normalize a Phoenix base URL: `http:` or `https:`, no credentials, query, or
 * fragment, and no trailing slash.
 */
export function normalizePhoenixUrl(value: string, fieldPath = 'phoenix.url'): string {
  let url: URL;
  try {
    url = new URL(value.trim());
  } catch {
    throw new Error(`${fieldPath} must be an absolute http or https URL`);
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error(`${fieldPath} must use http or https`);
  }
  if (url.username !== '' || url.password !== '' || url.search !== '' || url.hash !== '') {
    throw new Error(`${fieldPath} must not contain credentials, a query, or a fragment`);
  }
  const pathname = url.pathname.replace(/\/+$/, '');
  return `${url.origin}${pathname}`;
}

/** Whether a URL names this machine through a loopback address. */
export function isLoopbackUrl(value: string): boolean {
  try {
    const host = new URL(value).hostname.replace(/^\[|\]$/g, '').toLowerCase();
    return host === 'localhost' || host === '::1' || /^127(\.\d{1,3}){3}$/.test(host);
  } catch {
    return false;
  }
}

function parseGuardrails(value: unknown): GuardrailsConfig {
  const raw = asRecord(value, 'guardrails', true);
  rejectUnknownFields(raw, 'guardrails', ['enabled', 'failMode', 'rules']);
  return {
    enabled: optionalBoolean(raw.enabled, 'guardrails.enabled') ?? true,
    failMode: failModeReader(raw.failMode),
    rules: rulesReader(raw.rules),
    live: isReference(raw.rules) && isReference(raw.failMode),
  };
}

/** A volatile field as DeepSeek Harness passes it: a reference read with `get()`. */
function isReference(value: unknown): boolean {
  return typeof optionalRecord(value)?.get === 'function';
}

/** The fail mode is a plain string in a raw row and a volatile reference in DeepSeek Harness. */
function failModeReader(value: unknown): () => FailMode {
  const check = (mode: unknown): FailMode => {
    if (mode === undefined || mode === null) {
      return 'open';
    }
    if (mode !== 'open' && mode !== 'closed') {
      throw new Error('guardrails.failMode must be "open" or "closed"');
    }
    return mode;
  };
  const reference = optionalRecord(value);
  if (reference !== undefined && typeof reference.get === 'function') {
    const get = reference.get as () => unknown;
    check(get());
    return () => {
      try {
        return check(get());
      } catch {
        // A volatile update is validated by DeepSeek Harness first; never fail a tool call here.
        return 'open';
      }
    };
  }
  const mode = check(value);
  return () => mode;
}

/** Rules are a plain array in a raw row and a volatile reference in DeepSeek Harness. */
function rulesReader(value: unknown): () => readonly unknown[] | undefined {
  const check = (rules: unknown): readonly unknown[] | undefined => {
    if (rules === undefined || rules === null) {
      return undefined;
    }
    if (!Array.isArray(rules)) {
      throw new Error('guardrails.rules must be an array');
    }
    return rules;
  };
  const reference = optionalRecord(value);
  if (reference !== undefined && typeof reference.get === 'function') {
    const get = reference.get as () => unknown;
    check(get());
    return () => {
      try {
        return check(get());
      } catch {
        // A volatile update is validated by DeepSeek Harness first; never fail a tool call here.
        return undefined;
      }
    };
  }
  const rules = check(value);
  return () => rules;
}

function parsePhoenix(value: unknown): PhoenixConfig {
  const raw = asRecord(value, 'phoenix', true);
  rejectUnknownFields(raw, 'phoenix', ['enabled', 'url', 'project', 'autoConnect', 'launch']);
  const launch = asRecord(raw.launch, 'phoenix.launch', true);
  rejectUnknownFields(launch, 'phoenix.launch', ['enabled', 'version', 'python', 'command', 'startupTimeoutMs']);
  const url = normalizePhoenixUrl(optionalString(raw.url, 'phoenix.url') ?? DEFAULT_PHOENIX_URL);
  const project = optionalString(raw.project, 'phoenix.project')?.trim() ?? DEFAULT_PHOENIX_PROJECT;
  if (!PHOENIX_PROJECT.test(project)) {
    throw new Error('phoenix.project must be 1-80 letters, digits, spaces, ".", "_" or "-"');
  }
  const version = optionalString(launch.version, 'phoenix.launch.version')?.trim() ?? DEFAULT_PHOENIX_VERSION;
  if (!PHOENIX_VERSION.test(version)) {
    throw new Error('phoenix.launch.version must be an arize-phoenix release such as "21.0.0"');
  }
  const python = optionalString(launch.python, 'phoenix.launch.python')?.trim() ?? DEFAULT_PHOENIX_PYTHON;
  if (!PYTHON_VERSION.test(python)) {
    throw new Error('phoenix.launch.python must be a Python version such as "3.13"');
  }
  const startupTimeoutMs =
    optionalNumber(launch.startupTimeoutMs, 'phoenix.launch.startupTimeoutMs') ?? DEFAULT_PHOENIX_STARTUP_TIMEOUT_MS;
  if (
    !Number.isInteger(startupTimeoutMs) ||
    startupTimeoutMs < PHOENIX_STARTUP_TIMEOUT_RANGE.min ||
    startupTimeoutMs > PHOENIX_STARTUP_TIMEOUT_RANGE.max
  ) {
    throw new Error(
      `phoenix.launch.startupTimeoutMs must be an integer from ${PHOENIX_STARTUP_TIMEOUT_RANGE.min} to ${PHOENIX_STARTUP_TIMEOUT_RANGE.max}`,
    );
  }
  const command = parseCommand(launch.command);
  return {
    enabled: optionalBoolean(raw.enabled, 'phoenix.enabled') ?? true,
    url,
    project,
    autoConnect: optionalBoolean(raw.autoConnect, 'phoenix.autoConnect') ?? true,
    launch: {
      enabled: optionalBoolean(launch.enabled, 'phoenix.launch.enabled') ?? true,
      version,
      python,
      ...(command === undefined ? {} : { command }),
      startupTimeoutMs,
    },
  };
}

function parseCommand(value: unknown): string[] | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (!Array.isArray(value) || !value.every((part) => typeof part === 'string' && part.length > 0)) {
    throw new Error('phoenix.launch.command must be an array of non-empty strings');
  }
  return value.length === 0 ? undefined : [...(value as string[])];
}

/** Normalize the optional generic NeMo Relay plugin-host config embedded in the row config. */
function parsePluginHostConfig(value: unknown): NemoRelayPluginHostConfig {
  if (value === undefined) {
    return { version: 1, components: [] };
  }
  const record = asRecord(value, 'plugins', false);
  const version = optionalNumber(record.version, 'plugins.version') ?? 1;
  const components = record.components === undefined ? [] : record.components;
  if (!Array.isArray(components)) {
    throw new Error('plugins.components must be an array');
  }
  return {
    ...structuredClone(record),
    version,
    components: components.map((component) => structuredClone(component)),
  };
}

/**
 * Desktop hosts start with an arbitrary working directory, so relative output
 * directories resolve against the plugin data directory instead of `process.cwd()`.
 */
function resolveOutputDirectories(config: NemoRelayPluginHostConfig, dataDir: string): NemoRelayPluginHostConfig {
  for (const component of config.components) {
    const record = optionalRecord(component);
    if (record?.kind !== 'observability') {
      continue;
    }
    const componentConfig = optionalRecord(record.config);
    resolveDirectoryField(optionalRecord(componentConfig?.atif), dataDir);
    for (const sink of listOf(optionalRecord(componentConfig?.atof)?.sinks)) {
      resolveDirectoryField(optionalRecord(sink), dataDir);
    }
    for (const sink of listOf(optionalRecord(componentConfig?.opentelemetry)?.file_sinks)) {
      resolveDirectoryField(optionalRecord(sink), dataDir);
    }
  }
  return config;
}

function listOf(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function resolveDirectoryField(record: Record<string, unknown> | undefined, dataDir: string): void {
  const directory = record?.output_directory;
  if (record !== undefined && typeof directory === 'string' && directory.length > 0) {
    record.output_directory = path.resolve(dataDir, expandHome(directory));
  }
}

function expandHome(value: string): string {
  return value === '~' || value.startsWith('~/') || value.startsWith('~\\')
    ? path.join(homedir(), value.slice(1))
    : value;
}

/** Require an object config section, optionally treating undefined as an empty object. */
function asRecord(value: unknown, fieldPath: string, optional: boolean): Record<string, unknown> {
  if ((value === undefined || value === null) && optional) {
    return {};
  }
  const record = optionalRecord(value);
  if (record !== undefined) {
    return record;
  }
  throw new Error(`${fieldPath} must be an object`);
}

function optionalRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

/** Keep parser behavior aligned with a closed configuration contract. */
function rejectUnknownFields(raw: Record<string, unknown>, fieldPath: string, allowed: string[]): void {
  const allowedSet = new Set(allowed);
  for (const key of Object.keys(raw)) {
    if (!allowedSet.has(key)) {
      throw new Error(`${fieldPath}.${key} is not supported`);
    }
  }
}

function optionalBoolean(value: unknown, fieldPath: string): boolean | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== 'boolean') {
    throw new Error(`${fieldPath} must be a boolean`);
  }
  return value;
}

function optionalNumber(value: unknown, fieldPath: string): number | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`${fieldPath} must be a finite number`);
  }
  return value;
}

function optionalString(value: unknown, fieldPath: string): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== 'string') {
    throw new Error(`${fieldPath} must be a string`);
  }
  return value;
}
