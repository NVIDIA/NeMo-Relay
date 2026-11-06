// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The status snapshot (contract v4) behind each process's status file
 * (`<dataDir>/status/<pid>-<4 hex>.json`, see `status-file.ts`), `/relay json`,
 * and the human-readable `/relay` reply.
 *
 * The snapshot is a superset of the runtime health payload: it adds the Relay
 * and Node versions, the exporters of the activated plugin-host document, the
 * capture and redaction settings, the guardrail rules, and Phoenix.
 */
import type { CaptureConfig, FailMode, NemoRelayDeepSeekHarnessConfig } from './config.js';
import type { NemoRelayHealthSnapshot } from './health.js';
import type { PluginDocumentPlan } from './plugin-doc.js';
import { describeExporters, describeRedaction, type ExporterDescriptor } from './plugin-doc.js';
import type { GuardrailsSnapshot, RuleError } from './policy.js';

export const STATUS_VERSION = 4;
/** How many recent `/relay` action results the status snapshot keeps. */
export const ACTION_LOG_SIZE = 16;

/**
 * The structured result of one `/relay` action (a rules save, reset, test, or
 * fail-mode change, or a Phoenix action). The command replies in plain words;
 * the NeMo Relay view finds its own result here by the request id (`rid`) it
 * sent, because the status file is rewritten before the command replies.
 */
export type RelayActionRecord = {
  /** The request id the caller sent, when it sent one. */
  rid?: string;
  /** `rules.save`, `rules.reset`, `rules.test`, `rules.failmode`, or `phoenix.<start|stop|connect|disconnect>`. */
  action: string;
  ok: boolean;
  /** When the action finished (ISO 8601). */
  at: string;
  /** Why it failed: `invalid`, `stale`, `unavailable`, or `error`. */
  code?: string;
  error?: string;
  /** Rule problems that refused a save, by rule index and field. */
  errors?: RuleError[];
  /** The rules revision after a save or reset. */
  revision?: string;
  verdict?: 'allow' | 'block';
  ruleId?: string;
  reason?: string;
  timedOut?: string[];
  failMode?: FailMode;
  phoenix?: PhoenixSnapshot;
};

/** The last {@link ACTION_LOG_SIZE} action results, oldest first. */
export class ActionLog {
  private readonly entries: RelayActionRecord[] = [];

  record(entry: RelayActionRecord): void {
    this.entries.push(entry);
    if (this.entries.length > ACTION_LOG_SIZE) {
      this.entries.splice(0, this.entries.length - ACTION_LOG_SIZE);
    }
  }

  list(): RelayActionRecord[] {
    return this.entries.map((entry) => ({ ...entry }));
  }
}

export type PhoenixState = 'disabled' | 'stopped' | 'starting' | 'waiting' | 'running' | 'error';

export type PhoenixSnapshot = {
  enabled: boolean;
  state: PhoenixState;
  /** Whether Relay currently exports to this Phoenix. */
  connected: boolean;
  mode?: 'owned' | 'external';
  url: string;
  version?: string;
  project: string;
  error?: string;
  /** Session deep link with a `{sessionId}` placeholder for the URL-encoded session id. */
  sessionUrl?: string;
  /** Progress of a launch while `state` is `starting`. */
  phase?: 'checking' | 'launching' | 'downloading' | 'serving';
  /** Why Phoenix cannot be launched here, when it cannot. */
  launchHint?: string;
  /** Phoenix's own log, when this plugin started it. */
  log?: string;
};

/** Credential detectors; the status says secrets are redacted only when one of these is active. */
const SECRET_CLASS_DETECTORS = new Set([
  'api_key',
  'bearer_token',
  'jwt',
  'aws_access_key_id',
  'aws_secret_access_key',
  'gcp_api_key',
  'azure_storage_account_key',
]);

export type RedactionSnapshot = {
  /** True when an active Relay `pii_redaction` policy scrubs credentials. */
  secrets: boolean;
  /** Detectors of every active redaction policy. */
  detectors: string[];
  /** Who configures redaction: this integration's default, the plugin-host layers, or nobody. */
  source: 'default' | 'plugins' | 'none';
  /** Why the default secret redaction was not added, when it was not. */
  note?: string;
};

export type StatusSnapshot = NemoRelayHealthSnapshot & {
  v: typeof STATUS_VERSION;
  updatedAt: string;
  relay: { version?: string; node: string };
  /** The DeepSeek Harness process this snapshot describes. */
  pid: number;
  /** This process's status file (`<dataDir>/status/<pid>-<4 hex>.json`), when it writes one. */
  statusFile?: string;
  dataDir: string;
  exporters: ExporterDescriptor[];
  capture: CaptureConfig;
  redaction: RedactionSnapshot;
  guardrails: GuardrailsSnapshot;
  phoenix: PhoenixSnapshot;
  view: { enabled: boolean; directory: string };
  /** Root of the default per-session ATIF layout, when this integration added it. */
  atif?: { directory: string };
  /** Recent `/relay` action results, oldest first (see {@link RelayActionRecord}). */
  actions: RelayActionRecord[];
  /**
   * How the Loader row reached the plugin: `schema` when this plugin exported its
   * `Config` schema (schemastery loaded), `live` when DeepSeek Harness validated
   * the row with it, so rule and fail-mode saves apply without a plugin restart.
   */
  config: { schema: boolean; live: boolean };
};

export type StatusInputs = {
  health: NemoRelayHealthSnapshot;
  config: NemoRelayDeepSeekHarnessConfig;
  plan: PluginDocumentPlan | undefined;
  /** The resolved (merged) plugin-host document Relay reported. */
  resolved: unknown;
  relayVersion: string | undefined;
  guardrails: GuardrailsSnapshot;
  phoenix: PhoenixSnapshot;
  viewDirectory: string;
  /** This process's status file. */
  statusFile?: string;
  /** Recent `/relay` action results. */
  actions?: RelayActionRecord[];
  /** Whether this plugin exported a `Config` schema. */
  configSchema?: boolean;
  now?: Date;
};

/** Build the complete v4 status snapshot. */
export function buildStatusSnapshot(inputs: StatusInputs): StatusSnapshot {
  const { health, config, plan } = inputs;
  const resolved = inputs.resolved ?? plan?.document ?? config.plugins;
  const redaction = describeRedaction(resolved);
  const secrets = redaction.detectors.some((detector) => SECRET_CLASS_DETECTORS.has(detector));
  const relay: StatusSnapshot['relay'] = { node: process.version };
  if (inputs.relayVersion !== undefined) {
    relay.version = inputs.relayVersion;
  }
  const exporters = health.initializedPluginHost ? describeExporters(resolved) : [];
  const outputs = { ...health.outputs };
  // Phoenix is exported through a direct Relay subscriber, outside the plugin-host document.
  if (inputs.phoenix.connected) {
    exporters.push({ kind: 'openinference', target: `${inputs.phoenix.url}/v1/traces` });
    outputs.openInference = 'enabled';
  }
  return {
    v: STATUS_VERSION,
    ...health,
    outputs,
    updatedAt: (inputs.now ?? new Date()).toISOString(),
    relay,
    pid: process.pid,
    ...(inputs.statusFile === undefined ? {} : { statusFile: inputs.statusFile }),
    dataDir: config.dataDir,
    exporters,
    capture: { ...config.capture },
    redaction: {
      secrets: health.initializedPluginHost && secrets,
      detectors: redaction.detectors,
      source: !redaction.active ? 'none' : plan?.redaction.applied === true ? 'default' : 'plugins',
      ...(plan?.redaction.applied === false && plan.redaction.reason !== undefined
        ? { note: plan.redaction.reason }
        : {}),
    },
    guardrails: inputs.guardrails,
    phoenix: inputs.phoenix,
    view: { enabled: config.view.enabled, directory: inputs.viewDirectory },
    ...(plan?.atif.applied === true && plan.atif.directory !== undefined
      ? { atif: { directory: plan.atif.directory } }
      : {}),
    actions: inputs.actions ?? [],
    config: { schema: inputs.configSchema === true, live: config.guardrails.live },
  };
}

/** Render the snapshot as the plain-text `/relay` reply. */
export function formatStatus(snapshot: StatusSnapshot, extra: string[] = []): string {
  const counters = snapshot.counters;
  const lines = [
    `NeMo Relay: ${snapshot.state}${snapshot.lastError === undefined ? '' : ` (${snapshot.lastError})`}${
      snapshot.relay.version === undefined
        ? ''
        : ` — nemo-relay-node ${snapshot.relay.version}, Node ${snapshot.relay.node}`
    }`,
    `Plugin host: ${snapshot.initializedPluginHost ? 'initialized' : 'not initialized'}${
      snapshot.configPaths.length > 0 ? `; plugins.toml layers: ${snapshot.configPaths.join(', ')}` : ''
    }`,
    `Outputs: ATOF ${snapshot.outputs.atof}, ATIF ${snapshot.outputs.atif}, OpenTelemetry ${snapshot.outputs.otel}, OpenInference ${snapshot.outputs.openInference}`,
    snapshot.exporters.length === 0
      ? 'Exporters: none'
      : `Exporters: ${snapshot.exporters.map((exporter) => `${EXPORTER_LABELS[exporter.kind]} → ${exporter.target}`).join('; ')}`,
    formatRedaction(snapshot.redaction),
    formatRulesSummary(snapshot.guardrails),
    formatPhoenix(snapshot.phoenix),
    `Sessions: ${snapshot.activeSessions} active; turns: ${counters.turnsOpened} opened, ${counters.turnsClosed} closed`,
    `Managed calls: ${counters.llmCalls} LLM, ${counters.toolCalls} tool; ${counters.guardrailRejections} guardrail rejections; ${counters.middlewareErrors} middleware errors`,
    ...extra,
  ];
  return lines.join('\n');
}

const EXPORTER_LABELS: Record<ExporterDescriptor['kind'], string> = {
  atof: 'ATOF',
  atif: 'ATIF',
  otel: 'OpenTelemetry',
  openinference: 'OpenInference',
};

function formatRedaction(redaction: RedactionSnapshot): string {
  if (redaction.source === 'none') {
    return `Redaction: off${redaction.note === undefined ? '' : ` (${redaction.note})`}`;
  }
  const by =
    redaction.source === 'default'
      ? "Relay's pii_redaction (default secret detectors)"
      : "Relay's pii_redaction (plugin configuration)";
  const detectors = redaction.detectors.length > 0 ? `: ${redaction.detectors.join(', ')}` : '';
  return `Redaction: ${redaction.secrets ? 'secrets scrubbed' : 'on, no credential detectors'} by ${by}${detectors}`;
}

/** One line describing the active rules. */
export function formatRulesSummary(guardrails: GuardrailsSnapshot): string {
  if (!guardrails.enabled) {
    return 'Rules: off (guardrails.enabled is false)';
  }
  const on = guardrails.rules.filter((rule) => rule.enabled).length;
  const origin = {
    default: 'shipped defaults',
    config: 'saved in the plugin configuration',
    file: `saved in ${guardrails.location}`,
  }[guardrails.source];
  const problems = guardrails.errors === undefined ? '' : `; ${guardrails.errors.length} problem(s) skip rules`;
  return `Rules: ${on} of ${guardrails.rules.length} on, fail-${guardrails.failMode}, ${origin} (revision ${guardrails.revision}); ${guardrails.rejections} call(s) blocked${problems}`;
}

/** The plain-text rules list for `/relay rules`. */
export function formatRules(guardrails: GuardrailsSnapshot): string {
  const lines = [formatRulesSummary(guardrails)];
  guardrails.rules.forEach((rule, index) => {
    const tool = rule.tool.length === 0 ? 'any tool' : `tool ${rule.tool}`;
    const match =
      rule.match.length === 0
        ? 'any arguments'
        : `match ${rule.match}${rule.flags.length > 0 ? ` (flags ${rule.flags})` : ''}`;
    lines.push(`${index + 1}. ${rule.id} [${rule.enabled ? 'on' : 'off'}] ${tool}, ${match}`, `   ${rule.reason}`);
  });
  for (const error of guardrails.errors ?? []) {
    lines.push(`Problem: rule ${error.index + 1}${error.id === undefined ? '' : ` (${error.id})`}: ${error.message}`);
  }
  if (guardrails.loadError !== undefined) {
    lines.push(guardrails.loadError);
  }
  lines.push(
    ...(guardrails.editable
      ? [
          `Edit them in the NeMo Relay view, or send /relay rules save {"base":"${guardrails.revision}","rules":[...]}.`,
          `Calls the rules cannot check are ${guardrails.failMode === 'closed' ? 'blocked' : 'allowed'}; change it with /relay rules failmode ${guardrails.failMode === 'closed' ? 'open' : 'closed'}.`,
        ]
      : ['Rules cannot be edited from this session.']),
  );
  return lines.join('\n');
}

/** One line describing Phoenix. */
export function formatPhoenix(phoenix: PhoenixSnapshot): string {
  if (!phoenix.enabled || phoenix.state === 'disabled') {
    return 'Phoenix: off (phoenix.enabled is false)';
  }
  switch (phoenix.state) {
    case 'running': {
      const origin = phoenix.mode === 'external' ? ', started outside DeepSeek Harness' : ', started by NeMo Relay';
      const link = phoenix.connected ? `; project "${phoenix.project}" receives this host's turns` : '; not connected';
      return `Phoenix${phoenix.version === undefined ? '' : ` ${phoenix.version}`}: ${phoenix.url}${origin}${link}`;
    }
    case 'starting':
      return `Phoenix: starting at ${phoenix.url}…`;
    case 'waiting':
      return `Phoenix: waiting for ${phoenix.url}; reconnects as soon as it answers`;
    case 'error':
      return `Phoenix: not running (${phoenix.error ?? 'error'}); retry with /relay phoenix start`;
    default:
      return `Phoenix: not connected; start it with /relay phoenix start${phoenix.launchHint === undefined ? '' : ` (${phoenix.launchHint})`}`;
  }
}
