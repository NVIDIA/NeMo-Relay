// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The plugin-host document this integration activates, and what it exports.
 *
 * Relay layers the document passed to `initialize` over the discovered
 * `plugins.toml` files: components merge by `kind`, code values win over file
 * values, and component-level destination lists concatenate. Defaults are
 * therefore added only where no layer already decides the same thing, so an
 * operator or user `plugins.toml` keeps its meaning:
 *
 * - Secret redaction adds Relay's `pii_redaction` component with credential
 *   detectors unless `config.plugins` configures `pii_redaction`, or a file
 *   layer uses its single-policy form (which cannot be combined with profiles).
 * - Per-turn ATIF trajectories go to `<dataDir>/atif/<session>/` unless
 *   `config.plugins` configures `observability`, or a file layer configures
 *   `atif` (a code value would override it).
 * - Neither default is added for a kind that a file layer turns off
 *   (`enabled = false`): the default's `enabled: true` would turn the layer's
 *   component, with everything it configures, back on.
 */
import path from 'node:path';

import type { NemoRelayDeepSeekHarnessConfig, NemoRelayPluginHostConfig } from './config.js';

/**
 * Relay detectors for credentials. `aws_secret_access_key` and
 * `azure_storage_account_key` are left out: their patterns match any 40- or
 * 88-character base64-like string, which includes every git commit SHA and npm
 * integrity hash a coding agent reads.
 */
export const SECRET_DETECTORS = ['api_key', 'bearer_token', 'jwt', 'aws_access_key_id', 'gcp_api_key'] as const;

export const ATIF_AGENT_NAME = 'deepseek-harness';
/** One directory per DeepSeek Harness session, one file per turn scope. */
export const ATIF_FILENAME_TEMPLATE = '{metadata.session_id:-unassigned}/trajectory-{session_id}.json';

export type DefaultDecision = { applied: boolean; reason?: string };

export type PluginDocumentPlan = {
  document: NemoRelayPluginHostConfig;
  redaction: DefaultDecision;
  atif: DefaultDecision & { directory?: string; modelName?: string };
};

/** What the host knows about its agent when the plugin host activates. */
export type AgentDefaults = {
  /** DeepSeek Harness's default model, the trajectory-level model of ATIF (steps carry their own). */
  modelName?: string | undefined;
};

export type ExporterKind = 'atif' | 'atof' | 'otel' | 'openinference';
export type ExporterDescriptor = { kind: ExporterKind; target: string };

/**
 * Add the integration's defaults to the user's document.
 *
 * @param config - Normalized plugin configuration.
 * @param resolvedUserDocument - `resolved_config` from validating `config.plugins`
 *   alone: the merged file layers plus the user's components.
 * @param agent - Agent metadata for the default ATIF section.
 * @returns The document to activate and why each default was or was not added.
 */
export function planPluginDocument(
  config: NemoRelayDeepSeekHarnessConfig,
  resolvedUserDocument: unknown,
  agent: AgentDefaults = {},
): PluginDocumentPlan {
  const user = config.plugins;
  const resolved = componentsOf(resolvedUserDocument);
  const added: unknown[] = [];

  let redaction: DefaultDecision;
  if (!config.redaction.secrets) {
    redaction = { applied: false, reason: 'redaction.secrets is off' };
  } else if (hasKind(user.components, 'pii_redaction')) {
    redaction = { applied: false, reason: 'config.plugins configures pii_redaction' };
  } else if (isTurnedOff(resolved, 'pii_redaction')) {
    redaction = { applied: false, reason: 'a plugins.toml layer turns pii_redaction off' };
  } else if (resolved.some((component) => isSinglePolicyRedaction(component))) {
    redaction = {
      applied: false,
      reason: 'a plugins.toml layer configures single-policy pii_redaction, which cannot be combined with profiles',
    };
  } else {
    added.push(secretRedactionComponent());
    redaction = { applied: true };
  }

  let atif: PluginDocumentPlan['atif'];
  const directory = path.join(config.dataDir, 'atif');
  if (hasKind(user.components, 'observability')) {
    atif = { applied: false, reason: 'config.plugins configures observability' };
  } else if (isTurnedOff(resolved, 'observability')) {
    atif = { applied: false, reason: 'a plugins.toml layer turns observability off' };
  } else if (
    resolved.some((component) => component.kind === 'observability' && asRecord(component.config)?.atif !== undefined)
  ) {
    atif = { applied: false, reason: 'a plugins.toml layer configures ATIF' };
  } else {
    added.push(atifComponent(directory, agent.modelName));
    atif = { applied: true, directory, ...(agent.modelName === undefined ? {} : { modelName: agent.modelName }) };
  }

  return {
    document: added.length === 0 ? user : { ...user, components: [...user.components, ...added] },
    redaction,
    atif,
  };
}

/** Relay's `pii_redaction` component that redacts credentials from every published payload. */
export function secretRedactionComponent(): Record<string, unknown> {
  return {
    kind: 'pii_redaction',
    enabled: true,
    config: {
      version: 1,
      profiles: SECRET_DETECTORS.map((detector) => ({ mode: 'builtin', builtin: { action: 'redact', detector } })),
    },
  };
}

/**
 * Relay's `observability` component writing one ATIF trajectory per DeepSeek
 * Harness turn. The trajectory's agent is `deepseek-harness` rather than
 * Relay's default `NeMo Relay`; its model is DeepSeek Harness's default model
 * when known (Relay writes `unknown` otherwise), and every step keeps the model
 * of its own call.
 */
export function atifComponent(directory: string, modelName?: string): Record<string, unknown> {
  return {
    kind: 'observability',
    enabled: true,
    config: {
      version: 4,
      atif: {
        enabled: true,
        agent_name: ATIF_AGENT_NAME,
        ...(modelName === undefined ? {} : { model_name: modelName }),
        output_directory: directory,
        filename_template: ATIF_FILENAME_TEMPLATE,
        // Raw events, including per-chunk receipts, are already in the session's ATOF trace.
        include_observed_events: false,
        // Relay otherwise copies the whole provider request, tool list included, into every step.
        include_llm_requests: false,
      },
    },
  };
}

/**
 * Directory holding one session's trajectories under the default ATIF layout.
 * Mirrors Relay's metadata placeholder sanitization (`sanitize_atif_metadata_fragment`).
 */
export function atifSessionDirectory(atifDirectory: string, sessionId: string): string {
  return path.join(atifDirectory, sanitizeAtifFragment(sessionId));
}

/** Relay's filename-fragment sanitization: unsafe runs become one `-`; `..` is never kept. */
export function sanitizeAtifFragment(value: string): string {
  const characters = [...value];
  let sanitized = '';
  let replacing = false;
  for (let index = 0; index < characters.length; index += 1) {
    const character = characters[index] ?? '';
    let safe: boolean;
    if (character === '.') {
      let count = 1;
      while (characters[index + 1] === '.') {
        count += 1;
        index += 1;
      }
      safe = count === 1 && !(sanitized.length === 0 && index === characters.length - 1);
    } else {
      safe = /^[A-Za-z0-9_~-]$/.test(character);
    }
    if (safe) {
      sanitized += character;
      replacing = false;
    } else if (!replacing) {
      sanitized += '-';
      replacing = true;
    }
  }
  return sanitized;
}

/** Every Relay exporter an effective (resolved) document activates, with its destination. */
export function describeExporters(document: unknown): ExporterDescriptor[] {
  const exporters: ExporterDescriptor[] = [];
  for (const component of componentsOf(document)) {
    if (component.kind !== 'observability' || component.enabled === false) {
      continue;
    }
    const config = asRecord(component.config);
    const atof = asRecord(config?.atof);
    if (atof?.enabled === true) {
      for (const sink of listOf(atof.sinks)) {
        const record = asRecord(sink);
        const target = stringField(record, record?.type === 'stream' ? 'url' : 'output_directory');
        if (target !== undefined) {
          exporters.push({ kind: 'atof', target });
        }
      }
    }
    const atif = asRecord(config?.atif);
    if (atif?.enabled === true) {
      const storage = listOf(atif.storage).map(asRecord);
      if (storage.length === 0) {
        exporters.push({ kind: 'atif', target: stringField(atif, 'output_directory') ?? '.' });
      }
      for (const destination of storage) {
        const target =
          destination?.type === 's3'
            ? `s3://${stringField(destination, 'bucket') ?? '?'}/${stringField(destination, 'key_prefix') ?? ''}`
            : stringField(destination, 'endpoint');
        if (target !== undefined) {
          exporters.push({ kind: 'atif', target });
        }
      }
    }
    const opentelemetry = asRecord(config?.opentelemetry);
    if (opentelemetry?.enabled === true) {
      for (const destination of [...listOf(opentelemetry.endpoints), ...listOf(opentelemetry.file_sinks)]) {
        const record = asRecord(destination);
        const target = stringField(record, 'endpoint') ?? stringField(record, 'output_directory');
        if (target !== undefined) {
          exporters.push({ kind: record?.type === 'openinference' ? 'openinference' : 'otel', target });
        }
      }
    }
  }
  return exporters;
}

/** Detector names of every enabled built-in redaction policy in a resolved document. */
export function describeRedaction(document: unknown): { active: boolean; detectors: string[] } {
  const detectors = new Set<string>();
  let active = false;
  for (const component of componentsOf(document)) {
    if (component.kind !== 'pii_redaction' || component.enabled === false) {
      continue;
    }
    active = true;
    const config = asRecord(component.config);
    const policies = Array.isArray(config?.profiles) ? config.profiles.map(asRecord) : [config];
    for (const policy of policies) {
      const detector = asRecord(policy?.builtin)?.detector;
      if (policy?.enabled !== false && typeof detector === 'string') {
        detectors.add(detector);
      }
    }
  }
  return { active, detectors: [...detectors] };
}

function isSinglePolicyRedaction(component: Record<string, unknown>): boolean {
  if (component.kind !== 'pii_redaction' || component.enabled === false) {
    return false;
  }
  const config = asRecord(component.config) ?? {};
  return !Array.isArray(config.profiles) && ['mode', 'builtin', 'local'].some((key) => config[key] !== undefined);
}

/** True when the resolved layers hold a component of `kind` that is switched off. */
function isTurnedOff(resolved: Array<Record<string, unknown>>, kind: string): boolean {
  return resolved.some((component) => component.kind === kind && component.enabled === false);
}

function hasKind(components: unknown[], kind: string): boolean {
  return components.some((component) => asRecord(component)?.kind === kind);
}

function componentsOf(document: unknown): Array<Record<string, unknown>> {
  return listOf(asRecord(document)?.components)
    .map(asRecord)
    .filter((component): component is Record<string, unknown> => component !== undefined);
}

function stringField(record: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = record?.[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function listOf(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}
