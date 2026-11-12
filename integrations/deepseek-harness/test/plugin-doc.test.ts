// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The effective plugin-host document: which defaults apply under Relay's layer
 * merge, and what the status reports about exporters and redaction.
 */
import assert from 'node:assert/strict';
import path from 'node:path';
import { describe, it } from 'node:test';

import { parseConfig } from '../src/config.js';
import {
  ATIF_FILENAME_TEMPLATE,
  atifSessionDirectory,
  describeExporters,
  describeRedaction,
  planPluginDocument,
  sanitizeAtifFragment,
  SECRET_DETECTORS,
} from '../src/plugin-doc.js';

const DATA_DIR = path.resolve('/data/relay');

function plan(row: Record<string, unknown> = {}, fileLayers: unknown[] = []) {
  const config = parseConfig({ dataDir: DATA_DIR, ...row });
  // `validate(config.plugins).resolved_config`: the user's components merged over the file layers.
  return planPluginDocument(config, { version: 1, components: [...fileLayers, ...config.plugins.components] });
}

function kinds(document: { components: unknown[] }): string[] {
  return document.components.map((component) => (component as { kind: string }).kind);
}

describe('effective plugin document', () => {
  it('adds secret redaction and per-session ATIF when no layer configures them', () => {
    const { document, redaction, atif } = plan();
    assert.deepEqual(kinds(document), ['pii_redaction', 'observability']);
    assert.deepEqual(redaction, { applied: true });
    assert.deepEqual(atif, { applied: true, directory: path.join(DATA_DIR, 'atif') });

    const [pii, observability] = document.components as Array<{ config: Record<string, any> }>;
    assert.deepEqual(
      pii?.config.profiles.map((profile: any) => [profile.mode, profile.builtin.action, profile.builtin.detector]),
      SECRET_DETECTORS.map((detector) => ['builtin', 'redact', detector]),
    );
    assert.equal(observability?.config.version, 4);
    assert.deepEqual(observability?.config.atif, {
      enabled: true,
      agent_name: 'deepseek-harness',
      output_directory: path.join(DATA_DIR, 'atif'),
      filename_template: ATIF_FILENAME_TEMPLATE,
      include_observed_events: false,
      include_llm_requests: false,
    });
  });

  it("describes DeepSeek Harness as the ATIF agent, with the host's default model when it is known", () => {
    const config = parseConfig({ dataDir: DATA_DIR });
    const resolved = { version: 1, components: [] };
    const atifOf = (plan: ReturnType<typeof planPluginDocument>) =>
      (plan.document.components[1] as { config: { atif: Record<string, unknown> } }).config.atif;
    const known = planPluginDocument(config, resolved, { modelName: 'deepseek-chat' });
    assert.deepEqual([atifOf(known).agent_name, atifOf(known).model_name], ['deepseek-harness', 'deepseek-chat']);
    assert.equal(known.atif.modelName, 'deepseek-chat');
    // Unknown: Relay writes its own default rather than a guess.
    assert.equal('model_name' in atifOf(planPluginDocument(config, resolved)), false);
  });

  it('never redacts git SHAs or integrity hashes by default', () => {
    assert.ok(!(SECRET_DETECTORS as readonly string[]).includes('aws_secret_access_key'));
    assert.ok(!(SECRET_DETECTORS as readonly string[]).includes('azure_storage_account_key'));
  });

  it('leaves kinds the user configures to the user, so no layer holds the same kind twice', () => {
    const userObservability = { kind: 'observability', config: { version: 4, atof: { enabled: true } } };
    const userRedaction = { kind: 'pii_redaction', enabled: false, config: { version: 1 } };
    const result = plan({ plugins: { components: [userObservability, userRedaction] } });
    assert.deepEqual(kinds(result.document), ['observability', 'pii_redaction']);
    assert.match(result.redaction.reason ?? '', /config\.plugins configures pii_redaction/);
    assert.match(result.atif.reason ?? '', /config\.plugins configures observability/);
  });

  it('keeps a plugins.toml layer in charge of what it decides', () => {
    // An operator layer that exports OpenTelemetry and redacts with profiles: both merge with the defaults.
    const operator = [
      { kind: 'observability', config: { version: 4, opentelemetry: { enabled: true, endpoints: [] } } },
      {
        kind: 'pii_redaction',
        config: { version: 1, profiles: [{ builtin: { action: 'redact', detector: 'email' } }] },
      },
    ];
    assert.deepEqual(kinds(plan({}, operator).document), ['pii_redaction', 'observability']);

    // A layer that decides ATIF (even disabling it), or redacts with the single-policy form, keeps its decision.
    const deciding = [
      { kind: 'observability', config: { version: 4, atif: { enabled: false } } },
      {
        kind: 'pii_redaction',
        config: { version: 1, mode: 'builtin', builtin: { action: 'redact', detector: 'email' } },
      },
    ];
    const result = plan({}, deciding);
    assert.deepEqual(kinds(result.document), []);
    assert.match(result.atif.reason ?? '', /plugins\.toml layer configures ATIF/);
    assert.match(result.redaction.reason ?? '', /single-policy/);
  });

  it('never turns a component back on that a plugins.toml layer switched off', () => {
    // Relay merges components by kind and the code value wins, so a default with `enabled: true`
    // would start the operator's disabled exporter and redaction again.
    const switchedOff = [
      {
        kind: 'observability',
        enabled: false,
        config: {
          version: 4,
          opentelemetry: { enabled: true, endpoints: [{ endpoint: 'https://collector.example/v1/traces' }] },
        },
      },
      {
        kind: 'pii_redaction',
        enabled: false,
        config: { version: 1, profiles: [{ mode: 'builtin', builtin: { action: 'redact', detector: 'email' } }] },
      },
    ];
    const result = plan({}, switchedOff);
    assert.deepEqual(kinds(result.document), []);
    assert.deepEqual(result.atif, { applied: false, reason: 'a plugins.toml layer turns observability off' });
    assert.deepEqual(result.redaction, { applied: false, reason: 'a plugins.toml layer turns pii_redaction off' });
    // One kind switched off leaves the other default in place.
    assert.deepEqual(kinds(plan({}, switchedOff.slice(0, 1)).document), ['pii_redaction']);
    assert.deepEqual(kinds(plan({}, switchedOff.slice(1)).document), ['observability']);
  });

  it('honors redaction.secrets = false', () => {
    const result = plan({ redaction: { secrets: false } });
    assert.deepEqual(kinds(result.document), ['observability']);
    assert.match(result.redaction.reason ?? '', /redaction\.secrets is off/);
  });
});

describe('ATIF session directory', () => {
  it('sanitizes session ids exactly like Relay renders {metadata.session_id}', () => {
    // Vectors from Relay's `atif_metadata_template_values_are_sanitized` test.
    const vectors: Array<[string, string]> = [
      ['tenant-a', 'tenant-a'],
      ['../tenant/a', '-tenant-a'],
      ['tenant\\project', 'tenant-project'],
      ['tenant / : project', 'tenant-project'],
      ['run...name', 'run-name'],
      ['ténant///项目', 't-nant-'],
      ['.', '-'],
      ['..', '-'],
      // Observed with the real binding: Relay wrote this session's trajectory under `atif/sess-b-c/`.
      ['sess/äb..c', 'sess-b-c'],
    ];
    for (const [input, expected] of vectors) {
      assert.equal(sanitizeAtifFragment(input), expected, input);
    }
    assert.equal(atifSessionDirectory('/atif', 'session-1a2b.c'), path.join('/atif', 'session-1a2b.c'));
  });
});

describe('status descriptions', () => {
  it('lists every exporter of a resolved document with its destination', () => {
    const resolved = {
      components: [
        {
          kind: 'observability',
          config: {
            atof: {
              enabled: true,
              sinks: [
                { type: 'file', output_directory: '/logs' },
                { type: 'stream', url: 'http://x/e' },
              ],
            },
            atif: { enabled: true, output_directory: '/atif' },
            opentelemetry: {
              enabled: true,
              endpoints: [
                { type: 'gen_ai', endpoint: 'https://collector.example/v1/traces' },
                { type: 'openinference', endpoint: 'http://127.0.0.1:16006/v1/traces' },
              ],
              file_sinks: [{ type: 'full', output_directory: '/otel' }],
            },
          },
        },
        { kind: 'observability', enabled: false, config: { atif: { enabled: true, output_directory: '/ignored' } } },
      ],
    };
    assert.deepEqual(describeExporters(resolved), [
      { kind: 'atof', target: '/logs' },
      { kind: 'atof', target: 'http://x/e' },
      { kind: 'atif', target: '/atif' },
      { kind: 'otel', target: 'https://collector.example/v1/traces' },
      { kind: 'openinference', target: 'http://127.0.0.1:16006/v1/traces' },
      { kind: 'otel', target: '/otel' },
    ]);
    // Remote ATIF storage replaces the local directory.
    const remote = {
      components: [
        {
          kind: 'observability',
          config: { atif: { enabled: true, storage: [{ type: 's3', bucket: 'b', key_prefix: 'p/' }] } },
        },
      ],
    };
    assert.deepEqual(describeExporters(remote), [{ kind: 'atif', target: 's3://b/p/' }]);
  });

  it('reports the detectors of every enabled redaction policy', () => {
    const resolved = {
      components: [
        {
          kind: 'pii_redaction',
          config: {
            profiles: [
              { builtin: { action: 'redact', detector: 'api_key' } },
              { enabled: false, builtin: { action: 'redact', detector: 'email' } },
              { builtin: { action: 'redact', pattern: 'x' } },
            ],
          },
        },
      ],
    };
    assert.deepEqual(describeRedaction(resolved), { active: true, detectors: ['api_key'] });
    assert.deepEqual(describeRedaction({ components: [] }), { active: false, detectors: [] });
  });
});
