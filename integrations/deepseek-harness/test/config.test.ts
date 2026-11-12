// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Loader row configuration: defaults, validation, and agreement between the
 * exported schemastery `Config` and `parseConfig`.
 */
import assert from 'node:assert/strict';
import path from 'node:path';
import { describe, it } from 'node:test';

import { Config } from '../index.js';
import { isLoopbackUrl, normalizePhoenixUrl, parseConfig } from '../src/config.js';

type SchemaLike = ((value: unknown) => any) & { '~standard': { validate(value: unknown): { issues?: unknown[] } } };

const VOLATILE_WRITE = Symbol.for('cosmokit.volatile.write');

describe('configuration', () => {
  it('applies one-click defaults under DSH_HOME', () => {
    const config = parseConfig(undefined, { DSH_HOME: '/srv/dsh' });
    assert.equal(config.enabled, true);
    assert.equal(config.dataDir, path.resolve('/srv/dsh/nemo-relay'));
    assert.deepEqual(config.plugins, { version: 1, components: [] });
    // Full fidelity locally; credentials are scrubbed by Relay's pii_redaction instead.
    assert.deepEqual(config.capture, {
      includePrompts: true,
      includeResponses: true,
      stripToolArgs: false,
      stripToolResults: false,
    });
    assert.deepEqual(config.redaction, { secrets: true });
    assert.equal(config.guardrails.enabled, true);
    assert.equal(config.guardrails.failMode(), 'open');
    assert.equal(config.guardrails.rules(), undefined, 'no configured rules means the shipped defaults apply');
    assert.deepEqual(config.phoenix, {
      enabled: true,
      url: 'http://127.0.0.1:6006',
      project: 'deepseek-harness',
      autoConnect: true,
      launch: { enabled: true, version: '21.0.0', python: '3.13', startupTimeoutMs: 240_000 },
    });
    assert.deepEqual(config.agent, { enabled: true }, 'the agent can set up NeMo Relay for the user');
    assert.deepEqual(parseConfig({ agent: { enabled: false } }).agent, { enabled: false });
    assert.deepEqual(config.retention, { days: 30 }, 'per-conversation files do not accumulate forever');
    assert.deepEqual(parseConfig({ retention: { days: 0 } }).retention, { days: 0 }, '0 keeps every file');
  });

  it('resolves relative observability output directories against the data directory', () => {
    const config = parseConfig({
      dataDir: '/data/relay',
      plugins: {
        components: [
          {
            kind: 'observability',
            config: {
              atif: { enabled: true, output_directory: 'atif' },
              atof: { enabled: true, sinks: [{ type: 'file', output_directory: '/abs/atof' }] },
              opentelemetry: { enabled: true, file_sinks: [{ type: 'full', output_directory: 'otel' }] },
            },
          },
        ],
      },
    });
    const observability = config.plugins.components[0] as { config: Record<string, any> };
    assert.equal(observability.config.atif.output_directory, path.resolve('/data/relay/atif'));
    assert.equal(observability.config.atof.sinks[0].output_directory, path.resolve('/abs/atof'));
    assert.equal(observability.config.opentelemetry.file_sinks[0].output_directory, path.resolve('/data/relay/otel'));
  });

  it('rejects unknown fields and wrong types at every level', () => {
    assert.throws(() => parseConfig({ backend: 'hooks' }), /config\.backend is not supported/);
    assert.throws(
      () => parseConfig({ capture: { includePrompts: 'yes' } }),
      /capture\.includePrompts must be a boolean/,
    );
    assert.throws(() => parseConfig({ plugins: { components: {} } }), /plugins\.components must be an array/);
    assert.throws(() => parseConfig({ guardrails: { failMode: 'sometimes' } }), /guardrails\.failMode/);
    assert.throws(() => parseConfig({ guardrails: { rules: { id: 'x' } } }), /guardrails\.rules must be an array/);
    assert.throws(() => parseConfig({ guardrails: { tools: [] } }), /guardrails\.tools is not supported/);
    assert.throws(() => parseConfig({ redaction: { pii: true } }), /redaction\.pii is not supported/);
    assert.throws(() => parseConfig({ phoenix: { port: 6006 } }), /phoenix\.port is not supported/);
    assert.throws(() => parseConfig({ phoenix: { project: '../../etc' } }), /phoenix\.project/);
    assert.throws(() => parseConfig({ phoenix: { launch: { version: '21.0.0; rm -rf' } } }), /launch\.version/);
    assert.throws(() => parseConfig({ phoenix: { launch: { python: 'latest' } } }), /launch\.python/);
    assert.throws(() => parseConfig({ phoenix: { launch: { startupTimeoutMs: 5 } } }), /startupTimeoutMs/);
    assert.throws(() => parseConfig({ phoenix: { launch: { command: ['phoenix', ''] } } }), /launch\.command/);
    assert.throws(() => parseConfig({ agent: { tools: [] } }), /agent\.tools is not supported/);
    assert.throws(() => parseConfig({ agent: { enabled: 'yes' } }), /agent\.enabled must be a boolean/);
    assert.throws(() => parseConfig({ retention: { hours: 1 } }), /retention\.hours is not supported/);
    for (const days of [-1, 1.5, 4000, '7']) {
      assert.throws(() => parseConfig({ retention: { days } }), /retention\.days/);
    }
  });

  it('normalizes Phoenix URLs and refuses ones that could leak credentials', () => {
    assert.equal(normalizePhoenixUrl('http://127.0.0.1:16006/'), 'http://127.0.0.1:16006');
    assert.equal(normalizePhoenixUrl('https://phoenix.example.com/team/'), 'https://phoenix.example.com/team');
    assert.throws(() => normalizePhoenixUrl('ftp://127.0.0.1:6006'), /http or https/);
    assert.throws(() => normalizePhoenixUrl('http://user:secret@127.0.0.1:6006'), /credentials/);
    assert.throws(() => normalizePhoenixUrl('http://127.0.0.1:6006/?token=x'), /query/);
    assert.throws(() => normalizePhoenixUrl('not a url'), /absolute http or https URL/);
    assert.equal(isLoopbackUrl('http://localhost:6006'), true);
    assert.equal(isLoopbackUrl('http://127.0.0.2:6006'), true);
    assert.equal(isLoopbackUrl('http://[::1]:6006'), true);
    assert.equal(isLoopbackUrl('http://phoenix.internal:6006'), false);
  });

  it('keeps an explicitly empty rule list distinct from no rules', () => {
    assert.deepEqual(parseConfig({ guardrails: { rules: [] } }).guardrails.rules(), []);
    assert.equal(parseConfig({ guardrails: { rules: null } }).guardrails.rules(), undefined);
  });
});

describe('Config schema', () => {
  const schema = Config as SchemaLike;

  it('is a schemastery schema that DeepSeek Harness can validate rows with', () => {
    assert.equal(typeof schema, 'function');
    assert.equal(typeof schema['~standard'].validate, 'function');
    assert.ok(schema['~standard'].validate({ guardrails: { failMode: 'sometimes' } }).issues?.length);
    assert.ok(schema['~standard'].validate({ guardrails: { rules: [{ id: 7 }] } }).issues?.length);
    assert.equal(schema['~standard'].validate({}).issues, undefined);
  });

  it('produces output that parseConfig normalizes exactly like the raw row', () => {
    const rows: unknown[] = [
      {},
      { capture: { stripToolArgs: true }, redaction: { secrets: false } },
      {
        dataDir: '/data/relay',
        guardrails: { failMode: 'closed' },
        phoenix: { url: 'http://127.0.0.1:16006/', launch: { command: ['phoenix', 'serve'] } },
        plugins: { components: [{ kind: 'observability', enabled: true, config: { version: 4 } }] },
        agent: { enabled: false },
      },
    ];
    for (const row of rows) {
      const fromRaw = parseConfig(structuredClone(row), { DSH_HOME: '/srv/dsh' });
      const fromSchema = parseConfig(schema(structuredClone(row)), { DSH_HOME: '/srv/dsh' });
      assert.deepEqual(withoutReaders(fromSchema), withoutReaders(fromRaw));
    }
  });

  it('reads volatile rules in place, so a saved rule change needs no restart', () => {
    const rule = { id: 'no-force-push', tool: 'bash', match: 'push --force', flags: 'i', reason: 'No.', enabled: true };
    const resolved = schema({ guardrails: { rules: [rule] } });
    const config = parseConfig(resolved);
    assert.deepEqual(config.guardrails.rules(), [rule]);

    // DeepSeek Harness commits a volatile update by writing the new snapshot into the same reference.
    const reference = resolved.guardrails.rules;
    const updated = [{ ...rule, id: 'no-reset-hard', match: 'reset --hard' }];
    reference[VOLATILE_WRITE](Object.freeze(updated));
    assert.deepEqual(config.guardrails.rules(), updated);
    reference[VOLATILE_WRITE](undefined);
    assert.equal(config.guardrails.rules(), undefined);
  });

  it('reads the volatile fail mode in place, so a saved change needs no restart', () => {
    const resolved = schema({ guardrails: { failMode: 'closed' } });
    const config = parseConfig(resolved);
    assert.equal(config.guardrails.failMode(), 'closed');
    resolved.guardrails.failMode[VOLATILE_WRITE]('open');
    assert.equal(config.guardrails.failMode(), 'open');
    assert.equal(parseConfig(schema({})).guardrails.failMode(), 'open', 'open by default');
  });

  it('knows whether DeepSeek Harness validated the row with the schema (live fields) or passed it raw', () => {
    assert.equal(parseConfig(schema({})).guardrails.live, true, 'even a row without guardrails');
    assert.equal(parseConfig(schema({ guardrails: { rules: [], failMode: 'closed' } })).guardrails.live, true);
    assert.equal(parseConfig({}).guardrails.live, false);
    assert.equal(parseConfig({ guardrails: { rules: [], failMode: 'closed' } }).guardrails.live, false);
  });
});

/** Compare normalized configs without their function-valued readers. */
function withoutReaders(config: ReturnType<typeof parseConfig>) {
  // `live` records how the row arrived (schema output or raw), not what it says.
  const { live: _live, ...guardrails } = config.guardrails;
  return {
    ...config,
    guardrails: { ...guardrails, rules: config.guardrails.rules(), failMode: config.guardrails.failMode() },
  };
}
