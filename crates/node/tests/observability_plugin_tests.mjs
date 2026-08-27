// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdtempSync, readdirSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const require = createRequire(import.meta.url);
const plugin = require('../plugin.js');
const observability = require('../observability.js');
const { ScopeType, pushScope, popScope, event } = require('../index.js');

function tempDir(prefix) {
  return mkdtempSync(join(tmpdir(), `nemo-relay-${prefix}-`));
}

describe('observability plugin helpers', () => {
  it('builds defaults and plugin component shape', () => {
    assert.deepEqual(observability.defaultConfig(), { version: 4 });
    assert.equal(
      observability.ComponentSpec({ version: 3, enable_full_payloads: true }).config.enable_full_payloads,
      true,
    );
    assert.deepEqual(observability.atofConfig(), { enabled: false });
    assert.deepEqual(observability.atifConfig(), {
      enabled: false,
      agent_name: 'NeMo Relay',
      model_name: 'unknown',
      filename_template: 'nemo-relay-atif-{session_id}.json',
    });
    assert.deepEqual(observability.openTelemetryConfig(), {
      enabled: false,
      endpoints: [],
    });
    assert.deepEqual(
      observability.openTelemetryEndpoint({
        type: 'gen_ai',
        endpoint: 'http://localhost:4318/v1/traces',
        header_env: { authorization: 'OTEL_AUTHORIZATION' },
        max_queue_size: 4096,
        max_export_batch_size: 256,
        scheduled_delay_millis: 750,
      }),
      {
        type: 'gen_ai',
        endpoint: 'http://localhost:4318/v1/traces',
        transport: 'http_binary',
        headers: {},
        header_env: { authorization: 'OTEL_AUTHORIZATION' },
        resource_attributes: {},
        promote_metadata_prefixes: [],
        promote_resource_metadata_prefixes: [],
        service_name: 'unknown_service',
        instrumentation_scope: 'opentelemetry',
        timeout_millis: 3000,
        completed_span_context_ttl_millis: 60000,
        max_queue_size: 4096,
        max_export_batch_size: 256,
        scheduled_delay_millis: 750,
      },
    );

    const component = observability.ComponentSpec({ version: 3, atof: observability.atofConfig() });
    assert.equal(component.kind, observability.OBSERVABILITY_PLUGIN_KIND);
    assert.equal(component.enabled, true);
  });

  it('builds OTLP log and metric sections with signal defaults', () => {
    const endpoint = observability.openTelemetrySignalEndpoint({
      endpoint: 'https://collector.example/custom/logs',
      headers: { 'x-static': 'value' },
      header_env: { authorization: 'OTEL_AUTHORIZATION' },
      resource_attributes: { 'nv.project': 'observability-dev' },
      service_namespace: 'telemetry',
      service_version: '1.0',
    });
    assert.deepEqual(endpoint, {
      endpoint: 'https://collector.example/custom/logs',
      transport: 'http_binary',
      headers: { 'x-static': 'value' },
      header_env: { authorization: 'OTEL_AUTHORIZATION' },
      resource_attributes: { 'nv.project': 'observability-dev' },
      service_name: 'unknown_service',
      service_namespace: 'telemetry',
      service_version: '1.0',
      instrumentation_scope: 'opentelemetry',
      timeout_millis: 3000,
    });

    const logs = observability.openTelemetryLogConfig({ enabled: true, endpoints: [endpoint] });
    const metrics = observability.openTelemetryMetricConfig({ enabled: true, endpoints: [] });
    assert.deepEqual(observability.openTelemetryLogConfig(), {
      enabled: false,
      minimum_severity: 'info',
      max_queue_size: 2048,
      max_export_batch_size: 512,
      scheduled_delay_millis: 1000,
      completed_span_context_ttl_millis: 60000,
    });
    assert.deepEqual(observability.openTelemetryMetricConfig(), {
      enabled: false,
      export_interval_millis: 60000,
      temporality: 'cumulative',
      max_instruments: 256,
      cardinality_limit: 2000,
    });
    assert.deepEqual(observability.openTelemetryConfig({ enabled: true, logs, metrics }), {
      enabled: true,
      endpoints: [],
      logs,
      metrics,
    });
    assert.deepEqual(metrics.endpoints, []);

    assert.throws(() => observability.openTelemetrySignalEndpoint(), /config is required/);
    assert.throws(() => observability.openTelemetrySignalEndpoint({ endpoint: ' ' }), /nonblank/);
  });

  it('initializes a version 4 log and metric configuration without exporting signals', async () => {
    const config = {
      version: 4,
      opentelemetry: observability.openTelemetryConfig({
        enabled: true,
        logs: observability.openTelemetryLogConfig({
          enabled: true,
          endpoints: [observability.openTelemetrySignalEndpoint({ endpoint: 'http://127.0.0.1:4318/v1/logs' })],
        }),
        metrics: observability.openTelemetryMetricConfig({
          enabled: true,
          endpoints: [observability.openTelemetrySignalEndpoint({ endpoint: 'http://127.0.0.1:4318/v1/metrics' })],
        }),
      }),
    };

    await plugin.initialize({
      version: 1,
      components: [observability.ComponentSpec(config)],
    });
    try {
      assert.deepEqual(plugin.report()?.runtime_diagnostics ?? [], []);
    } finally {
      plugin.clear();
    }
  });

  it('lists builtin observability kind and validates bad values', () => {
    assert.throws(() => observability.openTelemetryEndpoint(), /config is required/);
    assert.throws(
      () => observability.openTelemetryEndpoint({ type: 'invalid', endpoint: 'http://localhost' }),
      /type must be/,
    );
    assert.throws(() => observability.openTelemetryEndpoint({ type: 'full', endpoint: ' ' }), /nonblank/);
    assert.equal(plugin.listKinds().includes(observability.OBSERVABILITY_PLUGIN_KIND), true);
    const report = plugin.validate({
      version: 1,
      components: [
        observability.ComponentSpec({
          version: 3,
          atof: observability.atofConfig({ sinks: [{ type: 'file', mode: 'bad' }] }),
          atif: observability.atifConfig({ filename_template: 'missing-placeholder.json' }),
        }),
      ],
    });
    assert.deepEqual(report.diagnostics.map((diagnostic) => diagnostic.field).sort(), [
      'filename_template',
      'sinks[0].mode',
    ]);
  });

  it('serializes ATOF stream sinks', () => {
    const config = observability.atofConfig({
      sinks: [
        {
          type: 'stream',
          name: 'switchyard',
          url: 'http://localhost:8080/events',
          transport: 'http_post',
          headers: { 'X-Test': 'yes' },
          timeout_millis: 1000,
          field_name_policy: 'replace_dots',
        },
      ],
    });

    assert.deepEqual(config.sinks, [
      {
        type: 'stream',
        name: 'switchyard',
        url: 'http://localhost:8080/events',
        transport: 'http_post',
        headers: { 'X-Test': 'yes' },
        timeout_millis: 1000,
        field_name_policy: 'replace_dots',
      },
    ]);
  });

  it('passes through mixed ATIF remote storage config', () => {
    const s3 = {
      type: 's3',
      bucket: 'archive',
      key_prefix: 'runs/',
    };
    const http = {
      type: 'http',
      endpoint: 'https://example.com/atif',
      headers: { 'x-static': 'value' },
      header_env: { authorization: 'NEMO_RELAY_ATIF_HTTP_AUTH' },
      timeout_millis: 1500,
    };
    const config = observability.atifConfig({
      enabled: true,
      storage: [s3, http],
    });
    assert.deepEqual(config.storage, [s3, http]);
  });

  it('activates ATOF and ATIF file sinks', async () => {
    const outputDirectory = tempDir('node-observability-plugin');
    const config = {
      version: 3,
      atof: observability.atofConfig({
        enabled: true,
        sinks: [{ type: 'file', output_directory: outputDirectory, filename: 'events.jsonl', mode: 'overwrite' }],
      }),
      atif: observability.atifConfig({
        enabled: true,
        agent_name: 'node-agent',
        agent_version: '1.2.3',
        model_name: 'node-model',
        tool_definitions: [{ name: 'search' }],
        extra: { binding: 'node' },
        output_directory: outputDirectory,
        filename_template: 'trajectory-{session_id}.json',
      }),
    };

    await plugin.initialize({
      version: 1,
      components: [observability.ComponentSpec(config)],
    });
    let scope = null;
    try {
      scope = pushScope('node-observability-agent', ScopeType.Agent, null, null, null, null, { agent: true });
      event('node-mark', scope, { step: 1 }, null);
      popScope(scope, { done: true });
      scope = null;
    } finally {
      plugin.clear();
      if (scope) {
        popScope(scope, { done: true });
      }
    }

    const records = readFileSync(join(outputDirectory, 'events.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
    assert.deepEqual(
      records.map((record) => record.kind),
      ['scope', 'mark', 'scope'],
    );

    const trajectory = JSON.parse(readFileSync(join(outputDirectory, `trajectory-${records[0].uuid}.json`), 'utf8'));
    assert.equal(trajectory.agent.name, 'node-agent');
    assert.equal(trajectory.agent.version, '1.2.3');
    assert.equal(trajectory.agent.model_name, 'node-model');
    assert.equal(trajectory.agent.tool_definitions[0].name, 'search');
    assert.equal(trajectory.agent.extra.binding, 'node');
    assert.match(JSON.stringify(trajectory.extra), /node-observability-agent/);
  });

  it('splits ATIF files for multiple top-level agent scopes', async () => {
    const outputDirectory = tempDir('node-observability-plugin-multi-agent');
    const config = {
      version: 3,
      atif: observability.atifConfig({
        enabled: true,
        output_directory: outputDirectory,
        filename_template: 'trajectory-{session_id}.json',
      }),
    };

    await plugin.initialize({
      version: 1,
      components: [observability.ComponentSpec(config)],
    });

    let first = null;
    let nested = null;
    let second = null;
    let firstUuid = null;
    let secondUuid = null;
    try {
      first = pushScope('node-first-agent', ScopeType.Agent, null, null, null, null, { agent: 'first' });
      firstUuid = first.uuid;
      event('node-first-mark', first, { agent: 'first' }, null);
      nested = pushScope('node-nested-agent', ScopeType.Agent, null, null, null, null, { agent: 'nested' });
      event('node-nested-mark', nested, { agent: 'nested' }, null);
      popScope(nested, { done: true });
      nested = null;
      popScope(first, { done: true });
      first = null;

      second = pushScope('node-second-agent', ScopeType.Agent, null, null, null, null, { agent: 'second' });
      secondUuid = second.uuid;
      event('node-second-mark', second, { agent: 'second' }, null);
      popScope(second, { done: true });
      second = null;
    } finally {
      plugin.clear();
      if (nested) {
        popScope(nested, { done: true });
      }
      if (first) {
        popScope(first, { done: true });
      }
      if (second) {
        popScope(second, { done: true });
      }
    }

    const files = readdirSync(outputDirectory).filter((name) => name.startsWith('trajectory-'));
    assert.equal(files.length, 2);

    const firstTrajectory = JSON.parse(readFileSync(join(outputDirectory, `trajectory-${firstUuid}.json`), 'utf8'));
    const secondTrajectory = JSON.parse(readFileSync(join(outputDirectory, `trajectory-${secondUuid}.json`), 'utf8'));
    const firstPayload = JSON.stringify(firstTrajectory.extra);
    const secondPayload = JSON.stringify(secondTrajectory.extra);

    assert.match(firstPayload, /node-first-agent/);
    assert.match(firstPayload, /node-nested-agent/);
    assert.doesNotMatch(firstPayload, /node-second-agent/);
    assert.match(secondPayload, /node-second-agent/);
    assert.doesNotMatch(secondPayload, /node-first-agent/);
    assert.doesNotMatch(secondPayload, /node-nested-agent/);
  });
});
