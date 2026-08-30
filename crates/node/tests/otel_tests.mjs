// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { assertOtlpStringAttribute, startCollector } from '../../../scripts/test-support/otel_test_utils.mjs';

const require = createRequire(import.meta.url);
const {
  LogSeverity,
  MetricKind,
  MetricTemporality,
  MetricValueType,
  OpenTelemetryLogSubscriber,
  OpenTelemetryMetricSubscriber,
  OpenTelemetrySubscriber,
  ScopeType,
  pushScope,
  popScope,
  event,
  metric,
} = require('../index.js');

function uniqueId(prefix) {
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function assertBodyContains(body, text) {
  assert.equal(body.includes(Buffer.from(text, 'utf8')), true, `expected OTLP payload to contain ${text}`);
}

describe('OpenTelemetrySubscriber', () => {
  it('constructs from a mutable config object and supports lifecycle methods', () => {
    const subscriber = new OpenTelemetrySubscriber({
      type: 'full',
      endpoint: 'http://localhost:4318/v1/traces',
      serviceName: 'node-agent',
      serviceNamespace: 'agents',
      serviceVersion: '1.0.0',
      instrumentationScope: 'node-tests',
      timeoutMillis: 1250,
      completedSpanContextTtlMillis: 4294967296n,
      headers: {
        authorization: 'Bearer token',
      },
      resourceAttributes: {
        'deployment.environment': 'test',
      },
      markProjection: 'tool',
      markExcludeNames: ['custom.mark'],
      attributeMappings: [{ key: 'nemo_relay.model_name', alias: 'model.alias' }],
      promoteMetadataPrefixes: ['nv.'],
    });

    const name = uniqueId('node_otel');
    subscriber.register(name);
    assert.equal(subscriber.deregister(name), true);
    assert.equal(subscriber.deregister(name), false);
    subscriber.forceFlush();
    assert.deepEqual(subscriber.runtimeDiagnostics(), []);
    subscriber.shutdown();
  });

  it('rejects invalid config values', () => {
    assert.throws(
      () =>
        new OpenTelemetrySubscriber({
          type: 'full',
          endpoint: 'http://localhost:4318/v1/traces',
          transport: 'invalid',
        }),
      /transport must be/i,
    );
    assert.throws(
      () =>
        new OpenTelemetrySubscriber({
          type: 'full',
          endpoint: 'http://localhost:4318/v1/traces',
          headers: {
            authorization: 1,
          },
        }),
      /headers must be an object of string values/i,
    );
    assert.throws(
      () =>
        new OpenTelemetrySubscriber({
          type: 'full',
          endpoint: 'http://localhost:4318/v1/traces',
          headerEnv: {
            authorization: 1,
          },
        }),
      /headerEnv must be an object of string values/i,
    );
    assert.throws(
      () =>
        new OpenTelemetrySubscriber({
          type: 'full',
          endpoint: 'http://localhost:4318/v1/traces',
          resourceAttributes: {
            env: 1,
          },
        }),
      /resourceAttributes must be an object of string values/i,
    );
    assert.throws(
      () =>
        new OpenTelemetrySubscriber({
          type: 'full',
          endpoint: 'http://localhost:4318/v1/traces',
          attributeMappings: [{ key: '', alias: 'model.alias' }],
        }),
      /attribute mapping key must not be blank/i,
    );
    assert.throws(
      () =>
        new OpenTelemetrySubscriber({
          type: 'full',
          endpoint: 'http://localhost:4318/v1/traces',
          promoteMetadataPrefixes: ['nv.*'],
        }),
      /literal prefix, not a glob/i,
    );
    assert.throws(() => new OpenTelemetrySubscriber({ endpoint: 'http://localhost:4318' }), /missing field `type`/i);
    assert.throws(() => new OpenTelemetrySubscriber({ type: 'full' }), /missing field `endpoint`/i);
    assert.throws(
      () =>
        new OpenTelemetrySubscriber({
          type: 'invalid',
          endpoint: 'http://localhost:4318/v1/traces',
        }),
      /type must be/i,
    );
    assert.throws(
      () => new OpenTelemetrySubscriber({ type: 'full', endpoint: ' \t' }),
      /endpoint must be a nonblank string/i,
    );
    assert.throws(
      () =>
        new OpenTelemetrySubscriber({
          type: 'full',
          endpoint: 'http://localhost:4318/v1/traces',
          completedSpanContextTtlMillis: 0n,
        }),
      /completedSpanContextTtlMillis must be greater than 0/i,
    );
    assert.throws(
      () =>
        new OpenTelemetrySubscriber({
          type: 'full',
          endpoint: 'http://localhost:4318/v1/traces',
          completedSpanContextTtlMillis: -1n,
        }),
      /must be a nonnegative u64 BigInt/i,
    );
  });

  it('exports scope push/pop and mark events end to end', async () => {
    const collector = await startCollector();
    const variable = `NEMO_RELAY_NODE_HEADER_${Date.now()}`;
    const secret = 'Bearer node-activation-secret';
    process.env[variable] = secret;
    const subscriber = new OpenTelemetrySubscriber({
      type: 'full',
      endpoint: collector.endpoint,
      serviceName: 'node-agent',
      promoteMetadataPrefixes: ['nv.'],
      headerEnv: { authorization: variable },
    });
    process.env[variable] = 'Bearer node-changed-secret';

    const name = uniqueId('node_otel_e2e');
    subscriber.register(name);
    try {
      const scope = pushScope('otel_scope', ScopeType.Agent, null, null, null, {
        'nv.binding': 'node',
      });
      event(
        'otel_mark',
        scope,
        {
          step: 1,
        },
        {
          source: 'node',
        },
      );
      popScope(scope, null, null, {
        'nv.binding': 'node',
      });

      subscriber.forceFlush();
      const request = await collector.nextRequest();
      assert.equal(request.url, '/v1/traces');
      assert.equal(request.headers['content-type'], 'application/x-protobuf');
      assert.equal(request.headers.authorization, secret);
      assert.ok(request.body.length > 0);
      assert.equal(request.body.includes(Buffer.from(secret, 'utf8')), false);
      assertBodyContains(request.body, 'nemo_relay.mark.metadata.source');
      assertOtlpStringAttribute(request.body, 'nv.binding', 'node');
      assert.equal(
        subscriber.runtimeDiagnostics().some((entry) => entry.message.includes(secret)),
        false,
      );
    } finally {
      subscriber.deregister(name);
      subscriber.shutdown();
      delete process.env[variable];
      await collector.close();
    }
  });

  it('rejects header_env names that collide with static headers ignoring case', () => {
    const variable = `NEMO_RELAY_NODE_DUPLICATE_${Date.now()}`;
    process.env[variable] = 'Bearer secret';
    try {
      assert.throws(
        () =>
          new OpenTelemetrySubscriber({
            type: 'full',
            endpoint: 'http://localhost:4318/v1/traces',
            headers: { Authorization: 'static' },
            headerEnv: { authorization: variable },
          }),
        /unique across headers and header_env/i,
      );
    } finally {
      delete process.env[variable];
    }
  });

  it('rejects unset, blank, and invalid header_env values without exposing secrets', () => {
    const variable = `NEMO_RELAY_NODE_INVALID_HEADER_${Date.now()}`;
    const config = {
      type: 'full',
      endpoint: 'http://localhost:4318/v1/traces',
      headerEnv: { authorization: variable },
    };
    delete process.env[variable];
    assert.throws(() => new OpenTelemetrySubscriber(config), /is not set/i);

    process.env[variable] = '  ';
    assert.throws(() => new OpenTelemetrySubscriber(config), /nonblank value/i);

    const secret = 'relay-node-secret';
    process.env[variable] = `${secret}\ninvalid`;
    try {
      assert.throws(
        () => new OpenTelemetrySubscriber(config),
        (error) => /valid header value/i.test(error.message) && !error.message.includes(secret),
      );
    } finally {
      delete process.env[variable];
    }
  });

  it('exports the GenAI agent projection end to end', async () => {
    const collector = await startCollector();
    const subscriber = new OpenTelemetrySubscriber({
      type: 'gen_ai',
      endpoint: collector.endpoint,
    });

    const name = uniqueId('node_gen_ai_e2e');
    subscriber.register(name);
    try {
      const scope = pushScope('research-agent', ScopeType.Agent, null, null, null, null);
      popScope(scope);

      subscriber.forceFlush();
      const request = await collector.nextRequest();
      assert.equal(request.url, '/v1/traces');
      assertBodyContains(request.body, 'invoke_agent research-agent');
      assertBodyContains(request.body, 'gen_ai.operation.name');
      assert.equal(request.body.includes(Buffer.from('nemo_relay.', 'utf8')), false);
    } finally {
      subscriber.deregister(name);
      subscriber.shutdown();
      await collector.close();
    }
  });
});

describe('OpenTelemetry log and metric subscribers', () => {
  it('constructs signal-specific subscribers and supports lifecycle methods', (t) => {
    const variable = uniqueId('NEMO_RELAY_NODE_SIGNAL_HEADER');
    process.env[variable] = 'signal-route';
    t.after(() => delete process.env[variable]);
    const logSubscriber = new OpenTelemetryLogSubscriber({
      endpoint: 'http://localhost:4318/v1/logs',
      minimumSeverity: LogSeverity.Warn,
      maxQueueSize: 32,
      maxExportBatchSize: 16,
      scheduledDelayMillis: 100,
      headers: { authorization: 'Bearer token' },
      headerEnv: { 'x-relay-route': variable },
      resourceAttributes: { 'deployment.environment': 'test' },
    });
    const logName = uniqueId('node_otel_log');
    logSubscriber.register(logName);
    assert.throws(() => logSubscriber.register(logName), /already exists/i);
    assert.equal(logSubscriber.deregister(logName), true);
    assert.equal(logSubscriber.deregister(logName), false);
    logSubscriber.forceFlush();
    assert.deepEqual(logSubscriber.runtimeDiagnostics(), []);
    logSubscriber.shutdown();

    const metricSubscriber = new OpenTelemetryMetricSubscriber({
      endpoint: 'http://localhost:4318/v1/metrics',
      exportIntervalMillis: 100,
      temporality: MetricTemporality.Delta,
      maxInstruments: 32,
      cardinalityLimit: 100,
      headers: { authorization: 'Bearer token' },
      headerEnv: { 'x-relay-route': variable },
      resourceAttributes: { 'deployment.environment': 'test' },
    });
    const metricName = uniqueId('node_otel_metric');
    metricSubscriber.register(metricName);
    assert.throws(() => metricSubscriber.register(metricName), /already exists/i);
    assert.equal(metricSubscriber.deregister(metricName), true);
    assert.equal(metricSubscriber.deregister(metricName), false);
    metricSubscriber.forceFlush();
    assert.deepEqual(metricSubscriber.runtimeDiagnostics(), []);
    metricSubscriber.shutdown();
  });

  it('validates signal-specific limits', () => {
    assert.throws(
      () =>
        new OpenTelemetryLogSubscriber({
          endpoint: 'http://localhost:4318/v1/logs',
          maxQueueSize: 0,
        }),
      /max_queue_size must be greater than 0/,
    );
    assert.throws(
      () => new OpenTelemetryLogSubscriber({ endpoint: 'http://localhost:4318', completedSpanContextTtlMillis: 0n }),
      /completedSpanContextTtlMillis must be greater than 0/i,
    );
    assert.throws(
      () =>
        new OpenTelemetryMetricSubscriber({
          endpoint: 'http://localhost:4318/v1/metrics',
          cardinalityLimit: 0,
        }),
      /cardinality_limit must be greater than 0/,
    );
  });

  it('exports logs to the logs signal path', async () => {
    const collector = await startCollector();
    const subscriber = new OpenTelemetryLogSubscriber({ endpoint: collector.endpoint });
    const name = uniqueId('node_otel_log_e2e');
    subscriber.register(name);
    try {
      event('log_mark', null, { message: 'ready' }, null, null, null, LogSeverity.Info);
      subscriber.forceFlush();
      const request = await collector.nextRequest();
      assert.equal(request.url, '/v1/logs');
      assertBodyContains(request.body, 'log_mark');
    } finally {
      subscriber.deregister(name);
      subscriber.shutdown();
      await collector.close();
    }
  });

  it('exports metrics to the metrics signal path', async () => {
    const collector = await startCollector();
    const subscriber = new OpenTelemetryMetricSubscriber({
      endpoint: collector.endpoint,
      exportIntervalMillis: 100,
    });
    const name = uniqueId('node_otel_metric_e2e');
    subscriber.register(name);
    try {
      metric(
        'metric_mark',
        [
          {
            name: 'relay.tokens',
            kind: MetricKind.Counter,
            valueType: MetricValueType.U64,
            value: 3,
          },
        ],
        null,
        null,
        null,
      );
      subscriber.forceFlush();
      const request = await collector.nextRequest();
      assert.equal(request.url, '/v1/metrics');
      assertBodyContains(request.body, 'relay.tokens');
    } finally {
      subscriber.deregister(name);
      subscriber.shutdown();
      await collector.close();
    }
  });
});
