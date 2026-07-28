// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { startCollector } from '../../../scripts/test-support/otel_test_utils.mjs';

const require = createRequire(import.meta.url);
const { OpenTelemetrySubscriber, ScopeType, pushScope, popScope, event } = require('../index.js');

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
      headers: {
        authorization: 'Bearer token',
      },
      resourceAttributes: {
        'deployment.environment': 'test',
      },
      markProjection: 'tool',
      markExcludeNames: ['custom.mark'],
      attributeMappings: [{ key: 'nemo_relay.model_name', alias: 'model.alias' }],
    });

    const name = uniqueId('node_otel');
    subscriber.register(name);
    assert.equal(subscriber.deregister(name), true);
    assert.equal(subscriber.deregister(name), false);
    subscriber.forceFlush();
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
      () => new OpenTelemetrySubscriber({ endpoint: 'http://localhost:4318' }),
      /missing field `type`/i,
    );
    assert.throws(
      () => new OpenTelemetrySubscriber({ type: 'full' }),
      /missing field `endpoint`/i,
    );
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
  });

  it('exports scope push/pop and mark events end to end', async () => {
    const collector = await startCollector();
    const subscriber = new OpenTelemetrySubscriber({
      type: 'full',
      endpoint: collector.endpoint,
      serviceName: 'node-agent',
    });

    const name = uniqueId('node_otel_e2e');
    subscriber.register(name);
    try {
      const scope = pushScope('otel_scope', ScopeType.Agent, null, null, null, null);
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
      popScope(scope);

      subscriber.forceFlush();
      const request = await collector.nextRequest();
      assert.equal(request.url, '/v1/traces');
      assert.equal(request.headers['content-type'], 'application/x-protobuf');
      assert.ok(request.body.length > 0);
      assertBodyContains(request.body, 'nemo_relay.mark.metadata.source');
    } finally {
      subscriber.deregister(name);
      subscriber.shutdown();
      await collector.close();
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
