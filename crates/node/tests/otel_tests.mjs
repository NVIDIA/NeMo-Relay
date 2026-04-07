// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { startCollector } from '../../../scripts/otel_test_utils.mjs';

const require = createRequire(import.meta.url);
const { OpenTelemetrySubscriber, ScopeType, pushScope, popScope, event } = require('../index.js');

function unique(prefix) {
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

describe('OpenTelemetrySubscriber', () => {
  it('constructs from a mutable config object and supports lifecycle methods', () => {
    const subscriber = new OpenTelemetrySubscriber({
      endpoint: 'http://localhost:4318/v1/traces',
      serviceName: 'node-agent',
      serviceNamespace: 'agents',
      serviceVersion: '1.0.0',
      instrumentationScope: 'node-tests',
      timeoutMillis: 1250,
      headers: { authorization: 'Bearer token' },
      resourceAttributes: { 'deployment.environment': 'test' },
    });

    const name = unique('node_otel');
    subscriber.register(name);
    assert.equal(subscriber.deregister(name), true);
    assert.equal(subscriber.deregister(name), false);
    subscriber.forceFlush();
    subscriber.shutdown();
  });

  it('rejects invalid config values', () => {
    assert.throws(
      () => new OpenTelemetrySubscriber({ transport: 'invalid' }),
      /transport must be/i,
    );
    assert.throws(
      () => new OpenTelemetrySubscriber({ headers: { authorization: 1 } }),
      /headers must be an object of string values/i,
    );
    assert.throws(
      () => new OpenTelemetrySubscriber({ resourceAttributes: { env: 1 } }),
      /resourceAttributes must be an object of string values/i,
    );
  });

  it('exports scope push/pop and mark events end to end', async () => {
    const collector = await startCollector();
    const subscriber = new OpenTelemetrySubscriber({
      endpoint: collector.endpoint,
      serviceName: 'node-agent',
    });

    const name = unique('node_otel_e2e');
    subscriber.register(name);
    try {
      const scope = pushScope('otel_scope', ScopeType.Agent, null, null, { scope: true }, null);
      event('otel_mark', scope, { step: 1 }, { source: 'node' });
      popScope(scope);

      subscriber.forceFlush();
      const request = await collector.nextRequest();
      assert.equal(request.url, '/v1/traces');
      assert.equal(request.headers['content-type'], 'application/x-protobuf');
      assert.ok(request.body.length > 0);
    } finally {
      subscriber.deregister(name);
      subscriber.shutdown();
      await collector.close();
    }
  });
});
