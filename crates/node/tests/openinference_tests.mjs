// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { startCollector } from '../../../scripts/test-support/otel_test_utils.mjs';

const require = createRequire(import.meta.url);
const { OpenInferenceSubscriber, ScopeType, pushScope, popScope, event } = require('../index.js');

function unique(prefix) {
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function assertBodyContains(body, text) {
  assert.equal(body.includes(Buffer.from(text, 'utf8')), true, `expected OTLP payload to contain ${text}`);
}

function assertStringAttribute(body, key, value) {
  const encoded = Buffer.concat([
    Buffer.from([0x0a, Buffer.byteLength(key)]),
    Buffer.from(key),
    Buffer.from([0x12, Buffer.byteLength(value) + 2, 0x0a, Buffer.byteLength(value)]),
    Buffer.from(value),
  ]);
  assert.equal(body.includes(encoded), true, `expected OTLP string attribute ${key}=${value}`);
}

describe('OpenInferenceSubscriber', () => {
  it('constructs from a mutable config object and supports lifecycle methods', () => {
    const subscriber = new OpenInferenceSubscriber({
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
      attributeMappings: [{ key: 'openinference.metadata.tenant', alias: 'tenant.id' }],
    });

    const name = unique('node_openinference');
    subscriber.register(name);
    assert.equal(subscriber.deregister(name), true);
    assert.equal(subscriber.deregister(name), false);
    subscriber.forceFlush();
    subscriber.shutdown();
  });

  it('rejects invalid config values', () => {
    assert.throws(
      () =>
        new OpenInferenceSubscriber({
          transport: 'invalid',
        }),
      /transport must be/i,
    );
    assert.throws(
      () =>
        new OpenInferenceSubscriber({
          headers: {
            authorization: 1,
          },
        }),
      /headers must be an object of string values/i,
    );
    assert.throws(
      () =>
        new OpenInferenceSubscriber({
          resourceAttributes: {
            env: 1,
          },
        }),
      /resourceAttributes must be an object of string values/i,
    );
    assert.throws(
      () =>
        new OpenInferenceSubscriber({
          attributeMappings: [{ key: '', alias: 'tenant.id' }],
        }),
      /attribute mapping key must not be blank/i,
    );
  });

  it('exports scope push/pop and mark events end to end', async () => {
    const collector = await startCollector();
    const subscriber = new OpenInferenceSubscriber({
      endpoint: collector.endpoint,
      serviceName: 'node-agent',
      attributeMappings: [{ key: 'openinference.metadata.tenant', alias: 'tenant.id' }],
    });

    const name = unique('node_openinference_e2e');
    subscriber.register(name);
    try {
      const scope = pushScope('openinference_scope', ScopeType.Agent, null, null, null, null);
      event(
        'openinference_mark',
        scope,
        {
          step: 1,
        },
        {
          source: 'node',
        },
      );
      popScope(scope, null, null, { tenant: 'node' });

      subscriber.forceFlush();
      const request = await collector.nextRequest();
      assert.equal(request.url, '/v1/traces');
      assert.equal(request.headers['content-type'], 'application/x-protobuf');
      assert.ok(request.body.length > 0);
      assertBodyContains(request.body, 'openinference.span.kind');
      assertBodyContains(request.body, 'AGENT');
      assertBodyContains(request.body, 'metadata');
      assertBodyContains(request.body, 'openinference_mark');
      assertStringAttribute(request.body, 'tenant.id', 'node');
    } finally {
      subscriber.deregister(name);
      subscriber.shutdown();
      await collector.close();
    }
  });
});
