// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { startCollector } from '../../../scripts/test-support/otel_test_utils.mjs';
import { unique, wasm } from './test_support.mjs';

test('WebAssembly package exposes OpenTelemetry config defaults', () => {
  const config = wasm.defaultOpenTelemetryConfig();
  assert.equal(config.transport, 'http_binary');
  assert.equal(config.endpoint, undefined);
  assert.equal(config.serviceName, 'nemo-flow');
  assert.equal(config.instrumentationScope, 'nemo-flow-otel');
  assert.equal(config.timeoutMillis, 3000);
  assert.equal(config.headers instanceof Map, true);
  assert.equal(config.headers.size, 0);
  assert.equal(config.resourceAttributes instanceof Map, true);
  assert.equal(config.resourceAttributes.size, 0);
});

test('WebAssembly OpenTelemetry subscriber supports lifecycle methods from mutable config objects', () => {
  const config = wasm.defaultOpenTelemetryConfig();
  config.endpoint = 'http://localhost:4318/v1/traces';
  config.serviceName = 'wasm-agent';
  config.serviceNamespace = 'agents';
  config.serviceVersion = '1.0.0';
  config.instrumentationScope = 'wasm-tests';
  config.timeoutMillis = 1250;
  config.headers = {
    authorization: 'Bearer token',
  };
  config.resourceAttributes = {
    'deployment.environment': 'test',
  };

  const subscriber = new wasm.OpenTelemetrySubscriber(config);
  const name = unique('wasm_otel');
  subscriber.register(name);
  assert.equal(subscriber.deregister(name), true);
  assert.equal(subscriber.deregister(name), false);
  subscriber.forceFlush();
  subscriber.shutdown();
});

test('WebAssembly OpenTelemetry subscriber rejects invalid config values', () => {
  assert.throws(
    () =>
      new wasm.OpenTelemetrySubscriber({
        transport: 'grpc',
      }),
    /not supported on this target/i,
  );
  assert.throws(
    () =>
      new wasm.OpenTelemetrySubscriber({
        transport: 'invalid',
      }),
    /transport must be/i,
  );
  assert.throws(
    () =>
      new wasm.OpenTelemetrySubscriber({
        headers: {
          authorization: 1,
        },
      }),
    /invalid type/i,
  );
  assert.throws(
    () =>
      new wasm.OpenTelemetrySubscriber({
        resourceAttributes: {
          env: 1,
        },
      }),
    /invalid type/i,
  );
});

test('WebAssembly package exports scope push/pop and mark events end to end', async () => {
  const collector = await startCollector();
  const config = wasm.defaultOpenTelemetryConfig();
  config.endpoint = collector.endpoint;
  config.serviceName = 'wasm-agent';

  const subscriber = new wasm.OpenTelemetrySubscriber(config);
  const name = unique('wasm_otel_e2e');
  subscriber.register(name);

  try {
    const scope = wasm.pushScope(
      'otel_scope',
      wasm.ScopeType.Agent,
      null,
      0,
      {
        scope: true,
      },
      null,
    );
    wasm.event(
      'otel_mark',
      scope,
      {
        step: 1,
      },
      {
        source: 'wasm',
      },
    );
    wasm.popScope(scope);

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
