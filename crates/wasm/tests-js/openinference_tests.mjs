// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { startCollector } from '../../../scripts/test-support/otel_test_utils.mjs';
import { assertBodyContains, unique, wasm } from './test_support.mjs';

test('WebAssembly package exposes OpenInference config defaults', () => {
  const config = wasm.defaultOpenInferenceConfig();
  assert.equal(config.transport, 'http_binary');
  assert.equal(config.endpoint, undefined);
  assert.equal(config.serviceName, 'nemo-relay');
  assert.equal(config.instrumentationScope, 'nemo-relay-openinference');
  assert.equal(config.timeoutMillis, 3000);
  assert.equal(config.headers instanceof Map, true);
  assert.equal(config.headers.size, 0);
  assert.equal(config.resourceAttributes instanceof Map, true);
  assert.equal(config.resourceAttributes.size, 0);
});

test('WebAssembly OpenInference subscriber supports lifecycle methods from mutable config objects', () => {
  const config = wasm.defaultOpenInferenceConfig();
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

  const subscriber = new wasm.OpenInferenceSubscriber(config);
  const name = unique('wasm_openinference');
  subscriber.register(name);
  assert.equal(subscriber.deregister(name), true);
  assert.equal(subscriber.deregister(name), false);
  subscriber.forceFlush();
  subscriber.shutdown();
});

test('WebAssembly OpenInference subscriber rejects invalid config values', () => {
  assert.throws(
    () =>
      new wasm.OpenInferenceSubscriber({
        transport: 'grpc',
      }),
    /not supported on this target/i,
  );
  assert.throws(
    () =>
      new wasm.OpenInferenceSubscriber({
        transport: 'invalid',
      }),
    /transport must be/i,
  );
  assert.throws(
    () =>
      new wasm.OpenInferenceSubscriber({
        headers: {
          authorization: 1,
        },
      }),
    /invalid type/i,
  );
  assert.throws(
    () =>
      new wasm.OpenInferenceSubscriber({
        resourceAttributes: {
          env: 1,
        },
      }),
    /invalid type/i,
  );
});

test('WebAssembly package exports OpenInference scope push/pop and mark events end to end', async () => {
  const collector = await startCollector();
  const config = wasm.defaultOpenInferenceConfig();
  config.endpoint = collector.endpoint;
  config.serviceName = 'wasm-agent';

  const subscriber = new wasm.OpenInferenceSubscriber(config);
  const name = unique('wasm_openinference_e2e');
  subscriber.register(name);

  try {
    const scope = wasm.pushScope(
      'openinference_scope',
      wasm.ScopeType.Agent,
      null,
      0,
      {
        scope: true,
      },
      null,
    );
    wasm.event(
      'openinference_mark',
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
    assertBodyContains(request.body, 'openinference.span.kind');
    assertBodyContains(request.body, 'AGENT');
    assertBodyContains(request.body, 'metadata');
    assertBodyContains(request.body, 'openinference_mark');
  } finally {
    subscriber.deregister(name);
    subscriber.shutdown();
    await collector.close();
  }
});
