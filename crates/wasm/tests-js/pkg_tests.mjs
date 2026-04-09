// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { test } from 'node:test';
import { startCollector } from '../../../scripts/otel_test_utils.mjs';

const require = createRequire(import.meta.url);
// These tests intentionally exercise only the public generated package API.
// Avoid asserting against wasm-bindgen implementation details or private helpers.
const wasm = require('../pkg-test');

function unique(prefix) {
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function assertBodyContains(body, text) {
  assert.equal(body.includes(Buffer.from(text, 'utf8')), true, `expected OTLP payload to contain ${text}`);
}

function resetScopeStack() {
  const stack = wasm.createScopeStack();
  wasm.setThreadScopeStack(stack);
  return stack;
}

function currentScope() {
  return wasm.getHandle();
}

function expectInvalidUuid(fn) {
  assert.throws(fn, /invalid UUID/i);
}

function expectInvalidLlmRequest(fn) {
  assert.throws(fn, /invalid type|LLMRequest/i);
}

async function rejectInvalidLlmRequest(promise) {
  await assert.rejects(promise, /invalid type|LLMRequest/i);
}

function expectClassError(fn) {
  assert.throws(fn, /expected instance of/i);
}

function expectAlreadyExists(fn) {
  assert.throws(fn, /already exists/i);
}

async function drainStream(stream) {
  const chunks = [];
  for (;;) {
    const next = await stream.next();
    if (next.done) {
      return chunks;
    }
    chunks.push(next.value);
  }
}

function globalRegistrationCases() {
  return [
    ['toolSanReq', wasm.registerToolSanitizeRequestGuardrail, wasm.deregisterToolSanitizeRequestGuardrail, (name, register) => register(name, 1, (toolName, args) => args)],
    ['toolSanResp', wasm.registerToolSanitizeResponseGuardrail, wasm.deregisterToolSanitizeResponseGuardrail, (name, register) => register(name, 1, (result) => result)],
    ['toolCond', wasm.registerToolConditionalExecutionGuardrail, wasm.deregisterToolConditionalExecutionGuardrail, (name, register) => register(name, 1, () => undefined)],
    ['toolReq', wasm.registerToolRequestIntercept, wasm.deregisterToolRequestIntercept, (name, register) => register(name, 1, false, (toolName, args) => args)],
    ['toolExec', wasm.registerToolExecutionIntercept, wasm.deregisterToolExecutionIntercept, (name, register) => register(name, 1, async (args, next) => next(args))],
    ['llmSanReq', wasm.registerLlmSanitizeRequestGuardrail, wasm.deregisterLlmSanitizeRequestGuardrail, (name, register) => register(name, 1, (request) => request)],
    ['llmSanResp', wasm.registerLlmSanitizeResponseGuardrail, wasm.deregisterLlmSanitizeResponseGuardrail, (name, register) => register(name, 1, (response) => response)],
    ['llmCond', wasm.registerLlmConditionalExecutionGuardrail, wasm.deregisterLlmConditionalExecutionGuardrail, (name, register) => register(name, 1, () => undefined)],
    ['llmReq', wasm.registerLlmRequestIntercept, wasm.deregisterLlmRequestIntercept, (name, register) => register(name, 1, false, (request) => request)],
    ['llmExec', wasm.registerLlmExecutionIntercept, wasm.deregisterLlmExecutionIntercept, (name, register) => register(name, 1, async (request, next) => next(request))],
    ['llmStreamExec', wasm.registerLlmStreamExecutionIntercept, wasm.deregisterLlmStreamExecutionIntercept, (name, register) => register(name, 1, async (request, next) => next(request))],
    ['subscriber', wasm.registerSubscriber, wasm.deregisterSubscriber, (name, register) => register(name, () => {})],
  ];
}

function scopeRegistrationCases(scopeUuid) {
  return [
    ['scopeToolSanReq', wasm.scopeRegisterToolSanitizeRequestGuardrail, wasm.scopeDeregisterToolSanitizeRequestGuardrail, (uuid, name, register) => register(uuid, name, 1, (toolName, args) => args)],
    ['scopeToolSanResp', wasm.scopeRegisterToolSanitizeResponseGuardrail, wasm.scopeDeregisterToolSanitizeResponseGuardrail, (uuid, name, register) => register(uuid, name, 1, (result) => result)],
    ['scopeToolCond', wasm.scopeRegisterToolConditionalExecutionGuardrail, wasm.scopeDeregisterToolConditionalExecutionGuardrail, (uuid, name, register) => register(uuid, name, 1, () => undefined)],
    ['scopeToolReq', wasm.scopeRegisterToolRequestIntercept, wasm.scopeDeregisterToolRequestIntercept, (uuid, name, register) => register(uuid, name, 1, false, (toolName, args) => args)],
    ['scopeToolExec', wasm.scopeRegisterToolExecutionIntercept, wasm.scopeDeregisterToolExecutionIntercept, (uuid, name, register) => register(uuid, name, 1, async (args, next) => next(args))],
    ['scopeLlmSanReq', wasm.scopeRegisterLlmSanitizeRequestGuardrail, wasm.scopeDeregisterLlmSanitizeRequestGuardrail, (uuid, name, register) => register(uuid, name, 1, (request) => request)],
    ['scopeLlmSanResp', wasm.scopeRegisterLlmSanitizeResponseGuardrail, wasm.scopeDeregisterLlmSanitizeResponseGuardrail, (uuid, name, register) => register(uuid, name, 1, (response) => response)],
    ['scopeLlmCond', wasm.scopeRegisterLlmConditionalExecutionGuardrail, wasm.scopeDeregisterLlmConditionalExecutionGuardrail, (uuid, name, register) => register(uuid, name, 1, () => undefined)],
    ['scopeLlmReq', wasm.scopeRegisterLlmRequestIntercept, wasm.scopeDeregisterLlmRequestIntercept, (uuid, name, register) => register(uuid, name, 1, false, (request) => request)],
    ['scopeLlmExec', wasm.scopeRegisterLlmExecutionIntercept, wasm.scopeDeregisterLlmExecutionIntercept, (uuid, name, register) => register(uuid, name, 1, async (request, next) => next(request))],
    ['scopeLlmStreamExec', wasm.scopeRegisterLlmStreamExecutionIntercept, wasm.scopeDeregisterLlmStreamExecutionIntercept, (uuid, name, register) => register(uuid, name, 1, async (request, next) => next(request))],
    ['scopeSubscriber', wasm.scopeRegisterSubscriber, wasm.scopeDeregisterSubscriber, (uuid, name, register) => register(uuid, name, () => {})],
  ];
}

test('scope stack and lifecycle wrappers work from the generated Node package', () => {
  const stack = resetScopeStack();
  const root = wasm.getHandle();

  assert.equal(wasm.scopeStackActive(), true);
  assert.equal(root.scopeType, 0);
  assert.equal(typeof root.uuid, 'string');
  assert.ok(root.uuid.length > 0);
  assert.equal(root.parentUuid, undefined);
  assert.equal(root.data, null);
  assert.equal(root.metadata, null);

  const scope = wasm.pushScope('pkg_scope', 1, null, 1, { scope: true }, { source: 'js' });
  assert.equal(scope.name, 'pkg_scope');
  assert.equal(scope.scopeType, 1);
  assert.equal(scope.attributes, 1);
  assert.deepEqual(scope.data, { scope: true });
  assert.deepEqual(scope.metadata, { source: 'js' });
  assert.equal(typeof scope.parentUuid, 'string');

  const callbackValue = wasm.withScope(
    'pkg_with_scope',
    1,
    (handle) => ({ name: handle.name, type: handle.scopeType, uuid: handle.uuid }),
    null,
    2,
    { nested: true },
    { origin: 'callback' },
  );
  assert.equal(callbackValue.name, 'pkg_with_scope');
  assert.equal(callbackValue.type, 1);
  assert.equal(typeof callbackValue.uuid, 'string');

  wasm.popScope(scope);
  root.free();
  scope.free();
  stack.free();
});

test('WASM package exposes OpenTelemetry config defaults and subscriber lifecycle', () => {
  const config = wasm.defaultOpenTelemetryConfig();
  assert.equal(config.transport, 'http_binary');
  assert.equal(config.endpoint, undefined);
  assert.equal(config.service_name, 'nemo-flow');
  assert.equal(config.instrumentation_scope, 'nemo-flow-otel');
  assert.equal(config.timeout_millis, 3000);
  assert.equal(config.headers instanceof Map, true);
  assert.equal(config.headers.size, 0);
  assert.equal(config.resource_attributes instanceof Map, true);
  assert.equal(config.resource_attributes.size, 0);

  config.endpoint = 'http://localhost:4318/v1/traces';
  config.service_name = 'wasm-agent';
  config.service_namespace = 'agents';
  config.service_version = '1.0.0';
  config.instrumentation_scope = 'wasm-tests';
  config.timeout_millis = 1250;
  config.headers = { authorization: 'Bearer token' };
  config.resource_attributes = { 'deployment.environment': 'test' };

  assert.throws(
    () => new wasm.OpenTelemetrySubscriber({ transport: 'grpc' }),
    /not supported on this target/i,
  );
  assert.throws(
    () => new wasm.OpenTelemetrySubscriber({ transport: 'invalid' }),
    /transport must be/i,
  );
  assert.throws(
    () => new wasm.OpenTelemetrySubscriber({ headers: { authorization: 1 } }),
    /invalid type/i,
  );
});

test('WASM package exports scope push/pop and mark events end to end', async () => {
  const collector = await startCollector();
  const config = wasm.defaultOpenTelemetryConfig();
  config.endpoint = collector.endpoint;
  config.service_name = 'wasm-agent';

  const subscriber = new wasm.OpenTelemetrySubscriber(config);
  const name = unique('wasm_otel');
  subscriber.register(name);

  try {
    const scope = wasm.pushScope('otel_scope', 0, null, 0, { scope: true }, null);
    wasm.event('otel_mark', scope, { step: 1 }, { source: 'wasm' });
    wasm.popScope(wasm.getHandle());

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

test('WASM package exposes OpenInference config defaults and subscriber lifecycle', () => {
  const config = wasm.defaultOpenInferenceConfig();
  assert.equal(config.transport, 'http_binary');
  assert.equal(config.endpoint, undefined);
  assert.equal(config.service_name, 'nemo-flow');
  assert.equal(config.instrumentation_scope, 'nemo-flow-openinference');
  assert.equal(config.timeout_millis, 3000);
  assert.equal(config.headers instanceof Map, true);
  assert.equal(config.headers.size, 0);
  assert.equal(config.resource_attributes instanceof Map, true);
  assert.equal(config.resource_attributes.size, 0);

  config.endpoint = 'http://localhost:4318/v1/traces';
  config.service_name = 'wasm-agent';
  config.service_namespace = 'agents';
  config.service_version = '1.0.0';
  config.instrumentation_scope = 'wasm-tests';
  config.timeout_millis = 1250;
  config.headers = { authorization: 'Bearer token' };
  config.resource_attributes = { 'deployment.environment': 'test' };

  assert.throws(
    () => new wasm.OpenInferenceSubscriber({ transport: 'grpc' }),
    /not supported on this target/i,
  );
  assert.throws(
    () => new wasm.OpenInferenceSubscriber({ transport: 'invalid' }),
    /transport must be/i,
  );
  assert.throws(
    () => new wasm.OpenInferenceSubscriber({ headers: { authorization: 1 } }),
    /invalid type/i,
  );
});

test('WASM package exports OpenInference scope push/pop and mark events end to end', async () => {
  const collector = await startCollector();
  const config = wasm.defaultOpenInferenceConfig();
  config.endpoint = collector.endpoint;
  config.service_name = 'wasm-agent';

  const subscriber = new wasm.OpenInferenceSubscriber(config);
  const name = unique('wasm_openinference');
  subscriber.register(name);

  try {
    const scope = wasm.pushScope('openinference_scope', 0, null, 0, { scope: true }, null);
    wasm.event('openinference_mark', scope, { step: 1 }, { source: 'wasm' });
    wasm.popScope(wasm.getHandle());

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

test('WASM JS wrappers cover nullable inputs, getters, and throw paths', async () => {
  const stack = new wasm.WasmScopeStack();
  wasm.setThreadScopeStack(stack);

  const root = currentScope();
  const rootUuid = root.uuid;
  root.free();

  const scope = wasm.pushScope('optional_scope', 1, currentScope(), undefined, null, undefined);
  assert.equal(scope.parentUuid, rootUuid);
  assert.equal(scope.data, null);
  assert.equal(scope.metadata, null);

  const currentStack = wasm.currentScopeStack();
  currentStack.free();

  const request = new wasm.WasmLLMRequest(
    { trace: '1' },
    { model: 'demo-model', messages: [] },
  );
  assert.deepEqual(request.headers, { trace: '1' });
  assert.deepEqual(request.content, { model: 'demo-model', messages: [] });
  request.headers = { trace: '2', nested: true };
  request.content = { model: 'demo-model', messages: [{ role: 'user', content: 'hi' }] };
  assert.equal(request.headers.trace, '2');
  assert.equal(request.content.messages[0].content, 'hi');

  const toolHandle = wasm.toolCall(
    'optional_tool',
    { value: 2 },
    currentScope(),
    undefined,
    null,
    undefined,
  );
  assert.equal(toolHandle.name, 'optional_tool');
  assert.equal(typeof toolHandle.uuid, 'string');
  assert.equal(toolHandle.attributes, 0);
  assert.equal(toolHandle.parentUuid, scope.uuid);
  wasm.toolCallEnd(toolHandle, { ok: true }, null, undefined);

  const llmRequest = { headers: {}, content: { model: 'demo-model', messages: [] } };
  const llmHandle = wasm.llmCall(
    'optional_llm',
    llmRequest,
    currentScope(),
    undefined,
    null,
    undefined,
  );
  assert.equal(llmHandle.name, 'optional_llm');
  assert.equal(typeof llmHandle.uuid, 'string');
  assert.equal(llmHandle.attributes, 0);
  assert.equal(llmHandle.parentUuid, scope.uuid);
  wasm.llmCallEnd(llmHandle, { role: 'assistant', content: 'done', tool_calls: [] }, null, undefined);

  const llmResult = await wasm.llmCallExecute(
    'optional_llm_exec',
    llmRequest,
    async () => ({ role: 'assistant', content: 'ok', tool_calls: [] }),
  );
  assert.equal(llmResult.content, 'ok');

  const stream = await wasm.llmStreamCallExecute(
    'optional_llm_stream',
    llmRequest,
    async () => [{ delta: 'solo' }],
  );
  assert.deepEqual(await drainStream(stream), [[{ delta: 'solo' }]]);

  const exporter = new wasm.WasmAtifExporter('session-js', 'wasm-js', '1.0.0', null);
  assert.equal(typeof JSON.parse(exporter.export_json()).schema_version, 'string');

  expectInvalidUuid(
    () => wasm.scopeRegisterToolRequestIntercept('not-a-uuid', unique('bad_scope'), 1, false, (name, args) => args),
  );
  expectInvalidUuid(
    () => wasm.scopeRegisterSubscriber('not-a-uuid', unique('bad_subscriber'), () => {}),
  );
  expectInvalidLlmRequest(
    () => wasm.llmRequestIntercepts('bad_llm', { headers: [], content: 'bad' }),
  );
  expectInvalidLlmRequest(
    () => wasm.llmConditionalExecution({ headers: [], content: 'bad' }),
  );
  await rejectInvalidLlmRequest(
    wasm.llmCallExecute('bad_exec', { headers: [], content: 'bad' }, async () => ({ role: 'assistant' })),
  );
  await rejectInvalidLlmRequest(
    wasm.llmStreamCallExecute('bad_stream', { headers: [], content: 'bad' }, async () => []),
  );

  const duplicateName = unique('dup_tool_req');
  wasm.registerToolRequestIntercept(duplicateName, 1, false, (name, args) => args);
  assert.throws(
    () => wasm.registerToolRequestIntercept(duplicateName, 1, false, (name, args) => args),
    /already exists/i,
  );
  wasm.deregisterToolRequestIntercept(duplicateName);

  wasm.event('pkg_parent_mark', currentScope(), { branch: true }, null);

  request.free();
  exporter.free();
  toolHandle.free();
  llmHandle.free();
  stream.free();
  wasm.popScope(scope);
  scope.free();
  stack.free();
});

test('WASM JS wrappers reject wrong handle classes and support async scope callbacks', async () => {
  const stack = resetScopeStack();
  const scope = wasm.pushScope('assert_scope', 1, null, 0, null, null);
  const llmHandle = wasm.llmCall(
    'assert_llm',
    { headers: {}, content: { model: 'demo-model', messages: [] } },
    null,
    0,
    null,
    null,
  );
  const toolHandle = wasm.toolCall('assert_tool', { ok: true }, null, 0, null, null);

  expectClassError(() => wasm.setThreadScopeStack({}));
  expectClassError(() => wasm.popScope({}));
  expectClassError(() => wasm.event('bad_parent', {}, null, null));
  expectClassError(() => wasm.pushScope('bad_push', 1, {}, 0, null, null));
  expectClassError(() => wasm.toolCall('bad_tool', {}, {}, 0, null, null));
  expectClassError(() => wasm.toolCallEnd({}, {}, null, null));
  expectClassError(() => wasm.llmCall('bad_llm', { headers: {}, content: {} }, {}, 0, null, null));
  expectClassError(() => wasm.llmCallEnd({}, { role: 'assistant', content: 'nope', tool_calls: [] }, null, null));
  expectClassError(() => wasm.withScope('bad_scope', 1, () => null, {}, 0, null, null));

  const asyncResult = await wasm.withScope(
    'async_scope',
    1,
    async (handle) => ({ uuid: handle.uuid, type: handle.scopeType }),
    null,
    0,
    null,
    null,
  );
  assert.equal(asyncResult.type, 1);
  assert.equal(typeof asyncResult.uuid, 'string');

  llmHandle.free();
  toolHandle.free();
  wasm.popScope(scope);
  scope.free();
  stack.free();
});

test('global and scope-local register/deregister wrappers are callable', () => {
  const stack = resetScopeStack();
  const scope = wasm.pushScope('registration_scope', 1, null, 0, null, null);

  for (const [prefix, register, deregister, invoke] of globalRegistrationCases()) {
    const name = unique(prefix);
    invoke(name, register);
    assert.equal(deregister(name), true, `${prefix} should deregister`);
    assert.equal(deregister(name), false, `${prefix} should not deregister twice`);
  }

  for (const [prefix, register, deregister, invoke] of scopeRegistrationCases(scope.uuid)) {
    const name = unique(prefix);
    invoke(scope.uuid, name, register);
    assert.equal(deregister(scope.uuid, name), true, `${prefix} should deregister`);
    assert.equal(deregister(scope.uuid, name), false, `${prefix} should not deregister twice`);
  }

  wasm.popScope(scope);
  scope.free();
  stack.free();
});

test('WASM JS registration wrappers cover duplicate and invalid UUID errors', () => {
  const stack = resetScopeStack();
  const scope = wasm.pushScope('registration_errors_scope', 1, null, 0, null, null);

  for (const [prefix, register, deregister, invoke] of globalRegistrationCases()) {
    const name = unique(`${prefix}_dup`);
    invoke(name, register);
    expectAlreadyExists(() => invoke(name, register));
    assert.equal(deregister(name), true, `${prefix} duplicate registration should clean up`);
  }

  for (const [prefix, register, deregister, invoke] of scopeRegistrationCases(scope.uuid)) {
    const name = unique(`${prefix}_scope`);
    expectInvalidUuid(() => invoke('not-a-uuid', name, register));
    expectInvalidUuid(() => deregister('not-a-uuid', name));
    invoke(scope.uuid, name, register);
    expectAlreadyExists(() => invoke(scope.uuid, name, register));
    assert.equal(deregister(scope.uuid, name), true, `${prefix} duplicate scope registration should clean up`);
  }

  wasm.popScope(scope);
  scope.free();
  stack.free();
});

test('tool, llm, stream, and exporter flows work from the generated Node package', async () => {
  const stack = resetScopeStack();
  const events = [];
  const subscriberName = unique('event_subscriber');
  wasm.registerSubscriber(subscriberName, (event) => events.push(event));

  const exporter = new wasm.WasmAtifExporter('session-js', 'wasm-js', '1.0.0', 'demo-model');
  const exporterName = unique('exporter');
  exporter.register(exporterName);

  const toolInterceptName = unique('tool_req');
  wasm.registerToolRequestIntercept(toolInterceptName, 1, false, (name, args) => ({
    ...args,
    intercepted: true,
  }));
  assert.deepEqual(
    wasm.toolRequestIntercepts('pkg_tool', { value: 1 }),
    { value: 1, intercepted: true },
  );
  wasm.toolConditionalExecution('pkg_tool', { value: 1 });

  const toolHandle = wasm.toolCall('pkg_tool', { value: 1 }, null, 1, { phase: 'start' }, { source: 'js' }, 'tool-123');
  assert.equal(toolHandle.name, 'pkg_tool');
  wasm.toolCallEnd(toolHandle, { ok: true }, { phase: 'end' }, { source: 'js' });

  const toolResult = await wasm.toolCallExecute(
    'pkg_tool_exec',
    { value: 3 },
    async (args) => ({ ...args, executed: true }),
    null,
    1,
    { from: 'tool' },
    { layer: 'js' },
  );
  assert.equal(toolResult.intercepted, true);
  assert.equal(toolResult.executed, true);

  const request = { headers: { trace: '1' }, content: { model: 'demo-model', messages: [] } };
  const llmInterceptName = unique('llm_req');
  wasm.registerLlmRequestIntercept(llmInterceptName, 1, false, (name, nextRequest, annotated) => ({
    request: {
      ...nextRequest,
      content: { ...nextRequest.content, intercepted: true },
    },
    annotated: annotated,
  }));
  const interceptedRequest = wasm.llmRequestIntercepts('pkg_llm', request);
  assert.equal(interceptedRequest.content.model, 'demo-model');
  wasm.llmConditionalExecution(request);

  const llmHandle = wasm.llmCall('pkg_llm', request, null, 1, { phase: 'start' }, { source: 'js' }, 'demo-model');
  assert.equal(llmHandle.name, 'pkg_llm');
  wasm.llmCallEnd(llmHandle, { role: 'assistant', content: 'done', tool_calls: [] }, { phase: 'end' }, { source: 'js' });

  const llmResult = await wasm.llmCallExecute(
    'pkg_llm_exec',
    request,
    async (nextRequest) => ({
      role: 'assistant',
      content: `hello ${nextRequest.content.model}`,
      tool_calls: [],
    }),
    null,
    1,
    { from: 'llm' },
    { layer: 'js' },
    'demo-model',
  );
  assert.equal(llmResult.role, 'assistant');
  assert.equal(llmResult.content, 'hello demo-model');

  const collected = [];
  let finalized = false;
  const stream = await wasm.llmStreamCallExecute(
    'pkg_llm_stream',
    request,
    async () => [
      { delta: 'hello' },
      { delta: 'world' },
    ],
    (chunk) => collected.push(chunk),
    () => {
      finalized = true;
      return { combined: true };
    },
    null,
    2,
    { from: 'stream' },
    { layer: 'js' },
    'demo-model',
  );
  const chunks = await drainStream(stream);
  assert.deepEqual(chunks, [[{ delta: 'hello' }, { delta: 'world' }]]);
  assert.deepEqual(collected, chunks);
  assert.equal(finalized, true);

  wasm.event('pkg_mark', null, { mark: true }, { source: 'js' });
  assert.ok(events.some((event) => event.name === 'pkg_mark'));
  assert.ok(events.some((event) => event.name === 'pkg_tool_exec'));
  assert.ok(events.some((event) => event.name === 'pkg_llm_exec'));

  const exported = JSON.parse(exporter.export_json());
  assert.equal(typeof exported.schema_version, 'string');
  assert.ok(exported.schema_version.length > 0);
  assert.ok(Array.isArray(exported.steps));
  assert.ok(exported.steps.length > 0);

  assert.equal(exporter.deregister(exporterName), true);
  exporter.clear();
  const afterClear = JSON.parse(exporter.export_json());
  assert.deepEqual(afterClear.steps, []);

  wasm.deregisterToolRequestIntercept(toolInterceptName);
  wasm.deregisterLlmRequestIntercept(llmInterceptName);
  wasm.deregisterSubscriber(subscriberName);
  exporter.free();
  toolHandle.free();
  llmHandle.free();
  stream.free();
  stack.free();
});
