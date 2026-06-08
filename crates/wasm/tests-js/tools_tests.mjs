// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { currentScope, resetScopeStack, unique, waitFor, wasm } from './test_support.mjs';

test('WebAssembly tool handles preserve nullable inputs and handle properties', () => {
  const stack = resetScopeStack();
  const scope = wasm.pushScope('tool_scope', wasm.ScopeType.Function, null, 0, null, null);
  let toolHandle;

  try {
    toolHandle = wasm.toolCall(
      'optional_tool',
      {
        value: 2,
      },
      currentScope(),
      undefined,
      null,
      undefined,
    );
    assert.equal(toolHandle.name, 'optional_tool');
    assert.equal(typeof toolHandle.uuid, 'string');
    assert.equal(toolHandle.attributes, 0);
    assert.equal(toolHandle.parentUuid, scope.uuid);
    wasm.toolCallEnd(
      toolHandle,
      {
        ok: true,
      },
      null,
      undefined,
    );
  } finally {
    if (toolHandle) {
      toolHandle.free();
    }
    wasm.popScope(scope);
    scope.free();
    stack.free();
  }
});

test('WebAssembly tool execute returns null when the JS callback yields undefined', async () => {
  const toolNullResult = await wasm.toolCallExecute(
    'optional_tool_exec_null',
    {
      value: 9,
    },
    async () => undefined,
  );
  assert.equal(toolNullResult, null);
});

test('WebAssembly tool request intercepts modify arguments in the generated package flow', () => {
  const toolInterceptName = unique('tool_req');

  wasm.registerToolRequestIntercept(toolInterceptName, 1, false, (_name, args) => ({
    ...args,
    intercepted: true,
  }));

  try {
    assert.deepEqual(
      wasm.toolRequestIntercepts('pkg_tool', {
        value: 1,
      }),
      {
        value: 1,
        intercepted: true,
      },
    );
    wasm.toolConditionalExecution('pkg_tool', {
      value: 1,
    });
  } finally {
    wasm.deregisterToolRequestIntercept(toolInterceptName);
  }
});

test('WebAssembly tool lifecycle flows emit events from the generated Node package', async () => {
  const stack = resetScopeStack();
  const events = [];
  const subscriberName = unique('event_subscriber');

  wasm.registerSubscriber(subscriberName, (event) => events.push(event));

  let toolHandle;
  try {
    toolHandle = wasm.toolCall(
      'pkg_tool',
      {
        value: 1,
      },
      null,
      1,
      {
        phase: 'start',
      },
      {
        source: 'js',
      },
      'tool-123',
    );
    assert.equal(toolHandle.name, 'pkg_tool');
    wasm.toolCallEnd(
      toolHandle,
      {
        ok: true,
      },
      {
        phase: 'end',
      },
      {
        source: 'js',
      },
    );
    await waitFor(() => events.filter((event) => event.name === 'pkg_tool').length >= 2);
  } finally {
    wasm.deregisterSubscriber(subscriberName);
    if (toolHandle) {
      toolHandle.free();
    }
    stack.free();
  }
});

test('WebAssembly tool execute runs through the generated Node package flow', async () => {
  const toolInterceptName = unique('tool_req_exec');

  wasm.registerToolRequestIntercept(toolInterceptName, 1, false, (_name, args) => ({
    ...args,
    intercepted: true,
  }));

  try {
    const toolResult = await wasm.toolCallExecute(
      'pkg_tool_exec',
      {
        value: 3,
      },
      async (args) => ({
        ...args,
        executed: true,
      }),
      null,
      1,
      {
        from: 'tool',
      },
      {
        layer: 'js',
      },
    );
    assert.equal(toolResult.intercepted, true);
    assert.equal(toolResult.executed, true);
  } finally {
    wasm.deregisterToolRequestIntercept(toolInterceptName);
  }
});

test('WebAssembly toolCallExecute adds OTEL status metadata to end events', async () => {
  const stack = resetScopeStack();
  const events = [];
  const subscriberName = unique('wasm_tool_status_sub');

  wasm.registerSubscriber(subscriberName, (event) => events.push(event));

  try {
    await wasm.toolCallExecute(
      'wasm_tool_status_ok',
      {
        value: 1,
      },
      async () => ({
        ok: true,
      }),
      null,
      0,
      null,
      {
        caller: 'wasm-tool',
        'otel.status_code': 'USER',
      },
    );

    await assert.rejects(
      () =>
        wasm.toolCallExecute(
          'wasm_tool_status_error',
          {
            value: 2,
          },
          async () => {
            throw new Error('wasm tool failure');
          },
          null,
          0,
          null,
          {
            caller: 'wasm-tool-error',
          },
        ),
      /wasm tool failure/,
    );

    const okEvent = await waitFor(() =>
      events.find(
        (event) => event.kind === 'scope' && event.scope_category === 'end' && event.name === 'wasm_tool_status_ok',
      ),
    );
    const errorEvent = await waitFor(() =>
      events.find(
        (event) => event.kind === 'scope' && event.scope_category === 'end' && event.name === 'wasm_tool_status_error',
      ),
    );

    assert.equal(okEvent.metadata.caller, 'wasm-tool');
    assert.equal(okEvent.metadata['otel.status_code'], 'OK');
    assert.equal(errorEvent.metadata.caller, 'wasm-tool-error');
    assert.equal(errorEvent.metadata['otel.status_code'], 'ERROR');
    assert.match(errorEvent.metadata['otel.status_description'], /wasm tool failure/);
  } finally {
    wasm.deregisterSubscriber(subscriberName);
    stack.free();
  }
});
