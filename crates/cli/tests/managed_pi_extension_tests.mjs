// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  decideManagedProviderRedirect,
  decideManagedToolTransform,
  summarizeManagedToolResult,
} from '../src/daemon/managed/pi_extension/index.ts';

test('custom Pi providers redirect only when every sibling API is supported', () => {
  const selected = {
    id: 'custom-response-model',
    api: 'openai-responses',
    provider: 'custom-enterprise',
    baseUrl: 'https://ignored-by-managed-policy.example/v1',
  };
  const serviceableCatalog = [
    selected,
    {
      id: 'custom-messages-model',
      api: 'anthropic-messages',
      provider: selected.provider,
      baseUrl: 'https://ignored-by-managed-policy.example/v1/',
    },
  ];

  assert.deepEqual(decideManagedProviderRedirect(selected, serviceableCatalog), {
    kind: 'redirect',
    upstream: selected.baseUrl,
    reason: 'provider uses only daemon-supported APIs and every model shares its endpoint',
  });
  assert.equal(decideManagedProviderRedirect(selected, undefined).kind, 'skip');
  assert.deepEqual(
    decideManagedProviderRedirect(selected, [
      ...serviceableCatalog,
      {
        id: 'custom-google-model',
        api: 'google-generative-ai',
        provider: selected.provider,
        baseUrl: 'https://google.example',
      },
    ]),
    {
      kind: 'skip',
      code: 'provider-mixed-apis',
      reason:
        'redirecting custom-enterprise would also move its unsupported ' +
        'google-generative-ai model custom-google-model',
    },
  );
  assert.deepEqual(
    decideManagedProviderRedirect(selected, [
      selected,
      {
        id: 'different-endpoint-model',
        api: 'openai-completions',
        provider: selected.provider,
        baseUrl: 'https://different.example/v1',
      },
    ]),
    {
      kind: 'skip',
      code: 'provider-mixed-endpoints',
      reason:
        'redirecting custom-enterprise would also move different-endpoint-model, which targets ' +
        'https://different.example/v1 rather than https://ignored-by-managed-policy.example/v1',
    },
  );
});

test('managed Pi tool rewrites require the exact call ID and recursively preserve shape', () => {
  const current = { path: '/before', flags: [true, { retries: 2 }] };
  assert.deepEqual(
    decideManagedToolTransform(
      {
        tool_call: {
          tool_call_id: 'call-1',
          input: { path: '/after', flags: [false, { retries: 3 }] },
        },
      },
      'call-1',
      current,
    ),
    {
      kind: 'replace',
      input: { path: '/after', flags: [false, { retries: 3 }] },
    },
  );
  assert.equal(
    decideManagedToolTransform({ tool_call: { input: { path: '/after', flags: current.flags } } }, 'call-1', current)
      .kind,
    'invalid',
  );
  assert.equal(
    decideManagedToolTransform(
      {
        tool_call: {
          tool_call_id: 'call-1',
          input: { path: '/after', flags: [false, { retries: 3 }], extra: true },
        },
      },
      'call-1',
      current,
    ).kind,
    'invalid',
  );
});

test('managed Pi summaries preserve ordered text blocks and Unicode boundaries', () => {
  assert.deepEqual(
    summarizeManagedToolResult(
      {
        content: [
          { type: 'text', text: 'first' },
          { type: 'image', data: 'not-forwarded' },
          { type: 'text', text: 'second' },
        ],
      },
      false,
    ),
    { content: 'first\nsecond', result_keys: ['content'] },
  );

  const summary = summarizeManagedToolResult('x'.repeat(1_999) + '😀tail', false).content;
  assert.equal(typeof summary, 'string');
  assert.equal(summary.includes('�'), false);
  assert.equal(summary.includes('... [truncated 6 chars]'), true);
});
