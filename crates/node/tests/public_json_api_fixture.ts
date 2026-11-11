// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  createScopeStack,
  LlmRequest,
  llmCall,
  registerSubscriber,
  ScopeType,
  toolCall,
  withScope,
  withScopeStack,
  type Json,
  type JsonValue,
  type LlmRequestShape,
  type ScopeHandle,
} from '../dist/index.js';
import type { JsonValue as TypedJsonValue } from '../dist/typed.js';

type AssertEqual<T, U> = (<V>() => V extends T ? 1 : 2) extends <V>() => V extends U ? 1 : 2 ? true : false;

const jsonIsJsonValue: AssertEqual<Json, JsonValue> = true;
const typedSharesJson: AssertEqual<JsonValue, TypedJsonValue> = true;
const requestShapeIsJson: AssertEqual<LlmRequestShape extends Json ? true : false, true> = true;

const request: LlmRequestShape = new LlmRequest({ 'x-tenant': 'a' }, { messages: [] });
const requestAsJson: Json = request;
llmCall('provider', request, null, null, null, null, null);
llmCall('provider', { headers: {}, content: null });
toolCall('search', { query: 'weather' });
registerSubscriber('typed', (event) => {
  const name: JsonValue = event.name;
  void name;
});

const scoped: Promise<{ ok: boolean }> = withScope('run', ScopeType.Agent, (handle: ScopeHandle) => ({
  ok: handle.name === 'run',
}));
const voided: Promise<null> = withScope('run', ScopeType.Agent, async () => {});
const passthrough: Promise<string> = withScopeStack(createScopeStack(), async () => 'done');

// @ts-expect-error Request headers must be a JSON object.
new LlmRequest('headers', null);

// @ts-expect-error A request needs headers and content.
llmCall('provider', { content: {} });

// @ts-expect-error Tool arguments must be JSON.
toolCall('search', () => 'weather');

// @ts-expect-error A scope callback must return JSON.
withScope('run', ScopeType.Agent, () => new Map());

export const checked = [
  jsonIsJsonValue,
  typedSharesJson,
  requestShapeIsJson,
  requestAsJson,
  scoped,
  voided,
  passthrough,
];
