// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  LlmRequest,
  llmCall,
  registerSubscriber,
  toolCall,
  type Json,
  type JsonValue,
  type LlmRequestShape,
} from '../index.js';
import type { JsonValue as TypedJsonValue } from '../typed.js';

type AssertEqual<T, U> = [T] extends [U] ? ([U] extends [T] ? true : false) : false;

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

// @ts-expect-error Request headers must be a JSON object.
new LlmRequest('headers', null);

// @ts-expect-error A request needs headers and content.
llmCall('provider', { content: {} });

// @ts-expect-error Tool arguments must be JSON.
toolCall('search', () => 'weather');

export const checked = [jsonIsJsonValue, typedSharesJson, requestShapeIsJson, requestAsJson];
