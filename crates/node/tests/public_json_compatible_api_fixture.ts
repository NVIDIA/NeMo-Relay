// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  event,
  info,
  llmCall,
  llmCallEnd,
  llmCallExecuteAsync,
  llmStreamCallExecute,
  LlmRequest,
  metric,
  MetricKind,
  MetricValueType,
  OpenAIChatCodec,
  registerEventMetadataInjector,
  registerLlmExecutionIntercept,
  registerLlmRequestIntercept,
  registerLlmSanitizeRequestGuardrail,
  registerLlmSanitizeResponseGuardrail,
  registerMarkSanitizeGuardrail,
  registerToolExecutionIntercept,
  registerToolRequestIntercept,
  ScopeType,
  toolCall,
  toolCallEnd,
  toolCallExecuteAsync,
  withScope,
  type EventMetadata,
  type EventSanitizeFields,
  type EventSanitizeFieldsCompatible,
  type Json,
  type JsonCompatible,
  type JsonObject,
  type JsonValue,
  type LlmRequestCompatible,
  type LlmRequestInterceptOutcome,
  type LlmRequestShape,
  type MetricMeasurement,
  type ToolExecutionInterceptOutcome,
  type ToolExecutionResult,
  type ToolExecutionResultCompatible,
  type ToolHandle,
} from '../dist/index.js';
import * as plugin from '../dist/plugin.js';
import { typedLlmExecute, typedToolExecute, type LlmCodec, type LlmResponseCodec } from '../dist/typed.js';

type AssertEqual<T, U> = (<V>() => V extends T ? 1 : 2) extends <V>() => V extends U ? 1 : 2 ? true : false;

interface SdkParams {
  model: string;
  messages: { role: string; content: string }[];
  temperature?: number;
  user?: string | null;
  nested?: { deep: { list: readonly number[] } };
}
interface SdkResponse {
  id: string;
  choices: { index: number; message: { role: string; content: string | null } }[];
  usage?: { total_tokens?: number };
}
interface Options {
  retries?: number;
  nested?: { limit?: number };
}
interface ToolDefinition {
  name: string;
  parameters?: Record<string, unknown>;
}
interface InputSchema {
  type: 'object';
  properties?: unknown | null;
  [key: string]: unknown;
}
interface SdkParamsWithTools extends SdkParams {
  tools?: ToolDefinition[];
  schema?: InputSchema;
}
interface SdkChunk {
  id: string;
  delta: { content?: string };
}
interface Headers {
  authorization: string;
}
interface Tree {
  value: string;
  children: Tree[];
}
interface Meta {
  stage: string;
  attempt?: number;
  tags: readonly string[];
}
interface UndefinedMeta {
  stage: string | undefined;
}
interface MarkData {
  reason: string;
}
interface PluginSettings {
  tag: string;
  limits?: { max: number };
}
interface WithMethod {
  a: string;
  run(): void;
}
interface WithUndefined {
  a: string | undefined;
}
interface BadMeta {
  stage: string;
  nested: { a: 1 };
}
class Thing {
  constructor(public a: string) {}
  describe(): string {
    return this.a;
  }
}

declare const params: SdkParams;
declare const headers: Headers;
declare const json: Json;
declare const nullableJson: Json | null;
declare const record: JsonObject;
declare const shape: LlmRequestShape;
declare const nullableShape: LlmRequestShape | null;
declare const toolResult: ToolExecutionResult;
declare const outcome: ToolExecutionInterceptOutcome;
declare const requestOutcome: LlmRequestInterceptOutcome;
declare const fields: EventSanitizeFields;
declare const measurements: MetricMeasurement[];
declare const markData: MarkData;
declare const sdk: SdkResponse;
declare const sdkOrNull: SdkResponse | null;
declare const mixed: SdkResponse | string;
declare const tree: Tree;
declare const tuple: readonly [string, number, { ok: boolean }];
declare const toolHandle: ToolHandle;
declare const settings: PluginSettings;
declare const options: Options;
declare const paramsWithTools: SdkParamsWithTools;
declare const unknownValue: unknown;
declare const date: Date;
declare const big: bigint;
declare const fnOrObject: SdkResponse | (() => void);
declare const bareObject: object;
declare const nullableObject: object | null;
declare const emptyObject: {};
declare const anyValue: any;
declare const unknownRecord: Record<string, unknown>;
declare const meta: Meta;
declare const badMeta: BadMeta;
declare const undefinedMeta: UndefinedMeta;
declare const withMethod: WithMethod;
declare const withUndefined: WithUndefined;
declare const paramsOrList: SdkParams | string[];
declare function parseSdkParams(data: Json): SdkParams;
declare function parseSdkResponse(data: Json): SdkResponse;
async function* chunks(): AsyncIterable<SdkChunk> {
  yield { id: '1', delta: { content: 'x' } };
}

const jsonRoundTrip: AssertEqual<JsonCompatible<JsonValue>, JsonValue> = true;
const shapeRoundTrip: AssertEqual<
  LlmRequestShape extends LlmRequestCompatible<LlmRequestShape> ? true : false,
  true
> = true;

toolCall('interface', params);
toolCall('json', json, null, null, record, json);
toolCall('nullable', params, null, null, null, nullableJson);
toolCall('recursive', tree);
toolCall('tuple', tuple);
toolCall('literal', { a: 1, b: [true, 'x', null], c: { d: 2 } });
toolCall('scalar', 'scalar');
toolCall('array', [params, params]);
toolCall('union', sdkOrNull, null, null, mixed);
toolCall('empty', {});
toolCall('optional', options, null, null, options);
llmCall('optional', { headers: options, content: options });
llmCall('tools', { headers: {}, content: paramsWithTools });
llmCall('tools-instance', new LlmRequest({}, paramsWithTools));
toolCall('unknown-record', unknownRecord);
llmCallExecuteAsync('unknown-record', shape, async () => unknownRecord);
toolCall('empty-typed', emptyObject);
toolCall('any', anyValue);
llmCallExecuteAsync('any', shape, async () => anyValue);
toolCallEnd(toolHandle, toolResult);
toolCallEnd(toolHandle, { result: sdk }, markData);
toolCallExecuteAsync('typed', params, async () => toolResult);
toolCallExecuteAsync('interface', params, async () => ({ result: sdk }));
toolCallExecuteAsync('args', params, async (args) => ({ result: args, annotation: { note: 1 } }));
toolCallExecuteAsync('extra-json', params, async () => ({ result: 1, extra: 2 }));
llmCall('object', { headers: {}, content: params });
llmCall('headers', { headers, content: null });
llmCall('instance', new LlmRequest({}, { x: 1 }));
llmCall('instance-typed', new LlmRequest(headers, params));
const built = new LlmRequest({}, params);
llmCall('built', built);
info('fields', null, settings);
info('json-fields', null, record);
llmCall('extra-json', { headers: {}, content: null, extra: 1 });
llmCall('shape', shape, null, null, markData, markData);
llmCallEnd(llmCall('end', shape), sdk, markData);
const executed: Promise<Json> = llmCallExecuteAsync(
  'async',
  { headers: {}, content: params },
  async (): Promise<SdkResponse> => ({ id: '1', choices: [] }),
);
llmCallExecuteAsync('sync', shape, (): SdkResponse => ({ id: '1', choices: [] }));
llmCallExecuteAsync('json', shape, () => json);
llmCallExecuteAsync('nullable', shape, () => nullableJson);
llmCallExecuteAsync('mixed', shape, () => mixed);
llmCallExecuteAsync('passthrough', shape, async (request) => request.content);
llmCallExecuteAsync(
  'codecs',
  shape,
  async () => sdk,
  null,
  null,
  markData,
  null,
  null,
  (request) => request.content,
  ({ original }) => ({ headers: original.headers, content: params }),
  (response) => response,
);
llmStreamCallExecute(
  'stream',
  { headers: {}, content: params },
  () => chunks(),
  undefined,
  (): SdkResponse => sdk,
);
llmStreamCallExecute(
  'stream-json',
  shape,
  async () => chunks(),
  (chunk) => void chunk,
  async () => json,
);
registerToolExecutionIntercept('outcome', 1, async (context, next) => ({
  result: await next(params),
  pendingMarks: [{ name: 'm', data: markData }],
}));
registerToolExecutionIntercept('typed', 1, async () => outcome);
registerToolExecutionIntercept('next', 1, (context, next) => next(json));
registerToolRequestIntercept('args', 1, false, () => params);
registerLlmRequestIntercept('request', 1, false, ({ request }) => ({ request, annotated: null }));
registerLlmRequestIntercept('interface', 1, false, ({ request }) => ({
  request: { headers: request.headers, content: params },
  annotated: sdk,
  pendingMarks: [{ name: 'm' }],
}));
registerLlmRequestIntercept('typed', 1, false, async () => requestOutcome);
registerLlmExecutionIntercept('next', 1, (request, context, next) =>
  next({ headers: request.headers, content: params }),
);
registerLlmExecutionIntercept('interface', 1, async () => sdk);
registerLlmSanitizeRequestGuardrail('shape', 1, () => shape);
registerLlmSanitizeRequestGuardrail('nullable', 1, async () => nullableShape);
registerLlmSanitizeRequestGuardrail('null', 1, () => null);
registerLlmSanitizeRequestGuardrail('interface', 1, (request) => ({ headers: request.headers, content: params }));
registerLlmSanitizeResponseGuardrail('interface', 1, () => sdkOrNull);
registerLlmSanitizeResponseGuardrail('null', 1, () => null);
registerMarkSanitizeGuardrail('interface', 1, () => ({ data: markData }));
registerMarkSanitizeGuardrail('passthrough', 1, (e, current) => current);
registerMarkSanitizeGuardrail('typed', 1, () => fields);
registerMarkSanitizeGuardrail('extra-json', 1, () => ({ data: markData, payload: 1 }));
registerEventMetadataInjector('interface', 1, () => meta);
event('data', null, markData, markData);
metric('typed', measurements);
metric('extra', [{ name: 'm', kind: MetricKind.Counter, valueType: MetricValueType.U64, value: 1, extra: () => 1 }]);
metric(
  'interface',
  [{ name: 'm', kind: MetricKind.Counter, valueType: MetricValueType.U64, value: 1, attributes: markData }],
  null,
  markData,
);
const scoped: Promise<SdkResponse> = withScope('scope', ScopeType.Agent, async (): Promise<SdkResponse> => sdk);
const voided: Promise<null> = withScope('void', ScopeType.Agent, async () => {});
const fromJson: Promise<Json> = withScope('json', ScopeType.Agent, () => json, null, null, markData, markData, params);
new OpenAIChatCodec().decode({ headers, content: params });
new OpenAIChatCodec().encode(sdk, { headers, content: params });
new OpenAIChatCodec().decodeResponse(sdk);
const codec: LlmCodec<SdkParams, { headers: Headers; content: SdkParams }> = {
  decode: () => params,
  encode: () => ({ headers, content: params }),
};
const responseCodec: LlmResponseCodec<SdkResponse> = { decodeResponse: () => sdk };
// @ts-expect-error a codec may only produce JSON
type BadCodec = LlmCodec<Map<string, string>>;
const typedResponse: Promise<SdkResponse> = typedLlmExecute(
  'typed',
  { headers, content: params },
  async (request): Promise<SdkResponse> => ({ id: JSON.stringify(request.content), choices: [] }),
  { toJson: (value) => value, fromJson: parseSdkResponse },
  { codec, responseCodec, data: markData },
);
const typedTool: Promise<{ result: SdkResponse }> = typedToolExecute(
  'typed',
  params,
  async () => ({ result: sdk }),
  { toJson: (value) => value, fromJson: parseSdkParams },
  { toJson: (value) => value, fromJson: parseSdkResponse },
  { metadata: markData },
);
const typedToolResult = typedToolExecute(
  'typed',
  params,
  async () => ({ result: sdk }),
  { toJson: (value) => value, fromJson: parseSdkParams },
  // @ts-expect-error a result codec may only produce JSON
  { toJson: (value): Map<string, string> => new Map([['id', value.id]]), fromJson: parseSdkResponse },
);
const open = plugin.ComponentSpec('example.add-tag', settings);
function forwardJson<T extends Json>(value: T): void {
  toolCall('forward', value);
  event('forward', null, value, value);
}
function forwardObject<T extends JsonObject>(value: T): void {
  plugin.ComponentSpec('example.add-tag', value);
}
function forwardRequest<T extends LlmRequestShape>(value: T): void {
  llmCall('forward', value);
}
function forwardResult<T extends ToolExecutionResultCompatible<T>>(value: T): void {
  toolCallEnd(toolHandle, value);
}
function forwardOutcome<T extends ToolExecutionInterceptOutcome>(value: T): void {
  registerToolExecutionIntercept('forward', 1, () => value);
}
function forwardRequestOutcome<T extends LlmRequestInterceptOutcome>(value: T): void {
  registerLlmRequestIntercept('forward', 1, false, () => value);
}
function forwardFields<T extends EventSanitizeFieldsCompatible<T>>(value: T): void {
  registerMarkSanitizeGuardrail('forward', 1, () => value);
}
function forwardMetadata<T extends EventMetadata>(value: T): void {
  registerEventMetadataInjector('forward', 1, () => value);
}
function forwardMeasurements<T extends MetricMeasurement[]>(value: T): void {
  metric('forward', value);
}
plugin.ComponentSpec('example.add-tag', {});
plugin.ComponentSpec('example.add-tag', sdkOrNull === null ? settings : sdk);
const openKind: AssertEqual<typeof open.kind, string> = true;

// @ts-expect-error methods are not JSON
toolCall('method', withMethod);
// @ts-expect-error explicit undefined is not JSON
toolCall('undefined', withUndefined);
// @ts-expect-error unknown is not JSON
toolCall('unknown', unknownValue);
// @ts-expect-error Date instances are not JSON
toolCall('date', date);
// @ts-expect-error bigint is not JSON
toolCall('bigint', big);
// @ts-expect-error class instances with methods are not JSON
toolCall('class', new Thing('a'));
// @ts-expect-error functions are not JSON
toolCall('function', () => 1);
// @ts-expect-error a union with a function is not JSON
toolCall('function-union', fnOrObject);
// @ts-expect-error object carries no JSON guarantee
toolCall('object', bareObject);
// @ts-expect-error object carries no JSON guarantee inside a union
toolCall('object-union', nullableObject);
// @ts-expect-error the LLM implementation must return JSON
llmCallExecuteAsync('object', shape, async () => bareObject);
// @ts-expect-error a request needs content
llmCall('content', { headers: {} });
// @ts-expect-error extra request keys must be JSON
llmCall('extra-function', { headers: {}, content: null, extra: () => 1 });
// @ts-expect-error headers must be an object
llmCall('headers-string', { headers: 'x', content: null });
// @ts-expect-error headers must not be an array
llmCall('headers-array', { headers: [], content: null });
// @ts-expect-error the LLM implementation must return JSON
llmCallExecuteAsync('map', shape, async () => new Map());
// @ts-expect-error the LLM implementation must return JSON
llmCallExecuteAsync('unknown', shape, async () => unknownValue);
// @ts-expect-error a tool result needs result
toolCallExecuteAsync('results', params, async () => ({ results: 1 }));
// @ts-expect-error extra tool result keys must be JSON
toolCallExecuteAsync('extra-function', params, async () => ({ result: 1, extra: () => 1 }));
// @ts-expect-error extra tool result keys must be JSON
toolCallEnd(toolHandle, { result: sdk, extra: () => 1 });
// @ts-expect-error explicit undefined is rejected by the runtime
toolCallEnd(toolHandle, { result: sdk, annotation: undefined });
// @ts-expect-error event metadata must be flat scalars
registerEventMetadataInjector('nested', 1, () => badMeta);
// @ts-expect-error explicit undefined is rejected by the runtime
registerEventMetadataInjector('undefined', 1, () => undefinedMeta);
// @ts-expect-error pendingMarks must be mark specs
registerToolExecutionIntercept('marks', 1, async (context, next) => ({ result: await next(json), pendingMarks: [1] }));
// @ts-expect-error a mark spec needs a name
registerToolExecutionIntercept('mark-name', 1, async (context, next) => ({
  result: await next(json),
  pendingMarks: [{ data: markData }],
}));
// @ts-expect-error a request intercept must return a request
registerLlmRequestIntercept('missing', 1, false, () => ({ annotated: null }));
// @ts-expect-error a request intercept returns JSON only
registerLlmRequestIntercept('map', 1, false, ({ request }) => ({ request, annotated: new Map() }));
// @ts-expect-error extra sanitize fields must be JSON
registerMarkSanitizeGuardrail('payload', 1, () => ({ payload: () => 1 }));
// @ts-expect-error a function is not sanitize fields
registerMarkSanitizeGuardrail('function', 1, () => () => 1);
// @ts-expect-error a function is not event metadata
registerEventMetadataInjector('function', 1, () => () => 1);
// @ts-expect-error sanitize fields must be JSON
registerMarkSanitizeGuardrail('map', 1, () => ({ data: new Map() }));
// @ts-expect-error a sanitized request must keep the request shape
registerLlmSanitizeRequestGuardrail('shape', 1, () => ({ headers: {} }));
// @ts-expect-error a sanitized request must be JSON
registerLlmSanitizeRequestGuardrail('map', 1, () => ({ headers: {}, content: new Map() }));
// @ts-expect-error a sanitized response must be JSON
registerLlmSanitizeResponseGuardrail('map', 1, () => new Map());
// @ts-expect-error measurements must be measurement records
metric('records', [{ name: 'm' }]);
metric('attributes', [
  // @ts-expect-error measurement attributes must be JSON
  { name: 'm', kind: MetricKind.Counter, valueType: MetricValueType.U64, value: 1, attributes: new Map() },
]);
// @ts-expect-error a scope callback must return JSON
withScope('map', ScopeType.Agent, () => new Map());
// @ts-expect-error stream chunks must be JSON
llmStreamCallExecute('map', shape, async function* () {
  yield new Map();
});
llmStreamCallExecute(
  'aggregate',
  shape,
  () => chunks(),
  undefined,
  // @ts-expect-error the stream aggregate must be JSON
  () => new Map(),
);
// @ts-expect-error codec outputs must be JSON
new OpenAIChatCodec().decodeResponse(new Map());
// @ts-expect-error a known kind rejects unknown fields
plugin.ComponentSpec('pricing', { sources: [], typo: 1 });
// @ts-expect-error an open kind still needs a JSON object
plugin.ComponentSpec('example.add-tag', { run: () => 1 });
// @ts-expect-error an open kind rejects arrays
plugin.ComponentSpec('example.add-tag', [1]);
// @ts-expect-error an open kind rejects functions
plugin.ComponentSpec('example.add-tag', () => 1);
// @ts-expect-error headers must be an object
new LlmRequest('x', null);
// @ts-expect-error content must be JSON
new LlmRequest({}, new Map());
// @ts-expect-error a union with an array member is not a JSON object
new LlmRequest(paramsOrList, null);
// @ts-expect-error a union with an array member is not a JSON object
plugin.ComponentSpec('example.add-tag', paramsOrList);
// @ts-expect-error log fields must be a JSON object
info('array-fields', null, [1]);
// @ts-expect-error log fields must be a JSON object
info('string-fields', null, 'x');

export const checked = [
  jsonRoundTrip,
  shapeRoundTrip,
  executed,
  scoped,
  voided,
  fromJson,
  typedResponse,
  typedTool,
  typedToolResult,
  openKind,
  built,
  forwardJson,
  forwardObject,
  forwardRequest,
  forwardResult,
  forwardOutcome,
  forwardRequestOutcome,
  forwardFields,
  forwardMetadata,
  forwardMeasurements,
];
