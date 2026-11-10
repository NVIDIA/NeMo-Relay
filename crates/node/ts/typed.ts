// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Typed wrappers for the execute APIs, converting values with a `Codec<T>`. */

import type {
  Json,
  JsonCompatible,
  JsonValue,
  LlmCodec,
  LlmRequestCompatible,
  LlmRequestShape,
  LlmResponseCodec,
  LlmStream,
  ScopeHandle,
  ToolExecutionResult as JsonToolExecutionResult,
} from '../binding.js';
import * as lib from './index.js';

export type {
  Json,
  JsonArray,
  JsonCompatible,
  JsonObject,
  JsonPrimitive,
  JsonValue,
  LlmCodec,
  LlmRequestCompatible,
  LlmRequestShape,
  LlmResponseCodec,
} from '../binding.js';

/** Result returned to the application after tool execution. */
export interface ToolExecutionResult<TResult> {
  result: TResult;
  annotation?: JsonValue;
}

/** Converts between a typed value `T` and JSON. */
export interface Codec<T, TJson extends JsonCompatible<TJson> = JsonValue> {
  /** Converts a typed value to JSON. */
  toJson: (value: T) => TJson;
  /** Rebuilds a typed value from JSON. */
  fromJson: (data: JsonValue) => T;
}

/** Codec that returns values unchanged. */
export class JsonPassthrough implements Codec<JsonValue> {
  toJson(value: JsonValue): JsonValue {
    return value;
  }

  fromJson(data: JsonValue): JsonValue {
    return data;
  }
}

/** Options for `typedToolExecute`. */
export interface TypedToolExecuteOptions<
  TData extends JsonCompatible<TData> = JsonValue,
  TMetadata extends JsonCompatible<TMetadata> = JsonValue,
> {
  handle?: ScopeHandle | null;
  attributes?: number | null;
  data?: TData;
  metadata?: TMetadata;
  toolCallId?: string | null;
}

/** Options for `typedLlmExecute`. */
export interface TypedLlmExecuteOptions<
  TData extends JsonCompatible<TData> = JsonValue,
  TMetadata extends JsonCompatible<TMetadata> = JsonValue,
  TAnnotated extends JsonCompatible<TAnnotated> = JsonValue,
  TEncoded extends LlmRequestCompatible<TEncoded> = LlmRequestShape,
  TDecodedResponse extends JsonCompatible<TDecodedResponse> = JsonValue,
> {
  handle?: ScopeHandle | null;
  attributes?: number | null;
  data?: TData;
  metadata?: TMetadata;
  modelName?: string | null;
  codec?: LlmCodec<TAnnotated, TEncoded> | null;
  responseCodec?: LlmResponseCodec<TDecodedResponse> | null;
}

/** Options for `typedLlmStreamExecute`. */
export interface TypedLlmStreamExecuteOptions<
  TData extends JsonCompatible<TData> = JsonValue,
  TMetadata extends JsonCompatible<TMetadata> = JsonValue,
  TAnnotated extends JsonCompatible<TAnnotated> = JsonValue,
  TEncoded extends LlmRequestCompatible<TEncoded> = LlmRequestShape,
  TDecodedResponse extends JsonCompatible<TDecodedResponse> = JsonValue,
> {
  handle?: ScopeHandle | null;
  attributes?: number | null;
  data?: TData;
  metadata?: TMetadata;
  modelName?: string | null;
  codec?: LlmCodec<TAnnotated, TEncoded> | null;
  responseCodec?: LlmResponseCodec<TDecodedResponse> | null;
}

interface EncodedToolExecutionResult<TResultJson> {
  result: TResultJson;
  annotation?: JsonValue;
}

function encodeToolExecutionResult<TResult, TResultJson extends JsonCompatible<TResultJson>>(
  executionResult: ToolExecutionResult<TResult>,
  resultCodec: Codec<TResult, TResultJson>,
): EncodedToolExecutionResult<TResultJson> {
  if (executionResult === null || typeof executionResult !== 'object' || !('result' in executionResult)) {
    throw new TypeError('typed tool callback must return ToolExecutionResult');
  }
  const { result, annotation } = executionResult;
  return {
    result: resultCodec.toJson(result),
    ...(annotation == null ? {} : { annotation }),
  };
}

/**
 * Executes a tool call with typed arguments and result.
 *
 * `args` is encoded with `argsCodec` before the tool lifecycle runs, and the
 * JSON result is decoded with `resultCodec` afterwards. The annotation is
 * returned as the pipeline produced it.
 *
 * @param name - Tool name.
 * @param args - Typed tool arguments.
 * @param func - The tool. It may return the result directly or in a promise.
 * @param argsCodec - Converts the arguments to and from JSON.
 * @param resultCodec - Converts the result to and from JSON.
 * @param options - Scope handle, attributes, data, metadata, and tool call id.
 * @returns The decoded result and its annotation.
 * @remarks A codec error or an execution error rejects the returned promise.
 */
export async function typedToolExecute<
  TArgs,
  TResult,
  TArgsJson extends JsonCompatible<TArgsJson> = JsonValue,
  TResultJson extends JsonCompatible<TResultJson> = JsonValue,
  TData extends JsonCompatible<TData> = JsonValue,
  TMetadata extends JsonCompatible<TMetadata> = JsonValue,
>(
  name: string,
  args: TArgs,
  func: (args: TArgs, signal: AbortSignal) => ToolExecutionResult<TResult> | Promise<ToolExecutionResult<TResult>>,
  argsCodec: Codec<TArgs, TArgsJson>,
  resultCodec: Codec<TResult, TResultJson>,
  options: TypedToolExecuteOptions<TData, TMetadata> = {},
): Promise<ToolExecutionResult<TResult>> {
  const jsonResult: JsonToolExecutionResult = await lib.toolCallExecuteAsync(
    name,
    argsCodec.toJson(args),
    async (jsonArgs: Json, signal: AbortSignal): Promise<EncodedToolExecutionResult<unknown>> =>
      encodeToolExecutionResult(await func(argsCodec.fromJson(jsonArgs), signal), resultCodec),
    options.handle,
    options.attributes,
    options.data,
    options.metadata,
    options.toolCallId,
  );
  return {
    result: resultCodec.fromJson(jsonResult.result),
    ...(jsonResult.annotation == null ? {} : { annotation: jsonResult.annotation }),
  };
}

/**
 * Executes an LLM call with a typed response.
 *
 * The LLM lifecycle runs on `request`, and the JSON response is decoded with
 * `responseCodec` before it is returned.
 *
 * @param name - Provider name.
 * @param request - Request with `headers` and `content`.
 * @param func - The LLM call. It may return the response directly or in a promise.
 * @param responseCodec - Converts the response to and from JSON.
 * @param options - Scope handle, attributes, data, metadata, model name, and codecs.
 * @returns The decoded response.
 * @remarks `options.responseCodec` decodes the response copied into events. It
 * does not change the returned response.
 */
export async function typedLlmExecute<
  TRequest extends LlmRequestCompatible<TRequest>,
  TResponse,
  TResponseJson extends JsonCompatible<TResponseJson> = JsonValue,
  TData extends JsonCompatible<TData> = JsonValue,
  TMetadata extends JsonCompatible<TMetadata> = JsonValue,
  TAnnotated extends JsonCompatible<TAnnotated> = JsonValue,
  TEncoded extends LlmRequestCompatible<TEncoded> = LlmRequestShape,
  TDecodedResponse extends JsonCompatible<TDecodedResponse> = JsonValue,
>(
  name: string,
  request: TRequest,
  func: (request: LlmRequestShape, signal: AbortSignal) => TResponse | Promise<TResponse>,
  responseCodec: Codec<TResponse, TResponseJson>,
  options: TypedLlmExecuteOptions<TData, TMetadata, TAnnotated, TEncoded, TDecodedResponse> = {},
): Promise<TResponse> {
  const { codec, responseCodec: eventCodec } = options;
  const jsonResult = await lib.llmCallExecuteAsync(
    name,
    request,
    async (jsonRequest: LlmRequestShape, signal: AbortSignal) => responseCodec.toJson(await func(jsonRequest, signal)),
    options.handle,
    options.attributes,
    options.data,
    options.metadata,
    options.modelName,
    codec == null ? undefined : (jsonRequest: LlmRequestShape) => codec.decode(jsonRequest),
    codec == null ? undefined : ({ annotated, original }) => codec.encode(annotated, original),
    eventCodec == null ? undefined : (response: Json) => eventCodec.decodeResponse(response),
  );
  return responseCodec.fromJson(jsonResult);
}

async function* encodeChunks<TChunk, TChunkJson extends JsonCompatible<TChunkJson>>(
  source: AsyncIterable<TChunk>,
  chunkCodec: Codec<TChunk, TChunkJson>,
): AsyncIterable<TChunkJson> {
  for await (const typedChunk of source) {
    yield chunkCodec.toJson(typedChunk);
  }
}

/**
 * Executes a streaming LLM call with typed chunks and response.
 *
 * Each chunk from `func` is encoded with `chunkCodec` before middleware sees it
 * and decoded again before `collector` receives it. The value from `finalizer`
 * is encoded with `responseCodec`.
 *
 * @param name - Provider name.
 * @param request - Request with `headers` and `content`.
 * @param func - The streaming LLM call. It receives the request and an `AbortSignal` and returns an async iterable of chunks.
 * @param collector - Receives one decoded chunk at a time. It may return a promise.
 * @param finalizer - Returns the final response once the stream ends. It may return a promise.
 * @param chunkCodec - Converts chunks to and from JSON.
 * @param responseCodec - Converts the final response to and from JSON.
 * @param options - Scope handle, attributes, data, metadata, model name, and codecs.
 * @returns The `LlmStream` to read chunks from.
 * @remarks An error in `collector` fails the stream. An error in `finalizer`
 * is thrown by `stream.close()`. A reader that stops early must await
 * `stream.close()`, which aborts the signal and waits for cleanup.
 */
export function typedLlmStreamExecute<
  TRequest extends LlmRequestCompatible<TRequest>,
  TChunk,
  TResponse,
  TChunkJson extends JsonCompatible<TChunkJson> = JsonValue,
  TResponseJson extends JsonCompatible<TResponseJson> = JsonValue,
  TData extends JsonCompatible<TData> = JsonValue,
  TMetadata extends JsonCompatible<TMetadata> = JsonValue,
  TAnnotated extends JsonCompatible<TAnnotated> = JsonValue,
  TEncoded extends LlmRequestCompatible<TEncoded> = LlmRequestShape,
  TDecodedResponse extends JsonCompatible<TDecodedResponse> = JsonValue,
>(
  name: string,
  request: TRequest,
  func: (request: LlmRequestShape, signal: AbortSignal) => AsyncIterable<TChunk> | Promise<AsyncIterable<TChunk>>,
  collector: (chunk: TChunk) => void | Promise<void>,
  finalizer: () => TResponse | Promise<TResponse>,
  chunkCodec: Codec<TChunk, TChunkJson>,
  responseCodec: Codec<TResponse, TResponseJson>,
  options: TypedLlmStreamExecuteOptions<TData, TMetadata, TAnnotated, TEncoded, TDecodedResponse> = {},
): Promise<LlmStream> {
  const { codec, responseCodec: eventCodec } = options;
  return lib.llmStreamCallExecute(
    name,
    request,
    async (jsonRequest: LlmRequestShape, signal: AbortSignal) => {
      const source = await func(jsonRequest, signal);
      if (source == null || typeof source[Symbol.asyncIterator] !== 'function') {
        throw new TypeError('typed streaming callback must return an AsyncIterable');
      }
      return encodeChunks(source, chunkCodec);
    },
    (jsonChunk: Json) => collector(chunkCodec.fromJson(jsonChunk)),
    async () => responseCodec.toJson(await finalizer()),
    options.handle,
    options.attributes,
    options.data,
    options.metadata,
    options.modelName,
    codec == null ? undefined : (jsonRequest: LlmRequestShape) => codec.decode(jsonRequest),
    codec == null ? undefined : ({ annotated, original }) => codec.encode(annotated, original),
    eventCodec == null ? undefined : (response: Json) => eventCodec.decodeResponse(response),
  );
}
