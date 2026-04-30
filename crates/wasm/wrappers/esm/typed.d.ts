// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Typed wrappers for NeMo Flow WebAssembly execute APIs.
 *
 * Provides generic typed versions of `toolCallExecute`, `llmCallExecute`,
 * and `llmStreamCallExecute` that use explicit `Codec<T>` objects to
 * serialize/deserialize at the API boundary.
 */

import { ScopeHandle, LlmStream } from './nemo_flow_wasm.js';

/** One JSON scalar value accepted by the typed wrapper APIs. */
export type JsonPrimitive = string | number | boolean | null;
/** A JSON object with recursively JSON-serializable values. */
export interface JsonObject {
  [key: string]: JsonValue;
}
/** A JSON array with recursively JSON-serializable values. */
export interface JsonArray extends Array<JsonValue> {}
/** Any JSON-serializable value accepted by the typed wrapper APIs. */
export type JsonValue = JsonPrimitive | JsonObject | JsonArray;

/** Canonical JSON shape for an opaque LLM request payload. */
export interface LlmRequestShape {
  headers: JsonObject;
  content: JsonValue;
}

/**
 * A codec for annotating and unwrapping LLM JSON request payloads.
 */
export interface LlmCodec {
  decode(request: JsonValue): JsonValue;
  encode(annotated: JsonValue, original: JsonValue): JsonValue;
}

/**
 * A codec for normalizing and decoding raw LLM responses into `JsonValue`.
 */
export interface LlmResponseCodec {
  decodeResponse(response: JsonValue): JsonValue;
}

/**
 * A codec that converts between a typed value `T` and a JSON-serializable
 * representation (`JsonValue` by default).
 */
export interface Codec<T, TJson = JsonValue> {
  /** Convert a typed value to a JSON-serializable object. */
  toJson(value: T): TJson;
  /** Reconstruct a typed value from a JSON-serializable object. */
  fromJson(data: TJson): T;
}

/**
 * A passthrough codec that performs no conversion.
 * Use when arguments or results are already plain JSON objects.
 */
export declare class JsonPassthrough implements Codec<JsonValue> {
  toJson(value: JsonValue): JsonValue;
  fromJson(data: JsonValue): JsonValue;
}

/**
 * Encode a plain or annotated payload with a request codec.
 *
 * Exposes the small dispatch helper used by the typed wrappers so tests can
 * verify how annotated intercept payloads are re-encoded.
 *
 * @param codec - Codec with an `encode` implementation.
 * @param payload - Plain request payload or `{ annotated, original }` wrapper.
 * @returns The value returned by `codec.encode`.
 * @remarks Plain payloads are forwarded as `annotated` with `original = null`;
 * annotated payloads preserve both values for intercept-aware codecs.
 */
export declare function __testEncodeWithCodec(
  codec: {
    encode(annotated: unknown, original: unknown): unknown;
  },
  payload: unknown,
): unknown;

/** Options for `typedToolExecute`. */
export interface TypedToolExecuteOptions {
  handle?: ScopeHandle | null;
  attributes?: number | null;
  data?: JsonValue;
  metadata?: JsonValue;
}

/** Options for `typedLlmExecute`. */
export interface TypedLlmExecuteOptions {
  handle?: ScopeHandle | null;
  attributes?: number | null;
  data?: JsonValue;
  metadata?: JsonValue;
  modelName?: string | null;
  codec?: LlmCodec | null;
  responseCodec?: LlmResponseCodec | null;
}

/** Options for `typedLlmStreamExecute`. */
export interface TypedLlmStreamExecuteOptions {
  handle?: ScopeHandle | null;
  attributes?: number | null;
  data?: JsonValue;
  metadata?: JsonValue;
  modelName?: string | null;
  codec?: LlmCodec | null;
  responseCodec?: LlmResponseCodec | null;
}

/**
 * Execute a typed tool call through the JSON middleware pipeline.
 *
 * Converts `args` to JSON, runs the native tool execution lifecycle, and
 * decodes the final JSON result back into the caller's typed result shape.
 *
 * @param name - Tool name.
 * @param args - Typed tool arguments.
 * @param func - The tool implementation.
 * @param argsCodec - Codec for args serialization/deserialization.
 * @param resultCodec - Codec for result serialization/deserialization.
 * @param options - Optional scope handle, attributes, data, metadata.
 * @returns A promise resolving to the decoded typed tool result.
 * @remarks The wrapper accepts both synchronous and promise-returning tool
 * implementations; codec failures and native execution errors propagate to the
 * returned promise.
 */
export declare function typedToolExecute<TArgs, TResult>(
  name: string,
  args: TArgs,
  func: (args: TArgs) => TResult | Promise<TResult>,
  argsCodec: Codec<TArgs>,
  resultCodec: Codec<TResult>,
  options?: TypedToolExecuteOptions,
): Promise<TResult>;

/**
 * Execute a typed LLM call through the JSON middleware pipeline.
 *
 * Forwards the JSON-shaped request payload into the native LLM lifecycle and
 * decodes the final response with the supplied response codec before resolving.
 *
 * @param name - Model/provider name.
 * @param request - The LLM request payload (plain JSON object).
 * @param func - The LLM implementation.
 * @param responseCodec - Codec for response serialization/deserialization.
 * @param options - Optional scope handle, attributes, data, metadata, modelName.
 * @returns A promise resolving to the decoded typed LLM response.
 * @remarks `options.responseCodec` only affects annotated response event
 * payloads; failures while decoding those event payloads are downgraded to
 * debug logging and do not rewrite the caller-visible response.
 */
export declare function typedLlmExecute<TRequest extends LlmRequestShape, TResponse>(
  name: string,
  request: TRequest,
  func: (request: TRequest) => TResponse | Promise<TResponse>,
  responseCodec: Codec<TResponse>,
  options?: TypedLlmExecuteOptions,
): Promise<TResponse>;

/**
 * Execute a typed streaming LLM call through the JSON middleware pipeline.
 *
 * Individual chunks yielded by the stream are converted to JSON via
 * `chunkCodec` before entering the middleware pipeline. After interception,
 * each chunk is converted back via `chunkCodec` before being passed to
 * `collector`.
 *
 * The `finalizer` returns a typed aggregated response which is converted
 * to JSON via `responseCodec` before flowing through sanitize-response
 * guardrails and the END event.
 *
 * @param name - Model/provider name.
 * @param request - The LLM request payload (plain JSON object).
 * @param func - The LLM stream implementation.
 * @param collector - Called with each typed chunk (after intercepts and
 *   deserialization via chunkCodec).
 * @param finalizer - Called once when the stream is exhausted; returns the
 *   typed aggregated response.
 * @param chunkCodec - Codec for converting individual stream chunks between
 *   TChunk and JSON.
 * @param responseCodec - Codec for converting the finalizer's typed result
 *   to JSON.
 * @param options - Optional scope handle, attributes, data, metadata, modelName.
 * @returns A promise resolving to the native `LlmStream` handle.
 * @remarks The wrapper bridges async iteration back into the native stream
 * lifecycle and closes the stream even when the source iterator throws.
 */
export declare function typedLlmStreamExecute<TRequest extends LlmRequestShape, TChunk, TResponse>(
  name: string,
  request: TRequest,
  func: (request: TRequest) => AsyncIterable<TChunk>,
  collector: (chunk: TChunk) => void,
  finalizer: () => TResponse,
  chunkCodec: Codec<TChunk>,
  responseCodec: Codec<TResponse>,
  options?: TypedLlmStreamExecuteOptions,
): Promise<LlmStream>;
