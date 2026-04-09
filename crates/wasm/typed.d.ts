// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Typed wrappers for NeMo Flow WASM execute APIs.
 *
 * Provides generic typed versions of `toolCallExecute`, `llmCallExecute`,
 * and `llmStreamCallExecute` that use explicit `Codec<T>` objects to
 * serialize/deserialize at the API boundary.
 */

import { WasmScopeHandle, WasmLlmStream } from './pkg/nemo_flow_wasm';

export type JsonPrimitive = string | number | boolean | null;
export interface JsonObject { [key: string]: JsonValue; }
export interface JsonArray extends Array<JsonValue> {}
export type JsonValue = JsonPrimitive | JsonObject | JsonArray;

export interface LlmRequestShape {
  headers: JsonObject;
  content: JsonValue;
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

/** Options for `typedToolExecute`. */
export interface TypedToolExecuteOptions {
  handle?: WasmScopeHandle | null;
  attributes?: number | null;
  data?: JsonValue;
  metadata?: JsonValue;
}

/** Options for `typedLlmExecute`. */
export interface TypedLlmExecuteOptions {
  handle?: WasmScopeHandle | null;
  attributes?: number | null;
  data?: JsonValue;
  metadata?: JsonValue;
  modelName?: string | null;
}

/** Options for `typedLlmStreamExecute`. */
export interface TypedLlmStreamExecuteOptions {
  handle?: WasmScopeHandle | null;
  attributes?: number | null;
  data?: JsonValue;
  metadata?: JsonValue;
  modelName?: string | null;
}

/**
 * Execute a tool call with explicit codec-based typed serialization.
 *
 * Converts `args` to JSON via `argsCodec.toJson`, runs the middleware
 * pipeline, calls `func` with deserialized typed args, and returns the
 * result deserialized via `resultCodec.fromJson`.
 *
 * @param name - Tool name.
 * @param args - Typed tool arguments.
 * @param func - The tool implementation.
 * @param argsCodec - Codec for args serialization/deserialization.
 * @param resultCodec - Codec for result serialization/deserialization.
 * @param options - Optional scope handle, attributes, data, metadata.
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
 * Execute an LLM call with explicit codec-based typed response deserialization.
 *
 * The native request is an opaque JSON payload. The response is converted via
 * `responseCodec`.
 *
 * @param name - Model/provider name.
 * @param native - The native LLM request payload (plain JSON object).
 * @param func - The LLM implementation.
 * @param responseCodec - Codec for response serialization/deserialization.
 * @param options - Optional scope handle, attributes, data, metadata, modelName.
 */
export declare function typedLlmExecute<TRequest extends LlmRequestShape, TResponse>(
  name: string,
  native: TRequest,
  func: (native: TRequest) => TResponse | Promise<TResponse>,
  responseCodec: Codec<TResponse>,
  options?: TypedLlmExecuteOptions,
): Promise<TResponse>;

/**
 * Execute a streaming LLM call with explicit codec-based typed serialization.
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
 * @param native - The native LLM request payload (plain JSON object).
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
 */
export declare function typedLlmStreamExecute<TRequest extends LlmRequestShape, TChunk, TResponse>(
  name: string,
  native: TRequest,
  func: (native: TRequest) => AsyncIterable<TChunk> | Promise<AsyncIterable<TChunk>>,
  collector: ((chunk: TChunk) => void) | null,
  finalizer: (() => TResponse) | null,
  chunkCodec: Codec<TChunk>,
  responseCodec: Codec<TResponse>,
  options?: TypedLlmStreamExecuteOptions,
): Promise<WasmLlmStream>;
