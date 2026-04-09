// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Typed wrappers for NeMo Flow Node.js execute APIs.
 *
 * Provides generic typed versions of `toolCallExecute` and `llmCallExecute`
 * that use explicit `Codec<T>` objects to serialize/deserialize at the API
 * boundary.
 */

import { JsScopeHandle, LlmStream } from './index';

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
  handle?: JsScopeHandle | null;
  attributes?: number | null;
  data?: JsonValue;
  metadata?: JsonValue;
}

/** Options for `typedLlmExecute`. */
export interface TypedLlmExecuteOptions {
  handle?: JsScopeHandle | null;
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
 * The request is an `LLMRequest` object ({headers, content}). The response
 * is converted via `responseCodec`.
 *
 * @param name - Model/provider name.
 * @param request - The LLM request object ({headers, content}).
 * @param func - The LLM implementation.
 * @param responseCodec - Codec for response serialization/deserialization.
 * @param options - Optional scope handle, attributes, data, metadata, modelName.
 */
export declare function typedLlmExecute<TRequest extends LlmRequestShape, TResponse>(
  name: string,
  request: TRequest,
  func: (request: TRequest) => TResponse | Promise<TResponse>,
  responseCodec: Codec<TResponse>,
  options?: TypedLlmExecuteOptions,
): Promise<TResponse>;

/** Options for `typedLlmStreamExecute`. */
export interface TypedLlmStreamExecuteOptions {
  handle?: JsScopeHandle | null;
  attributes?: number | null;
  data?: JsonValue;
  metadata?: JsonValue;
  modelName?: string | null;
}

/**
 * Execute a streaming LLM call with codec-based conversion.
 *
 * Chunks yielded by `func` are converted to JSON via `chunkCodec.toJson`
 * before entering the middleware pipeline. After interception, chunks are
 * converted back via `chunkCodec.fromJson` before reaching `collector`.
 * The `finalizer` result is converted via `responseCodec.toJson`.
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
