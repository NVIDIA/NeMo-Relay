// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Typed wrappers for Nexus WASM execute APIs.
 *
 * Provides generic typed versions of `toolCallExecute`, `llmCallExecute`,
 * and `llmStreamCallExecute` that use explicit `Codec<T>` objects to
 * serialize/deserialize at the API boundary.
 */

import { WasmScopeHandle, WasmLlmStream } from './pkg/nvidia_nat_nexus_wasm';

/**
 * A codec that converts between a typed value `T` and a JSON-serializable
 * representation (`any`).
 */
export interface Codec<T> {
  /** Convert a typed value to a JSON-serializable object. */
  toJson(value: T): any;
  /** Reconstruct a typed value from a JSON-serializable object. */
  fromJson(data: any): T;
}

/**
 * A passthrough codec that performs no conversion.
 * Use when arguments or results are already plain JSON objects.
 */
export declare class JsonPassthrough implements Codec<any> {
  toJson(value: any): any;
  fromJson(data: any): any;
}

/** Options for `typedToolExecute`. */
export interface TypedToolExecuteOptions {
  handle?: WasmScopeHandle | null;
  attributes?: number | null;
  data?: any;
  metadata?: any;
}

/** Options for `typedLlmExecute`. */
export interface TypedLlmExecuteOptions {
  handle?: WasmScopeHandle | null;
  attributes?: number | null;
  data?: any;
  metadata?: any;
  modelName?: string | null;
}

/** Options for `typedLlmStreamExecute`. */
export interface TypedLlmStreamExecuteOptions {
  handle?: WasmScopeHandle | null;
  attributes?: number | null;
  data?: any;
  metadata?: any;
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
export declare function typedLlmExecute<TResponse>(
  name: string,
  native: any,
  func: (native: any) => TResponse | Promise<TResponse>,
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
export declare function typedLlmStreamExecute<TChunk, TResponse>(
  name: string,
  native: any,
  func: (native: any) => any | Promise<any>,
  collector: ((chunk: TChunk) => void) | null,
  finalizer: (() => TResponse) | null,
  chunkCodec: Codec<TChunk>,
  responseCodec: Codec<TResponse>,
  options?: TypedLlmStreamExecuteOptions,
): Promise<WasmLlmStream>;
