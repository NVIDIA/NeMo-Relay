// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Typed wrappers for NeMo Flow WASM execute APIs.
 *
 * Provides generic typed versions of `toolCallExecute`, `llmCallExecute`,
 * and `llmStreamCallExecute` that use explicit `Codec<T>` objects to
 * serialize/deserialize at the API boundary. The native runtime operates
 * on plain JSON throughout.
 *
 * @example
 * import { typedToolExecute, JsonPassthrough } from './typed.js';
 * const myCodec = {
 *   toJson(val) { return { x: val.x }; },
 *   fromJson(data) { return new MyClass(data.x); },
 * };
 * const result = await typedToolExecute('tool', myObj, fn, myCodec, myResultCodec);
 */

import {
  toolCallExecute,
  llmCallExecute,
  llmStreamCallExecute,
} from './pkg/nemo_flow_wasm.js';

/**
 * A passthrough codec that performs no conversion (identity).
 */
export class JsonPassthrough {
  toJson(value) {
    return value;
  }
  fromJson(data) {
    return data;
  }
}

/**
 * Execute a tool call with explicit codec-based typed serialization.
 *
 * @template TArgs
 * @template TResult
 * @param {string} name - Tool name.
 * @param {TArgs} args - Typed tool arguments.
 * @param {function(TArgs): Promise<TResult>} func - The tool implementation.
 * @param {Codec<TArgs>} argsCodec - Codec for serializing/deserializing args.
 * @param {Codec<TResult>} resultCodec - Codec for serializing/deserializing the result.
 * @param {object} [options] - Optional parameters.
 * @param {WasmScopeHandle} [options.handle] - Parent scope handle.
 * @param {number} [options.attributes] - Tool attribute bitflags.
 * @param {*} [options.data] - Application data.
 * @param {*} [options.metadata] - Metadata.
 * @returns {Promise<TResult>}
 */
export async function typedToolExecute(name, args, func, argsCodec, resultCodec, options) {
  const opts = options || {};
  const jsonArgs = argsCodec.toJson(args);

  const jsonFunc = (jsonArgsInner) => {
    const typedArgs = argsCodec.fromJson(jsonArgsInner);
    const typedResult = func(typedArgs);
    if (typedResult && typeof typedResult.then === 'function') {
      return typedResult.then((r) => resultCodec.toJson(r));
    }
    return resultCodec.toJson(typedResult);
  };

  const jsonResult = await toolCallExecute(
    name,
    jsonArgs,
    jsonFunc,
    opts.handle || null,
    opts.attributes || null,
    opts.data || null,
    opts.metadata || null,
  );

  return resultCodec.fromJson(jsonResult);
}

/**
 * Execute an LLM call with explicit codec-based typed response deserialization.
 *
 * @template TResponse
 * @param {string} name - Model/provider name.
 * @param {*} native - The native LLM request payload (plain JSON object).
 * @param {function(*): Promise<TResponse>} func - The LLM implementation.
 * @param {Codec<TResponse>} responseCodec - Codec for serializing/deserializing the response.
 * @param {object} [options] - Optional parameters.
 * @param {WasmScopeHandle} [options.handle] - Parent scope handle.
 * @param {number} [options.attributes] - LLM attribute bitflags.
 * @param {*} [options.data] - Application data.
 * @param {*} [options.metadata] - Metadata.
 * @param {string} [options.modelName] - Model name for ATIF export.
 * @returns {Promise<TResponse>}
 */
export async function typedLlmExecute(name, request, func, responseCodec, options) {
  const opts = options || {};

  const jsonFunc = (req) => {
    const typedResult = func(req);
    if (typedResult && typeof typedResult.then === 'function') {
      return typedResult.then((r) => responseCodec.toJson(r));
    }
    return responseCodec.toJson(typedResult);
  };

  const jsonResult = await llmCallExecute(
    name,
    request,
    jsonFunc,
    opts.handle || null,
    opts.attributes || null,
    opts.data || null,
    opts.metadata || null,
    opts.modelName || null,
    null,
    null,
  );

  return responseCodec.fromJson(jsonResult);
}

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
 * @template TChunk
 * @template TResponse
 * @param {string} name - Model/provider name.
 * @param {*} native - The native LLM request payload (plain JSON object).
 * @param {function(*): Promise<*>} func - The LLM stream implementation.
 * @param {function(TChunk): void} collector - Called with each typed chunk (after intercepts).
 * @param {function(): TResponse} finalizer - Called once when stream is exhausted; returns
 *   the typed aggregated response.
 * @param {Codec<TChunk>} chunkCodec - Codec for converting individual stream chunks.
 * @param {Codec<TResponse>} responseCodec - Codec for converting the finalizer's result.
 * @param {object} [options] - Optional parameters.
 * @param {WasmScopeHandle} [options.handle] - Parent scope handle.
 * @param {number} [options.attributes] - LLM attribute bitflags.
 * @param {*} [options.data] - Application data.
 * @param {*} [options.metadata] - Metadata.
 * @param {string} [options.modelName] - Model name for ATIF export.
 * @returns {Promise<WasmLlmStream>}
 */
export async function typedLlmStreamExecute(
  name, native, func, collector, finalizer,
  chunkCodec, responseCodec, options,
) {
  const opts = options || {};

  // Wrap func: convert typed chunks to JSON
  const jsonFunc = async (nativeInner) => {
    const chunks = [];
    for await (const typedChunk of func(nativeInner)) {
      chunks.push(chunkCodec.toJson(typedChunk));
    }
    return chunks;
  };

  const jsonCollector = collector
    ? (jsonChunk) => { collector(chunkCodec.fromJson(jsonChunk)); }
    : null;

  const jsonFinalizer = finalizer
    ? () => responseCodec.toJson(finalizer())
    : null;

  return await llmStreamCallExecute(
    name,
    native,
    jsonFunc,
    jsonCollector,
    jsonFinalizer,
    opts.handle || null,
    opts.attributes || null,
    opts.data || null,
    opts.metadata || null,
    opts.modelName || null,
    null,
    null,
  );
}
