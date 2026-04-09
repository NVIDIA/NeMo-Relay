// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Typed wrappers for NeMo Flow Node.js execute APIs.
 *
 * Provides generic typed versions of `toolCallExecute` and `llmCallExecute`
 * that use explicit `Codec<T>` objects to serialize/deserialize at the API
 * boundary. The native runtime operates on plain JSON throughout.
 *
 * @example
 * const { typedToolExecute, JsonPassthrough } = require('./typed');
 * const myCodec = {
 *   toJson(val) { return { x: val.x }; },
 *   fromJson(data) { return new MyClass(data.x); },
 * };
 * const result = await typedToolExecute('tool', myObj, fn, myCodec, myResultCodec);
 */

'use strict';

const { createRequire } = require('module');
const path = require('path');

// Load the native binding from the same directory as this file.
const nativeRequire = createRequire(path.join(__dirname, 'index.js'));
const lib = nativeRequire('./index.js');

/**
 * A passthrough codec that performs no conversion (identity).
 */
class JsonPassthrough {
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
 * Uses the Promise-aware native entrypoint so typed callbacks may return
 * either plain values or Promises.
 *
 * @template TArgs
 * @template TResult
 * @param {string} name
 * @param {TArgs} args
 * @param {function(TArgs): TResult | Promise<TResult>} func
 * @param {Codec<TArgs>} argsCodec
 * @param {Codec<TResult>} resultCodec
 * @param {object} [options]
 * @returns {Promise<TResult>}
 */
async function typedToolExecute(name, args, func, argsCodec, resultCodec, options) {
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

  const jsonResult = await lib.toolCallExecuteAsync(
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
 * @param {*} request - The LLM request object ({headers, content}).
 * @param {function(*): Promise<TResponse>} func - The LLM implementation.
 * @param {Codec<TResponse>} responseCodec - Codec for serializing/deserializing the response.
 * @param {object} [options] - Optional parameters.
 * @param {JsScopeHandle} [options.handle] - Parent scope handle.
 * @param {number} [options.attributes] - LLM attribute bitflags.
 * @param {*} [options.data] - Application data.
 * @param {*} [options.metadata] - Metadata.
 * @param {string} [options.modelName] - Model name for ATIF export.
 * @returns {Promise<TResponse>}
 */
async function typedLlmExecute(name, request, func, responseCodec, options) {
  const opts = options || {};

  const jsonFunc = (req) => {
    const typedResult = func(req);
    if (typedResult && typeof typedResult.then === 'function') {
      return typedResult.then((r) => responseCodec.toJson(r));
    }
    return responseCodec.toJson(typedResult);
  };

  const jsonResult = await lib.llmCallExecuteAsync(
    name,
    request,
    jsonFunc,
    opts.handle || null,
    opts.attributes || null,
    opts.data || null,
    opts.metadata || null,
    opts.modelName || null,
  );

  return responseCodec.fromJson(jsonResult);
}

/**
 * Execute a streaming LLM call with explicit codec-based typed serialization.
 *
 * Chunks yielded by `func` are converted to JSON via `chunkCodec.toJson`
 * before entering the middleware pipeline. After interception, chunks are
 * converted back via `chunkCodec.fromJson` before reaching `collector`.
 * The `finalizer` result is converted via `responseCodec.toJson`.
 *
 * @template TChunk
 * @template TResponse
 * @param {string} name - Model/provider name.
 * @param {*} request - The LLM request object ({headers, content}).
 * @param {function(*): AsyncIterable<TChunk>} func - The streaming LLM implementation.
 * @param {function(TChunk): void} collector - Called with each typed chunk after intercepts.
 * @param {function(): TResponse} finalizer - Called once when the stream is exhausted.
 * @param {Codec<TChunk>} chunkCodec - Codec for serializing/deserializing chunks.
 * @param {Codec<TResponse>} responseCodec - Codec for serializing/deserializing the final response.
 * @param {object} [options] - Optional parameters.
 * @param {JsScopeHandle} [options.handle] - Parent scope handle.
 * @param {number} [options.attributes] - LLM attribute bitflags.
 * @param {*} [options.data] - Application data.
 * @param {*} [options.metadata] - Metadata.
 * @param {string} [options.modelName] - Model name for ATIF export.
 * @returns {Promise<LlmStream>}
 */
async function typedLlmStreamExecute(name, request, func, collector, finalizer, chunkCodec, responseCodec, options) {
  const opts = options || {};

  // Push-based stream bridge: NAPI cannot resolve JS Promises from
  // call_with_return_value, so the JS side drives async generator iteration
  // and pushes each chunk into Rust via the exported pushStreamChunk function.
  // The request and stream ID are passed as a wrapper object.
  const jsonFunc = (wrapper) => {
    const req = wrapper.__nemo_flow_native;
    const streamId = wrapper.__nemo_flow_stream_id;
    (async () => {
      try {
        for await (const typedChunk of func(req)) {
          lib.pushStreamChunk(streamId, chunkCodec.toJson(typedChunk));
        }
      } finally {
        lib.endStream(streamId);
      }
    })();
  };

  // Wrap collector: convert JSON chunks back to typed
  const jsonCollector = (jsonChunk) => {
    collector(chunkCodec.fromJson(jsonChunk));
  };

  // Wrap finalizer: convert typed response to JSON
  const jsonFinalizer = () => {
    return responseCodec.toJson(finalizer());
  };

  return await lib.llmStreamCallExecute(
    name,
    request,
    jsonFunc,
    jsonCollector,
    jsonFinalizer,
    opts.handle || null,
    opts.attributes || null,
    opts.data || null,
    opts.metadata || null,
    opts.modelName || null,
  );
}

module.exports = {
  JsonPassthrough,
  typedToolExecute,
  typedLlmExecute,
  typedLlmStreamExecute,
};
