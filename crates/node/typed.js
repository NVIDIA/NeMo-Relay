// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Typed wrappers for NeMo Relay Node.js execute APIs.
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

const { createRequire } = require('node:module');
const path = require('node:path');

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
 * Encode an annotated LLM payload with the caller-provided codec.
 *
 * Supports both plain request payloads and the `{ annotated, original }`
 * wrapper used by annotated-aware intercept pipelines.
 *
 * @param {{ encode(annotated: *, original: *): * }} codec - Codec used to re-encode the payload.
 * @param {*} payload - Plain or annotated payload to encode.
 * @returns {*} The payload produced by `codec.encode`.
 * @remarks Plain payloads are forwarded as `annotated` with `original = null`;
 * annotated payloads preserve both values for intercept-aware codecs.
 */
function encodeWithCodec(codec, payload) {
  if (payload && typeof payload === 'object' && 'annotated' in payload && 'original' in payload) {
    return codec.encode(payload.annotated, payload.original);
  }
  return codec.encode(payload, null);
}

function collectCodecReturn(value) {
  return value === undefined ? null : value;
}

function encodeToolExecutionResult(executionResult, resultCodec) {
  if (executionResult === null || typeof executionResult !== 'object' || !('result' in executionResult)) {
    throw new TypeError('typed tool callback must return ToolExecutionResult');
  }
  const result = executionResult.result;
  const annotation = executionResult.annotation;
  return {
    result: resultCodec.toJson(result),
    ...(annotation == null ? {} : { annotation }),
  };
}

function decodeResponseWithCodec(codec, response) {
  try {
    return collectCodecReturn(codec.decodeResponse(response));
  } catch (err) {
    if (typeof console !== 'undefined' && typeof console.debug === 'function') {
      console.debug('decodeResponseWithCodec failed', {
        error: err,
        response,
      });
    }
    return null;
  }
}

/**
 * Execute a typed tool call through the JSON middleware pipeline.
 *
 * Converts typed arguments to JSON, invokes the native tool execution
 * lifecycle, and decodes the final JSON result back into the caller's typed
 * result shape while preserving the opaque annotation unchanged.
 *
 * @template TArgs
 * @template TResult
 * @param {string} name - Tool name reported to the runtime.
 * @param {TArgs} args - Typed tool arguments supplied by the caller.
 * @param {function(TArgs, AbortSignal): { result: TResult, annotation?: * } | Promise<{ result: TResult, annotation?: * }>} func - Tool implementation to execute.
 * @param {Codec<TArgs>} argsCodec - Codec used to serialize and deserialize tool args.
 * @param {Codec<TResult>} resultCodec - Codec used to serialize and deserialize tool results.
 * @param {object} [options] - Optional execution-scoping metadata.
 * @param {string} [options.toolCallId] - Provider-specific tool call correlation ID.
 * @returns {Promise<{ result: TResult, annotation?: * }>} A promise resolving to the decoded typed tool result.
 * @remarks The wrapper accepts both synchronous and promise-returning tool
 * implementations; codec failures and native execution errors propagate to the
 * returned promise.
 */
async function typedToolExecute(name, args, func, argsCodec, resultCodec, options) {
  const opts = options || {};
  const jsonArgs = argsCodec.toJson(args);

  const jsonFunc = (jsonArgsInner, signal) => {
    const typedArgs = argsCodec.fromJson(jsonArgsInner);
    const typedResult = func(typedArgs, signal);
    if (typedResult && typeof typedResult.then === 'function') {
      return typedResult.then((result) => encodeToolExecutionResult(result, resultCodec));
    }
    return encodeToolExecutionResult(typedResult, resultCodec);
  };

  const jsonResult = await lib.toolCallExecuteAsync(
    name,
    jsonArgs,
    jsonFunc,
    opts.handle ?? null,
    opts.attributes ?? null,
    opts.data ?? null,
    opts.metadata ?? null,
    opts.toolCallId ?? null,
  );

  return {
    result: resultCodec.fromJson(jsonResult.result),
    ...(jsonResult.annotation == null ? {} : { annotation: jsonResult.annotation }),
  };
}

/**
 * Execute a typed LLM call through the JSON middleware pipeline.
 *
 * Forwards the JSON-shaped request payload into the native LLM lifecycle and
 * decodes the final response with the supplied response codec before resolving.
 *
 * @template TResponse
 * @param {string} name - Model/provider name.
 * @param {*} request - The LLM request object ({headers, content}).
 * @param {function(*, AbortSignal): TResponse | Promise<TResponse>} func - The LLM implementation.
 * @param {Codec<TResponse>} responseJsonCodec - Codec for serializing/deserializing the response.
 * @param {object} [options] - Optional parameters.
 * @param {JsScopeHandle} [options.handle] - Parent scope handle.
 * @param {number} [options.attributes] - LLM attribute bitflags.
 * @param {*} [options.data] - Application data.
 * @param {*} [options.metadata] - Metadata.
 * @param {string} [options.modelName] - Model name for ATIF export.
 * @param {{ decode(request: *): *, encode(annotated: *, original: *): * }} [options.codec]
 *   Request codec for annotated request intercepts.
 * @param {{ decodeResponse(response: *): * }} [options.responseCodec]
 *   Response codec for annotated response events.
 * @returns {Promise<TResponse>} A promise resolving to the decoded typed LLM response.
 * @remarks `options.responseCodec` only affects annotated response event
 * payloads; failures while decoding those event payloads are downgraded to
 * debug logging and do not rewrite the caller-visible response.
 */
async function typedLlmExecute(name, request, func, responseJsonCodec, options) {
  const opts = options || {};

  const jsonFunc = (req, signal) => {
    const typedResult = func(req, signal);
    if (typedResult && typeof typedResult.then === 'function') {
      return typedResult.then((r) => responseJsonCodec.toJson(r));
    }
    return responseJsonCodec.toJson(typedResult);
  };

  const jsonResult = await lib.llmCallExecuteAsync(
    name,
    request,
    jsonFunc,
    opts.handle ?? null,
    opts.attributes ?? null,
    opts.data ?? null,
    opts.metadata ?? null,
    opts.modelName ?? null,
    opts.codec ? opts.codec.decode.bind(opts.codec) : null,
    opts.codec ? (payload) => encodeWithCodec(opts.codec, payload) : null,
    opts.responseCodec ? (response) => decodeResponseWithCodec(opts.responseCodec, response) : null,
  );

  return responseJsonCodec.fromJson(jsonResult);
}

/**
 * Execute a typed streaming LLM call through the JSON middleware pipeline.
 *
 * Chunks yielded by `func` are converted to JSON via `chunkCodec.toJson`
 * before entering the middleware pipeline. After interception, chunks are
 * converted back via `chunkCodec.fromJson` before reaching `collector`.
 * The `finalizer` result is converted via `responseCodec.toJson`.
 *
 * @template TChunk
 * @template TResponse
 * Accepts the public positional call shape
 * `(name, request, func, collector, finalizer, chunkJsonCodec, responseJsonCodec, options)`.
 * The implementation captures the trailing codec/options values via rest
 * parameters to keep the external API stable while avoiding an oversized formal
 * parameter list.
 *
 * @param {string} name - Model/provider name.
 * @param {*} request - The LLM request object ({headers, content}).
 * @param {function(*, AbortSignal): AsyncIterable<TChunk> | Promise<AsyncIterable<TChunk>>} func - The streaming LLM implementation.
 * @param {function(TChunk): void | Promise<void>} collector - Called with each typed chunk after intercepts.
 * @param {function(): TResponse | Promise<TResponse>} finalizer - Called once when the stream is exhausted or closed early.
 * @param {Codec<TChunk>} chunkJsonCodec - Codec for serializing/deserializing chunks.
 * @param {Codec<TResponse>} responseJsonCodec - Codec for serializing/deserializing the final response.
 * @param {object} [options] - Optional parameters.
 * @param {JsScopeHandle} [options.handle] - Parent scope handle.
 * @param {number} [options.attributes] - LLM attribute bitflags.
 * @param {*} [options.data] - Application data.
 * @param {*} [options.metadata] - Metadata.
 * @param {string} [options.modelName] - Model name for ATIF export.
 * @param {{ decode(request: *): *, encode(annotated: *, original: *): * }} [options.codec]
 *   Request codec for annotated request intercepts.
 * @param {{ decodeResponse(response: *): * }} [options.responseCodec]
 *   Response codec for annotated response events.
 * @returns {Promise<LlmStream>} A promise resolving to the native stream handle.
 * @remarks Chunks are encoded as they are pulled from `func`. A throw or
 * rejection from `func` rejects the call, an iterator failure rejects the
 * pending `next()`, and the native bridge aborts the signal and runs iterator
 * cleanup when the consumer closes early. The producer must stop once the
 * signal aborts. Consumers that stop reading early must await `stream.close()`,
 * which waits for that cleanup without a timeout.
 */
async function typedLlmStreamExecute(name, request, func, collector, finalizer, ...streamArgs) {
  const [chunkJsonCodec, responseJsonCodec, options] = streamArgs;
  const opts = options || {};

  const jsonFunc = async (jsonRequest, signal) => {
    const source = await func(jsonRequest, signal);
    if (source == null || typeof source[Symbol.asyncIterator] !== 'function') {
      throw new TypeError('typed streaming callback must return an AsyncIterable');
    }
    return encodeChunks(source, chunkJsonCodec);
  };
  const jsonCollector = (jsonChunk) => collector(chunkJsonCodec.fromJson(jsonChunk));
  const jsonFinalizer = async () => collectCodecReturn(responseJsonCodec.toJson(await finalizer()));

  return lib.llmStreamCallExecute(
    name,
    request,
    jsonFunc,
    jsonCollector,
    jsonFinalizer,
    opts.handle ?? null,
    opts.attributes ?? null,
    opts.data ?? null,
    opts.metadata ?? null,
    opts.modelName ?? null,
    opts.codec ? opts.codec.decode.bind(opts.codec) : null,
    opts.codec ? (payload) => encodeWithCodec(opts.codec, payload) : null,
    opts.responseCodec ? (response) => decodeResponseWithCodec(opts.responseCodec, response) : null,
  );
}

async function* encodeChunks(source, chunkJsonCodec) {
  for await (const typedChunk of source) {
    yield chunkJsonCodec.toJson(typedChunk);
  }
}

module.exports = {
  __testEncodeWithCodec: encodeWithCodec,
  JsonPassthrough,
  typedToolExecute,
  typedLlmExecute,
  typedLlmStreamExecute,
};
