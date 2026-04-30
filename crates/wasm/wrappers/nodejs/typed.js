// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const { toolCallExecute, llmCallExecute, llmStreamCallExecute } = require('./nemo_flow_wasm.js');

/**
 * A passthrough codec that performs no conversion.
 *
 * Mirrors the JSON payloads passed through the wrapper unchanged so tests and
 * simple integrations can opt out of custom serialization logic.
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
 * result shape.
 *
 * @template TArgs
 * @template TResult
 * @param {string} name - Tool name reported to the runtime.
 * @param {TArgs} args - Typed tool arguments supplied by the caller.
 * @param {function(TArgs): TResult | Promise<TResult>} func - Tool implementation to execute.
 * @param {{ toJson(value: TArgs): *, fromJson(data: *): TArgs }} argsCodec - Codec used to serialize and deserialize tool args.
 * @param {{ toJson(value: TResult): *, fromJson(data: *): TResult }} resultCodec - Codec used to serialize and deserialize tool results.
 * @param {object} [options] - Optional execution-scoping metadata.
 * @returns {Promise<TResult>} A promise resolving to the decoded typed tool result.
 * @remarks The wrapper accepts both synchronous and promise-returning tool
 * implementations; codec failures and native execution errors propagate to the
 * returned promise.
 */
async function typedToolExecute(name, args, func, argsCodec, resultCodec, options) {
  const opts = options || {};
  const jsonArgs = argsCodec.toJson(args);

  const jsonFunc = (jsonArgsInner) => {
    const typedArgs = argsCodec.fromJson(jsonArgsInner);
    const typedResult = func(typedArgs);
    if (typedResult && typeof typedResult.then === 'function') {
      return typedResult.then((result) => resultCodec.toJson(result));
    }
    return resultCodec.toJson(typedResult);
  };

  const jsonResult = await toolCallExecute(
    name,
    jsonArgs,
    jsonFunc,
    opts.handle ?? null,
    opts.attributes ?? null,
    opts.data ?? null,
    opts.metadata ?? null,
  );

  return resultCodec.fromJson(jsonResult);
}

/**
 * Execute a typed LLM call through the JSON middleware pipeline.
 *
 * Forwards the JSON-shaped request payload into the native LLM lifecycle and
 * decodes the final response with the supplied response codec before resolving.
 *
 * @template TResponse
 * @param {string} name - Model/provider name.
 * @param {*} request - The LLM request payload ({ headers, content }).
 * @param {function(*): TResponse | Promise<TResponse>} func - LLM implementation to execute.
 * @param {{ toJson(value: TResponse): *, fromJson(data: *): TResponse }} responseCodec - Codec used to serialize and deserialize the final response.
 * @param {object} [options] - Optional execution-scoping metadata and codec hooks.
 * @returns {Promise<TResponse>} A promise resolving to the decoded typed LLM response.
 * @remarks `options.responseCodec` only affects annotated response event
 * payloads; failures while decoding those event payloads are downgraded to
 * debug logging and do not rewrite the caller-visible response.
 */
async function typedLlmExecute(name, request, func, responseCodec, options) {
  const opts = options || {};

  const jsonFunc = (requestInner) => {
    const typedResult = func(requestInner);
    if (typedResult && typeof typedResult.then === 'function') {
      return typedResult.then((result) => responseCodec.toJson(result));
    }
    return responseCodec.toJson(typedResult);
  };

  const jsonResult = await llmCallExecute(
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

  return responseCodec.fromJson(jsonResult);
}

/**
 * Execute a typed streaming LLM call through the JSON middleware pipeline.
 *
 * Converts typed stream chunks to JSON before they enter the native runtime
 * and decodes intercepted chunks back into typed values for the collector.
 *
 * @template TChunk
 * @template TResponse
 * Accepts the public positional call shape
 * `(name, request, func, collector, finalizer, chunkCodec, responseCodec, options)`.
 * The implementation captures the trailing codec/options values via rest
 * parameters so callers keep the same API while the formal parameter list stays
 * below the lint threshold.
 *
 * @param {string} name - Model/provider name.
 * @param {*} request - The LLM request payload ({ headers, content }).
 * @param {function(*): AsyncIterable<TChunk>} func - Async iterable producer for typed stream chunks.
 * @param {function(TChunk): void} collector - Callback invoked with each decoded typed chunk.
 * @param {function(): TResponse} finalizer - Callback that returns the final typed aggregate response.
 * @param {{ toJson(value: TChunk): *, fromJson(data: *): TChunk }} chunkCodec - Codec used to serialize and deserialize stream chunks.
 * @param {{ toJson(value: TResponse): *, fromJson(data: *): TResponse }} responseCodec - Codec used to serialize and deserialize the final response.
 * @param {object} [options] - Optional execution-scoping metadata and codec hooks.
 * @returns {Promise<*>} A promise resolving to the native stream handle.
 * @remarks The wrapper buffers encoded chunks into the array-based WebAssembly stream
 * bridge and still runs the finalizer-driven response path for END-event
 * processing.
 */
async function typedLlmStreamExecute(name, request, func, collector, finalizer, ...streamArgs) {
  const [chunkCodec, responseCodec, options] = streamArgs;
  const opts = options || {};

  const jsonFunc = async (requestInner) => {
    const chunks = [];
    for await (const typedChunk of func(requestInner)) {
      chunks.push(chunkCodec.toJson(typedChunk));
    }
    return chunks;
  };

  const jsonCollector = collector
    ? (jsonChunk) => {
        collector(chunkCodec.fromJson(jsonChunk));
      }
    : null;

  const jsonFinalizer = finalizer ? () => responseCodec.toJson(finalizer()) : null;

  return llmStreamCallExecute(
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

exports.JsonPassthrough = JsonPassthrough;
exports.typedToolExecute = typedToolExecute;
exports.typedLlmExecute = typedLlmExecute;
exports.typedLlmStreamExecute = typedLlmStreamExecute;
exports.__testEncodeWithCodec = encodeWithCodec;
