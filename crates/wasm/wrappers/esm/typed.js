// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Typed wrappers for NeMo Flow WebAssembly execute APIs.
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

import { toolCallExecute, llmCallExecute, llmStreamCallExecute } from './pkg/index.js';

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
 * @param {string} name - Tool name.
 * @param {TArgs} args - Typed tool arguments.
 * @param {function(TArgs): Promise<TResult>} func - The tool implementation.
 * @param {Codec<TArgs>} argsCodec - Codec for serializing/deserializing args.
 * @param {Codec<TResult>} resultCodec - Codec for serializing/deserializing the result.
 * @param {object} [options] - Optional parameters.
 * @param {ScopeHandle} [options.handle] - Parent scope handle.
 * @param {number} [options.attributes] - Tool attribute bitflags.
 * @param {*} [options.data] - Application data.
 * @param {*} [options.metadata] - Metadata.
 * @returns {Promise<TResult>} A promise resolving to the decoded typed tool result.
 * @remarks The wrapper accepts both synchronous and promise-returning tool
 * implementations; codec failures and native execution errors propagate to the
 * returned promise.
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
 * @param {*} request - The LLM request payload (plain JSON object).
 * @param {function(*): Promise<TResponse>} func - The LLM implementation.
 * @param {Codec<TResponse>} responseCodec - Codec for serializing/deserializing the response.
 * @param {object} [options] - Optional parameters.
 * @param {ScopeHandle} [options.handle] - Parent scope handle.
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
 * Accepts the public positional call shape
 * `(name, request, func, collector, finalizer, chunkCodec, responseCodec, options)`.
 * The implementation captures the trailing codec/options values via rest
 * parameters so the exported API stays unchanged while the function signature
 * remains within the linter limit.
 *
 * @param {string} name - Model/provider name.
 * @param {*} request - The LLM request payload (plain JSON object).
 * @param {function(*): AsyncIterable<*>} func - The LLM stream implementation.
 * @param {function(TChunk): void} collector - Called with each typed chunk (after intercepts).
 * @param {function(): TResponse} finalizer - Called once when stream is exhausted; returns
 *   the typed aggregated response.
 * @param {Codec<TChunk>} chunkCodec - Codec for converting individual stream chunks.
 * @param {Codec<TResponse>} responseCodec - Codec for converting the finalizer's result.
 * @param {object} [options] - Optional parameters.
 * @param {ScopeHandle} [options.handle] - Parent scope handle.
 * @param {number} [options.attributes] - LLM attribute bitflags.
 * @param {*} [options.data] - Application data.
 * @param {*} [options.metadata] - Metadata.
 * @param {string} [options.modelName] - Model name for ATIF export.
 * @param {{ decode(request: *): *, encode(annotated: *, original: *): * }} [options.codec]
 *   Request codec for annotated request intercepts.
 * @param {{ decodeResponse(response: *): * }} [options.responseCodec]
 *   Response codec for annotated response events.
 * @returns {Promise<LlmStream>} A promise resolving to the native stream handle.
 * @remarks The wrapper bridges async iteration back into the native stream
 * lifecycle and closes the stream even when the source iterator throws.
 */
export async function typedLlmStreamExecute(name, request, func, collector, finalizer, ...streamArgs) {
  const [chunkCodec, responseCodec, options] = streamArgs;
  if (
    typeof chunkCodec !== 'object' ||
    chunkCodec === null ||
    typeof responseCodec !== 'object' ||
    responseCodec === null ||
    typeof chunkCodec.toJson !== 'function' ||
    typeof responseCodec.toJson !== 'function'
  ) {
    throw new TypeError('chunkCodec and responseCodec are required and must implement toJson');
  }
  const opts = options || {};

  // Wrap func: convert typed chunks to JSON
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

  return await llmStreamCallExecute(
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

export const __testEncodeWithCodec = encodeWithCodec;
