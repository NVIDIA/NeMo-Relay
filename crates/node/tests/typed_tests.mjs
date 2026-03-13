// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');
const { typedToolExecute, typedLlmExecute, typedLlmStreamExecute, JsonPassthrough } = require('../typed.js');

const {
  registerToolRequestIntercept, deregisterToolRequestIntercept,
  registerToolResponseIntercept, deregisterToolResponseIntercept,
} = lib;

// ===========================================================================
// Codec helpers for testing
// ===========================================================================

/** A simple codec that wraps/unwraps a { value } envelope. */
const envelopeCodec = {
  toJson(val) { return { value: val }; },
  fromJson(data) { return data.value; },
};

/** A codec for a Point { x, y } class. */
class Point {
  constructor(x, y) { this.x = x; this.y = y; }
}

const pointCodec = {
  toJson(p) { return { x: p.x, y: p.y }; },
  fromJson(d) { return new Point(d.x, d.y); },
};

function makeNative() {
  return { headers: {}, content: { messages: [], model: 'test-model' } };
}

// ===========================================================================
// JsonPassthrough
// ===========================================================================

describe('JsonPassthrough', () => {
  it('toJson returns same value', () => {
    const p = new JsonPassthrough();
    const obj = { a: 1 };
    assert.equal(p.toJson(obj), obj);
  });

  it('fromJson returns same value', () => {
    const p = new JsonPassthrough();
    const obj = { b: 2 };
    assert.equal(p.fromJson(obj), obj);
  });
});

// ===========================================================================
// typedToolExecute
// ===========================================================================

describe('typedToolExecute', () => {
  it('basic roundtrip with JsonPassthrough', async () => {
    const passthrough = new JsonPassthrough();
    const result = await typedToolExecute(
      'pass_tool',
      { x: 10 },
      (args) => ({ result: args.x + 1 }),
      passthrough,
      passthrough,
    );
    assert.deepEqual(result, { result: 11 });
  });

  it('custom codec transforms args and result', async () => {
    const result = await typedToolExecute(
      'point_tool',
      new Point(3, 4),
      (p) => new Point(p.x * 2, p.y * 2),
      pointCodec,
      pointCodec,
    );
    assert.ok(result instanceof Point);
    assert.equal(result.x, 6);
    assert.equal(result.y, 8);
  });

  it('envelope codec wraps/unwraps', async () => {
    const passthrough = new JsonPassthrough();
    const result = await typedToolExecute(
      'envelope_tool',
      42,
      (val) => val * 3,
      envelopeCodec,
      envelopeCodec,
    );
    assert.equal(result, 126);
  });

  it('intercepts operate on JSON', async () => {
    const seen = [];
    registerToolRequestIntercept('typed_node_req', 10, false, (name, args) => {
      seen.push(args);
      args.x = 99;
      return args;
    });

    const result = await typedToolExecute(
      'int_tool',
      new Point(1, 2),
      (p) => new Point(p.x, p.y),
      pointCodec,
      pointCodec,
    );

    assert.equal(result.x, 99);
    assert.equal(seen.length, 1);
    assert.equal(typeof seen[0], 'object');
    assert.ok(!(seen[0] instanceof Point));

    deregisterToolRequestIntercept('typed_node_req');
  });

  it('response intercepts modify JSON before deserialization', async () => {
    registerToolResponseIntercept('typed_node_resp', 10, false, (name, result) => {
      result.x = 999;
      return result;
    });

    const result = await typedToolExecute(
      'resp_int_tool',
      new Point(1, 1),
      (p) => new Point(p.x, p.y),
      pointCodec,
      pointCodec,
    );

    assert.ok(result instanceof Point);
    assert.equal(result.x, 999);

    deregisterToolResponseIntercept('typed_node_resp');
  });

  it('with options (attributes, data, metadata)', async () => {
    const passthrough = new JsonPassthrough();
    const result = await typedToolExecute(
      'opts_tool',
      { v: 1 },
      (args) => args,
      passthrough,
      passthrough,
      { attributes: lib.TOOL_ATTR_LOCAL, data: { custom: true }, metadata: { ver: '1' } },
    );
    assert.deepEqual(result, { v: 1 });
  });
});

// ===========================================================================
// typedLlmExecute
// ===========================================================================

describe('typedLlmExecute', () => {
  it('basic roundtrip with JsonPassthrough', async () => {
    const passthrough = new JsonPassthrough();
    const native = makeNative();
    const result = await typedLlmExecute(
      'pass_llm',
      native,
      (n) => ({ response: 'hello' }),
      passthrough,
    );
    assert.deepEqual(result, { response: 'hello' });
  });

  it('custom codec for response', async () => {
    const responseCodec = {
      toJson(val) { return { text: val }; },
      fromJson(data) { return data.text; },
    };

    const native = makeNative();
    const result = await typedLlmExecute(
      'codec_llm',
      native,
      (n) => 'hello world',
      responseCodec,
    );
    assert.equal(result, 'hello world');
  });

  it('with modelName option', async () => {
    const passthrough = new JsonPassthrough();
    const native = makeNative();
    const result = await typedLlmExecute(
      'named_llm',
      native,
      (n) => ({ ok: true }),
      passthrough,
      { modelName: 'gpt-4-turbo' },
    );
    assert.deepEqual(result, { ok: true });
  });
});

// ===========================================================================
// typedLlmStreamExecute
// ===========================================================================

describe('typedLlmStreamExecute', () => {
  it('basic stream with JsonPassthrough', async () => {
    const passthrough = new JsonPassthrough();
    const native = makeNative();

    const collected = [];
    const func = async function*(n) {
      yield { token: 'hello' };
      yield { token: 'world' };
    };
    const collector = (chunk) => collected.push(chunk);
    const finalizer = () => ({ chunks: collected });

    const stream = await typedLlmStreamExecute(
      'stream_llm', native, func, collector, finalizer,
      passthrough, passthrough,
    );

    const chunks = [];
    let chunk;
    while ((chunk = await stream.next()) !== null) {
      chunks.push(chunk);
    }

    assert.equal(chunks.length, 2);
    assert.deepEqual(chunks[0], { token: 'hello' });
    assert.equal(collected.length, 2);
  });

  it('stream with envelopeCodec for chunks and response', async () => {
    const native = makeNative();

    // The collector receives typed (unwrapped) values thanks to chunkCodec.fromJson
    const collected = [];
    const func = async function*(n) {
      // Yield raw values; typedLlmStreamExecute wraps them via chunkCodec.toJson
      yield 'alpha';
      yield 'beta';
      yield 'gamma';
    };
    const collector = (chunk) => collected.push(chunk);
    // Finalizer returns a typed value; typedLlmStreamExecute wraps via responseCodec.toJson
    const finalizer = () => collected.join(',');

    const stream = await typedLlmStreamExecute(
      'env_stream', native, func, collector, finalizer,
      envelopeCodec, envelopeCodec,
    );

    const chunks = [];
    let chunk;
    while ((chunk = await stream.next()) !== null) {
      chunks.push(chunk);
    }

    // Chunks should be the JSON-encoded form (envelopeCodec.toJson wraps as { value })
    assert.equal(chunks.length, 3);
    assert.deepEqual(chunks[0], { value: 'alpha' });
    assert.deepEqual(chunks[1], { value: 'beta' });
    assert.deepEqual(chunks[2], { value: 'gamma' });

    // Collector receives decoded (unwrapped) values via chunkCodec.fromJson
    assert.deepEqual(collected, ['alpha', 'beta', 'gamma']);
  });

  it('stream with pointCodec for chunks and response', async () => {
    const native = makeNative();

    const collected = [];
    const func = async function*(n) {
      yield new Point(1, 2);
      yield new Point(3, 4);
    };
    const collector = (chunk) => collected.push(chunk);
    const finalizer = () => new Point(
      collected.reduce((s, p) => s + p.x, 0),
      collected.reduce((s, p) => s + p.y, 0),
    );

    const stream = await typedLlmStreamExecute(
      'point_stream', native, func, collector, finalizer,
      pointCodec, pointCodec,
    );

    const chunks = [];
    let chunk;
    while ((chunk = await stream.next()) !== null) {
      chunks.push(chunk);
    }

    // Raw chunks from the stream are JSON (pointCodec.toJson output)
    assert.equal(chunks.length, 2);
    assert.deepEqual(chunks[0], { x: 1, y: 2 });
    assert.deepEqual(chunks[1], { x: 3, y: 4 });

    // Collector receives decoded Point instances via pointCodec.fromJson
    assert.equal(collected.length, 2);
    assert.ok(collected[0] instanceof Point);
    assert.equal(collected[0].x, 1);
    assert.equal(collected[0].y, 2);
    assert.ok(collected[1] instanceof Point);
    assert.equal(collected[1].x, 3);
    assert.equal(collected[1].y, 4);
  });
});

// ===========================================================================
// typedToolExecute — mixed codecs
// ===========================================================================

describe('typedToolExecute — mixed codecs', () => {
  it('pointCodec for args and envelopeCodec for result', async () => {
    const result = await typedToolExecute(
      'mixed_tool',
      new Point(5, 10),
      (p) => p.x + p.y,  // receives Point, returns number
      pointCodec,
      envelopeCodec,
    );
    // envelopeCodec.fromJson unwraps { value: 15 } to 15
    assert.equal(result, 15);
  });

  it('envelopeCodec for args and pointCodec for result', async () => {
    const result = await typedToolExecute(
      'mixed_tool_rev',
      42,
      (val) => new Point(val, val * 2),  // receives number, returns Point
      envelopeCodec,
      pointCodec,
    );
    assert.ok(result instanceof Point);
    assert.equal(result.x, 42);
    assert.equal(result.y, 84);
  });
});

// ===========================================================================
// typedToolExecute — sync function
// ===========================================================================

describe('typedToolExecute — sync function', () => {
  it('sync function with custom codec works correctly', async () => {
    // The function is synchronous (no async/Promise)
    const result = await typedToolExecute(
      'sync_tool',
      new Point(7, 3),
      (p) => new Point(p.x - p.y, p.x + p.y),
      pointCodec,
      pointCodec,
    );
    assert.ok(result instanceof Point);
    assert.equal(result.x, 4);
    assert.equal(result.y, 10);
  });

  it('sync function with envelope codec', async () => {
    const result = await typedToolExecute(
      'sync_env_tool',
      'hello',
      (val) => val.toUpperCase(),
      envelopeCodec,
      envelopeCodec,
    );
    assert.equal(result, 'HELLO');
  });
});

// ===========================================================================
// typedLlmExecute — sync function
// ===========================================================================

describe('typedLlmExecute — sync function', () => {
  it('sync function with custom codec works correctly', async () => {
    const native = makeNative();
    // The function is synchronous (no async/Promise)
    const result = await typedLlmExecute(
      'sync_llm',
      native,
      (n) => new Point(100, 200),
      pointCodec,
    );
    assert.ok(result instanceof Point);
    assert.equal(result.x, 100);
    assert.equal(result.y, 200);
  });

  it('sync function with envelope codec', async () => {
    const native = makeNative();
    const result = await typedLlmExecute(
      'sync_env_llm',
      native,
      (n) => 'sync-response',
      envelopeCodec,
    );
    assert.equal(result, 'sync-response');
  });
});
