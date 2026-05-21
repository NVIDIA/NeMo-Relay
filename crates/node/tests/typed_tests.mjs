// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');
const {
  __testEncodeWithCodec,
  typedToolExecute,
  typedLlmExecute,
  typedLlmStreamExecute,
  JsonPassthrough,
} = require('../typed.js');
const adaptive = require('../adaptive.js');
const plugin = require('../plugin.js');

const TOOL_ATTR_LOCAL = 0b01;

async function* emitHelloWorldStream() {
  yield {
    token: 'hello',
  };
  yield {
    token: 'world',
  };
}

async function* emitEnvelopeStream() {
  yield 'alpha';
  yield 'beta';
  yield 'gamma';
}

const {
  OpenAIChatCodec,
  OpenAIResponsesCodec,
  AnthropicMessagesCodec,
  ScopeType,
  pushScope,
  popScope,
  registerToolRequestIntercept,
  deregisterToolRequestIntercept,
  registerLlmRequestIntercept,
  deregisterLlmRequestIntercept,
  registerSubscriber,
  deregisterSubscriber,
} = lib;

// ===========================================================================
// Codec helpers for testing
// ===========================================================================

/** A simple codec that wraps/unwraps a { value } envelope. */
const envelopeCodec = {
  toJson(val) {
    return {
      value: val,
    };
  },
  fromJson(data) {
    return data.value;
  },
};

/** A codec for a Point { x, y } class. */
function Point(x, y) {
  this.x = x;
  this.y = y;
}

const pointCodec = {
  toJson(p) {
    return {
      x: p.x,
      y: p.y,
    };
  },
  fromJson(d) {
    return new Point(d.x, d.y);
  },
};

function makeNative() {
  return {
    headers: {},
    content: {
      messages: [],
      model: 'test-model',
    },
  };
}

function makeOpenAIChatRequest() {
  return {
    headers: {},
    content: {
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'user',
          content: 'Hello from chat',
        },
      ],
      temperature: 0.1,
    },
  };
}

function makeOpenAIResponsesRequest() {
  return {
    headers: {},
    content: {
      model: 'gpt-4.1-mini',
      instructions: 'Be terse.',
      input: 'Hello from responses',
    },
  };
}

function makeAnthropicRequest() {
  return {
    headers: {},
    content: {
      model: 'claude-3-5-sonnet',
      messages: [
        {
          role: 'user',
          content: 'Hello from anthropic',
        },
      ],
      max_tokens: 32,
    },
  };
}

// ===========================================================================
// JsonPassthrough
// ===========================================================================

describe('JsonPassthrough', () => {
  it('toJson returns same value', () => {
    const p = new JsonPassthrough();
    const obj = {
      a: 1,
    };
    assert.equal(p.toJson(obj), obj);
  });

  it('fromJson returns same value', () => {
    const p = new JsonPassthrough();
    const obj = {
      b: 2,
    };
    assert.equal(p.fromJson(obj), obj);
  });

  it('dispatches non-annotated codec payloads through the plain encode path', () => {
    const seen = [];
    const result = __testEncodeWithCodec(
      {
        encode(annotated, original) {
          seen.push({
            annotated,
            original,
          });
          return {
            annotated,
            original,
          };
        },
      },
      {
        model: 'plain-request',
      },
    );

    assert.deepEqual(result, {
      annotated: {
        model: 'plain-request',
      },
      original: null,
    });
    assert.deepEqual(seen, [
      {
        annotated: {
          model: 'plain-request',
        },
        original: null,
      },
    ]);
  });
});

describe('adaptive typed helpers', () => {
  it('keeps adaptive helpers out of typed.js', () => {
    assert.equal('defaultAdaptiveConfig' in require('../typed.js'), false);
    assert.equal(typeof adaptive.defaultConfig, 'function');
  });

  it('build default adaptive config and components', () => {
    const config = adaptive.defaultConfig();
    config.state = {
      backend: adaptive.inMemoryBackend(),
    };
    config.telemetry = adaptive.telemetryConfig({
      learners: ['latency_sensitivity'],
    });
    config.adaptive_hints = adaptive.adaptiveHintsConfig();
    config.tool_parallelism = adaptive.toolParallelismConfig();
    config.acg = adaptive.acgConfig();

    const report = plugin.validate({
      version: 1,
      components: [adaptive.ComponentSpec(config)],
    });
    assert.deepEqual(report.diagnostics, []);
  });

  it('configures adaptive through the core plugin system', async () => {
    const report = await plugin.initialize({
      version: 1,
      components: [
        adaptive.ComponentSpec({
          version: 1,
          state: {
            backend: adaptive.inMemoryBackend(),
          },
          telemetry: adaptive.telemetryConfig({
            learners: ['latency_sensitivity'],
          }),
          adaptive_hints: adaptive.adaptiveHintsConfig(),
          tool_parallelism: adaptive.toolParallelismConfig(),
          acg: adaptive.acgConfig({
            provider: 'openai',
          }),
        }),
      ],
    });

    assert.deepEqual(report.diagnostics, []);
    plugin.clear();
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
      {
        x: 10,
      },
      (args) => ({
        result: args.x + 1,
      }),
      passthrough,
      passthrough,
    );
    assert.deepEqual(result, {
      result: 11,
    });
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
    const result = await typedToolExecute('envelope_tool', 42, (val) => val * 3, envelopeCodec, envelopeCodec);
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

  it('with options (attributes, data, metadata)', async () => {
    const passthrough = new JsonPassthrough();
    const result = await typedToolExecute(
      'opts_tool',
      {
        v: 1,
      },
      (args) => args,
      passthrough,
      passthrough,
      {
        attributes: TOOL_ATTR_LOCAL,
        data: {
          custom: true,
        },
        metadata: {
          ver: '1',
        },
      },
    );
    assert.deepEqual(result, {
      v: 1,
    });
  });

  it('preserves falsy metadata/model options and uses non-null request data', async () => {
    const events = [];
    registerSubscriber('typed_node_falsy_opts', (event) => events.push(event));

    try {
      await typedLlmExecute(
        'typed_node_falsy_opts_llm',
        makeNative(),
        () => ({
          ok: true,
        }),
        new JsonPassthrough(),
        {
          data: false,
          metadata: 0,
          modelName: '',
        },
      );

      const deadline = Date.now() + 2000;
      while (
        !events.some(
          (event) =>
            event.kind === 'scope' &&
            event.category === 'llm' &&
            event.scope_category === 'start' &&
            event.name === 'typed_node_falsy_opts_llm',
        ) &&
        Date.now() < deadline
      ) {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }

      const startEvent = events.find(
        (event) =>
          event.kind === 'scope' &&
          event.category === 'llm' &&
          event.scope_category === 'start' &&
          event.name === 'typed_node_falsy_opts_llm',
      );
      assert.deepEqual(startEvent.data, makeNative());
      assert.equal(startEvent.metadata, 0);
      assert.equal(startEvent.category_profile.model_name, '');
    } finally {
      deregisterSubscriber('typed_node_falsy_opts');
    }
  });

  it('awaits async tool functions before re-encoding results', async () => {
    const result = await typedToolExecute(
      'async_codec_tool',
      new Point(2, 5),
      async (p) => new Point(p.x + 1, p.y + 1),
      pointCodec,
      pointCodec,
    );
    assert.ok(result instanceof Point);
    assert.equal(result.x, 3);
    assert.equal(result.y, 6);
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
      (n) => ({
        response: 'hello',
      }),
      passthrough,
    );
    assert.deepEqual(result, {
      response: 'hello',
    });
  });

  it('custom codec for response', async () => {
    const responseCodec = {
      toJson(val) {
        return {
          text: val,
        };
      },
      fromJson(data) {
        return data.text;
      },
    };

    const native = makeNative();
    const result = await typedLlmExecute('codec_llm', native, (n) => 'hello world', responseCodec);
    assert.equal(result, 'hello world');
  });

  it('with modelName option', async () => {
    const passthrough = new JsonPassthrough();
    const native = makeNative();
    const result = await typedLlmExecute(
      'named_llm',
      native,
      (n) => ({
        ok: true,
      }),
      passthrough,
      {
        modelName: 'gpt-4-turbo',
      },
    );
    assert.deepEqual(result, {
      ok: true,
    });
  });

  it('awaits async llm functions before re-encoding responses', async () => {
    const native = makeNative();
    const result = await typedLlmExecute('async_codec_llm', native, async () => new Point(8, 13), pointCodec);
    assert.ok(result instanceof Point);
    assert.equal(result.x, 8);
    assert.equal(result.y, 13);
  });

  it('passes OpenAI Chat request codec through typedLlmExecute', async () => {
    const scope = pushScope('typed-chat-codec', ScopeType.Agent, null, null);
    let interceptedAnnotated = null;
    registerLlmRequestIntercept('typed_chat_codec_req', 10, false, ({ request, annotated }) => {
      interceptedAnnotated = annotated;
      return {
        request,
        annotated,
      };
    });

    try {
      const result = await typedLlmExecute(
        'typed_chat_codec_llm',
        makeOpenAIChatRequest(),
        () => ({
          ok: true,
        }),
        new JsonPassthrough(),
        {
          codec: new OpenAIChatCodec(),
        },
      );

      assert.deepEqual(result, {
        ok: true,
      });
      assert.notEqual(interceptedAnnotated, null);
      assert.equal(interceptedAnnotated.model, 'gpt-4o-mini');
      assert.equal(interceptedAnnotated.messages[0].role, 'user');
      assert.equal(interceptedAnnotated.messages[0].content, 'Hello from chat');
    } finally {
      deregisterLlmRequestIntercept('typed_chat_codec_req');
      popScope(scope);
    }
  });

  it('decodes Anthropic responses through typedLlmExecute responseCodec', async () => {
    const scope = pushScope('typed-anthropic-codec', ScopeType.Agent, null, null);
    const events = [];
    registerSubscriber('typed_anthropic_codec_sub', (event) => events.push(event));

    try {
      const result = await typedLlmExecute(
        'typed_anthropic_codec_llm',
        makeAnthropicRequest(),
        () => ({
          id: 'msg_123',
          model: 'claude-3-5-sonnet',
          content: [
            {
              type: 'text',
              text: 'Anthropic hello',
            },
          ],
          stop_reason: 'end_turn',
          usage: {
            input_tokens: 5,
            output_tokens: 3,
          },
        }),
        new JsonPassthrough(),
        {
          responseCodec: new AnthropicMessagesCodec(),
        },
      );

      assert.equal(result.content[0].text, 'Anthropic hello');

      const deadline = Date.now() + 2000;
      while (
        !events.some(
          (event) =>
            event.kind === 'scope' &&
            event.category === 'llm' &&
            event.scope_category === 'end' &&
            event.name === 'typed_anthropic_codec_llm',
        ) &&
        Date.now() < deadline
      ) {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }

      const endEvent = events.find(
        (event) =>
          event.kind === 'scope' &&
          event.category === 'llm' &&
          event.scope_category === 'end' &&
          event.name === 'typed_anthropic_codec_llm',
      );
      assert.equal(endEvent.category_profile.annotated_response.model, 'claude-3-5-sonnet');
      assert.equal(endEvent.category_profile.annotated_response.message, 'Anthropic hello');
      assert.equal(endEvent.category_profile.annotated_response.finish_reason, 'complete');
    } finally {
      deregisterSubscriber('typed_anthropic_codec_sub');
      popScope(scope);
    }
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
    const collector = (chunk) => collected.push(chunk);
    const finalizer = () => ({
      chunks: collected,
    });

    const stream = await typedLlmStreamExecute(
      'stream_llm',
      native,
      emitHelloWorldStream,
      collector,
      finalizer,
      passthrough,
      passthrough,
    );

    const chunks = [];
    let chunk;
    while ((chunk = await stream.next()) !== null) {
      chunks.push(chunk);
    }

    assert.equal(chunks.length, 2);
    assert.deepEqual(chunks[0], {
      token: 'hello',
    });
    assert.equal(collected.length, 2);
  });

  it('stream with envelopeCodec for chunks and response', async () => {
    const native = makeNative();

    // The collector receives typed (unwrapped) values thanks to chunkCodec.fromJson
    const collected = [];
    const collector = (chunk) => collected.push(chunk);
    // Finalizer returns a typed value; typedLlmStreamExecute wraps via responseCodec.toJson
    const finalizer = () => collected.join(',');

    const stream = await typedLlmStreamExecute(
      'env_stream',
      native,
      emitEnvelopeStream,
      collector,
      finalizer,
      envelopeCodec,
      envelopeCodec,
    );

    const chunks = [];
    let chunk;
    while ((chunk = await stream.next()) !== null) {
      chunks.push(chunk);
    }

    // Chunks should be the JSON-encoded form (envelopeCodec.toJson wraps as { value })
    assert.equal(chunks.length, 3);
    assert.deepEqual(chunks[0], {
      value: 'alpha',
    });
    assert.deepEqual(chunks[1], {
      value: 'beta',
    });
    assert.deepEqual(chunks[2], {
      value: 'gamma',
    });

    // Collector receives decoded (unwrapped) values via chunkCodec.fromJson
    assert.deepEqual(collected, ['alpha', 'beta', 'gamma']);
  });

  it('stream with pointCodec for chunks and response', async () => {
    const native = makeNative();

    const collected = [];
    const func = async function* (n) {
      yield new Point(1, 2);
      yield new Point(3, 4);
    };
    const collector = (chunk) => collected.push(chunk);
    const finalizer = () =>
      new Point(
        collected.reduce((s, p) => s + p.x, 0),
        collected.reduce((s, p) => s + p.y, 0),
      );

    const stream = await typedLlmStreamExecute(
      'point_stream',
      native,
      func,
      collector,
      finalizer,
      pointCodec,
      pointCodec,
    );

    const chunks = [];
    let chunk;
    while ((chunk = await stream.next()) !== null) {
      chunks.push(chunk);
    }

    // Raw chunks from the stream are JSON (pointCodec.toJson output)
    assert.equal(chunks.length, 2);
    assert.deepEqual(chunks[0], {
      x: 1,
      y: 2,
    });
    assert.deepEqual(chunks[1], {
      x: 3,
      y: 4,
    });

    // Collector receives decoded Point instances via pointCodec.fromJson
    assert.equal(collected.length, 2);
    assert.ok(collected[0] instanceof Point);
    assert.equal(collected[0].x, 1);
    assert.equal(collected[0].y, 2);
    assert.ok(collected[1] instanceof Point);
    assert.equal(collected[1].x, 3);
    assert.equal(collected[1].y, 4);
  });

  it('supports OpenAI Responses request and response codecs in typedLlmStreamExecute', async () => {
    const scope = pushScope('typed-responses-stream-codec', ScopeType.Agent, null, null);
    let interceptedAnnotated = null;
    const events = [];
    registerLlmRequestIntercept('typed_responses_stream_req', 10, false, ({ request, annotated }) => {
      interceptedAnnotated = annotated;
      return {
        request,
        annotated,
      };
    });
    registerSubscriber('typed_responses_stream_sub', (event) => events.push(event));

    try {
      const collected = [];
      const stream = await typedLlmStreamExecute(
        'typed_responses_stream_llm',
        makeOpenAIResponsesRequest(),
        async function* () {
          yield {
            type: 'message',
            delta: 'hello',
          };
          yield {
            type: 'message',
            delta: ' world',
          };
        },
        (chunk) => collected.push(chunk),
        () => ({
          id: 'resp_123',
          model: 'gpt-4.1-mini',
          status: 'completed',
          output: [
            {
              type: 'message',
              role: 'assistant',
              content: [
                {
                  type: 'output_text',
                  text: 'hello world',
                },
              ],
            },
          ],
          usage: {
            input_tokens: 4,
            output_tokens: 2,
            total_tokens: 6,
          },
        }),
        new JsonPassthrough(),
        new JsonPassthrough(),
        {
          codec: new OpenAIResponsesCodec(),
          responseCodec: new OpenAIResponsesCodec(),
        },
      );

      const chunks = [];
      let chunk;
      while ((chunk = await stream.next()) !== null) {
        chunks.push(chunk);
      }

      assert.equal(chunks.length, 2);
      assert.equal(collected.length, 2);
      assert.notEqual(interceptedAnnotated, null);
      assert.equal(interceptedAnnotated.model, 'gpt-4.1-mini');
      assert.equal(interceptedAnnotated.messages[0].role, 'system');
      assert.equal(interceptedAnnotated.messages[1].role, 'user');

      const deadline = Date.now() + 2000;
      while (
        !events.some(
          (event) =>
            event.kind === 'scope' &&
            event.category === 'llm' &&
            event.scope_category === 'end' &&
            event.name === 'typed_responses_stream_llm',
        ) &&
        Date.now() < deadline
      ) {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }

      const endEvent = events.find(
        (event) =>
          event.kind === 'scope' &&
          event.category === 'llm' &&
          event.scope_category === 'end' &&
          event.name === 'typed_responses_stream_llm',
      );
      assert.equal(endEvent.category_profile.annotated_response.model, 'gpt-4.1-mini');
      assert.equal(endEvent.category_profile.annotated_response.message, 'hello world');
      assert.equal(endEvent.category_profile.annotated_response.finish_reason, 'complete');
    } finally {
      deregisterSubscriber('typed_responses_stream_sub');
      deregisterLlmRequestIntercept('typed_responses_stream_req');
      popScope(scope);
    }
  });

  it('treats response codec decode failures as non-fatal in typedLlmStreamExecute', async () => {
    const scope = pushScope('typed-stream-bad-response-codec', ScopeType.Agent, null, null);
    const events = [];
    registerSubscriber('typed_stream_bad_response_codec_sub', (event) => events.push(event));

    try {
      const stream = await typedLlmStreamExecute(
        'typed_stream_bad_response_codec_llm',
        makeOpenAIChatRequest(),
        async function* () {
          yield {
            token: 'ok',
          };
        },
        () => {},
        () => ({
          malformed: true,
        }),
        new JsonPassthrough(),
        new JsonPassthrough(),
        {
          responseCodec: {
            decodeResponse() {
              throw new Error('bad response codec');
            },
          },
        },
      );

      while ((await stream.next()) !== null) {
        // Drain stream
      }

      const deadline = Date.now() + 2000;
      while (
        !events.some(
          (event) =>
            event.kind === 'scope' &&
            event.category === 'llm' &&
            event.scope_category === 'end' &&
            event.name === 'typed_stream_bad_response_codec_llm',
        ) &&
        Date.now() < deadline
      ) {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }

      const endEvent = events.find(
        (event) =>
          event.kind === 'scope' &&
          event.category === 'llm' &&
          event.scope_category === 'end' &&
          event.name === 'typed_stream_bad_response_codec_llm',
      );
      assert.equal(endEvent.category_profile?.annotated_response, undefined);
    } finally {
      deregisterSubscriber('typed_stream_bad_response_codec_sub');
      popScope(scope);
    }
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
      (p) => p.x + p.y, // receives Point, returns number
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
      (val) => new Point(val, val * 2), // receives number, returns Point
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
    const result = await typedLlmExecute('sync_llm', native, (n) => new Point(100, 200), pointCodec);
    assert.ok(result instanceof Point);
    assert.equal(result.x, 100);
    assert.equal(result.y, 200);
  });

  it('sync function with envelope codec', async () => {
    const native = makeNative();
    const result = await typedLlmExecute('sync_env_llm', native, (n) => 'sync-response', envelopeCodec);
    assert.equal(result, 'sync-response');
  });
});
