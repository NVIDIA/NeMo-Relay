// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const { ScopeType, LlmRequest, ScopeStack } = lib;

// ===========================================================================
// Type constants
// ===========================================================================

describe('Type constants', () => {
  it('exports canonical non-Js binding names', () => {
    assert.equal(typeof lib.ScopeStack, 'function');
    assert.equal(typeof lib.ScopeHandle, 'function');
    assert.equal(typeof lib.ToolHandle, 'function');
    assert.equal(typeof lib.LlmRequest, 'function');
    assert.equal(typeof lib.OpenAIChatCodec, 'function');
    assert.equal(typeof lib.OpenAIResponsesCodec, 'function');
    assert.equal(typeof lib.AnthropicMessagesCodec, 'function');
    assert.equal(typeof lib.OCIGenAIChatCodec, 'function');
    assert.equal(typeof lib.GeminiGenerateContentCodec, 'function');
  });

  it('scope type enum values', () => {
    assert.equal(ScopeType.Agent, 0);
    assert.equal(ScopeType.Function, 1);
    assert.equal(ScopeType.Tool, 2);
    assert.equal(ScopeType.Llm, 3);
    assert.equal(ScopeType.Retriever, 4);
    assert.equal(ScopeType.Embedder, 5);
    assert.equal(ScopeType.Reranker, 6);
    assert.equal(ScopeType.Guardrail, 7);
    assert.equal(ScopeType.Evaluator, 8);
    assert.equal(ScopeType.Custom, 9);
    assert.equal(ScopeType.Unknown, 10);
  });

  it('type-checks the public API fixtures', () => {
    execFileSync(
      process.execPath,
      [
        fileURLToPath(new URL('../../../node_modules/typescript/bin/tsc', import.meta.url)),
        '--noEmit',
        '--skipLibCheck',
        '--strict',
        '--target',
        'ES2022',
        '--module',
        'NodeNext',
        '--moduleResolution',
        'NodeNext',
        'tests/public_observability_api_fixture.ts',
        'tests/public_event_metadata_api_fixture.ts',
      ],
      {
        cwd: fileURLToPath(new URL('..', import.meta.url)),
        stdio: 'pipe',
      },
    );
  });
});

// ===========================================================================
// LlmRequest
// ===========================================================================

describe('LlmRequest', () => {
  it('construction and getters', () => {
    const req = new LlmRequest(
      {
        'Content-Type': 'application/json',
      },
      {
        model: 'gpt-4',
      },
    );
    assert.deepEqual(req.headers, {
      'Content-Type': 'application/json',
    });
    assert.deepEqual(req.content, {
      model: 'gpt-4',
    });
  });

  it('coerces non-object headers to an empty object', () => {
    const req = new LlmRequest(null, {
      model: 'gpt-4',
    });
    assert.deepEqual(req.headers, {});
    assert.deepEqual(req.content, {
      model: 'gpt-4',
    });
  });
});

describe('ScopeStack', () => {
  it('constructs a scope stack instance', () => {
    const stack = new ScopeStack();
    assert.ok(stack instanceof ScopeStack);
  });
});

// ===========================================================================
// OCIGenAIChatCodec
// ===========================================================================

describe('OCIGenAIChatCodec', () => {
  const { OCIGenAIChatCodec } = lib;

  const chatDetails = () => ({
    headers: {},
    content: {
      compartmentId: 'ocid1.compartment.oc1..example',
      servingMode: { servingType: 'ON_DEMAND', modelId: 'meta.llama-3.3-70b-instruct' },
      chatRequest: {
        apiFormat: 'GENERIC',
        messages: [
          { role: 'USER', content: [{ type: 'TEXT', text: 'My SSN is 111-22-3333.' }] },
        ],
        maxTokens: 600,
        seed: 7,
      },
    },
  });

  it('instantiates', () => {
    const codec = new OCIGenAIChatCodec();
    assert.ok(codec instanceof OCIGenAIChatCodec);
  });

  it('decode returns an AnnotatedLLMRequest with model and params', () => {
    const codec = new OCIGenAIChatCodec();
    const annotated = codec.decode(chatDetails());
    assert.equal(annotated.model, 'meta.llama-3.3-70b-instruct');
    assert.equal(annotated.messages.length, 1);
    assert.equal(annotated.messages[0].role, 'user');
    // Rust serializes GenerationParams fields in snake_case (max_tokens, not maxTokens)
    assert.equal(annotated.params.max_tokens, 600);
  });

  it('encode is an identity for an unedited annotation', () => {
    const codec = new OCIGenAIChatCodec();
    const req = chatDetails();
    const annotated = codec.decode(req);
    const reEncoded = codec.encode(annotated, req);
    assert.deepEqual(reEncoded.content, req.content);
  });

  it('encode applies edited messages and keeps unmodeled fields', () => {
    const codec = new OCIGenAIChatCodec();
    const req = chatDetails();
    const annotated = codec.decode(req);
    annotated.messages = [{ role: 'user', content: 'My SSN is [REDACTED].' }];
    const reEncoded = codec.encode(annotated, req);
    assert.deepEqual(reEncoded.content.chatRequest.messages[0].content, [
      { type: 'TEXT', text: 'My SSN is [REDACTED].' },
    ]);
    assert.equal(reEncoded.content.chatRequest.seed, 7, 'unmodeled fields must survive edits');
  });

  it('decodeResponse extracts text, finish reason, and usage', () => {
    const codec = new OCIGenAIChatCodec();
    const raw = {
      modelId: 'meta.llama-3.3-70b-instruct',
      chatResponse: {
        apiFormat: 'GENERIC',
        choices: [{
          index: 0,
          message: { role: 'ASSISTANT', content: [{ type: 'TEXT', text: 'Hello!' }] },
          finishReason: 'stop',
        }],
        usage: { promptTokens: 10, completionTokens: 5, totalTokens: 15 },
      },
    };
    const resp = codec.decodeResponse(raw);
    // message is a plain string (MessageContent::Text serializes to a string, not {text: ...})
    assert.equal(resp.message, 'Hello!');
    assert.equal(resp.finish_reason, 'complete');
    assert.equal(resp.model, 'meta.llama-3.3-70b-instruct');
    assert.equal(resp.usage?.prompt_tokens, 10);
  });
});

// ===========================================================================
// GeminiGenerateContentCodec
// ===========================================================================

describe('GeminiGenerateContentCodec', () => {
  const { GeminiGenerateContentCodec } = lib;

  it('instantiates', () => {
    const codec = new GeminiGenerateContentCodec();
    assert.ok(codec instanceof GeminiGenerateContentCodec);
  });

  it('decode returns an AnnotatedLLMRequest with messages and no params', () => {
    const codec = new GeminiGenerateContentCodec();
    const req = {
      headers: {},
      content: {
        contents: [
          { role: 'user', parts: [{ text: 'hello' }] },
          { role: 'model', parts: [{ text: 'hi' }] },
        ],
        systemInstruction: { parts: [{ text: 'Be helpful.' }] },
      },
    };
    const annotated = codec.decode(req);
    const msgs = annotated.messages;
    assert.equal(msgs.length, 3, 'system + user + model = 3 messages');
    assert.equal(msgs[0].role, 'system');
    assert.equal(msgs[1].role, 'user');
    assert.equal(msgs[2].role, 'assistant', 'model role must normalize to assistant');
  });

  it('decode captures generationConfig into params', () => {
    const codec = new GeminiGenerateContentCodec();
    const req = {
      headers: {},
      content: {
        contents: [{ role: 'user', parts: [{ text: 'hi' }] }],
        generationConfig: { temperature: 0.5, maxOutputTokens: 256 },
      },
    };
    const annotated = codec.decode(req);
    assert.ok(annotated.params !== null && annotated.params !== undefined, 'params must be set');
    assert.ok(Math.abs(annotated.params.temperature - 0.5) < 1e-6);
    // Rust serializes GenerationParams fields in snake_case (max_tokens, not maxTokens)
    assert.equal(annotated.params.max_tokens, 256);
  });

  it('encode round-trips extra fields', () => {
    const codec = new GeminiGenerateContentCodec();
    const req = {
      headers: {},
      content: {
        contents: [{ role: 'user', parts: [{ text: 'hi' }] }],
        safetySettings: [{ category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_NONE' }],
      },
    };
    const annotated = codec.decode(req);
    const reEncoded = codec.encode(annotated, req);
    assert.ok(
      Array.isArray(reEncoded.content.safetySettings),
      'safetySettings must survive encode round-trip',
    );
  });

  it('decodeResponse extracts text and finish reason', () => {
    const codec = new GeminiGenerateContentCodec();
    const raw = {
      candidates: [{
        content: { role: 'model', parts: [{ text: 'Hello!' }] },
        finishReason: 'STOP',
        index: 0,
      }],
      usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 2, totalTokenCount: 7 },
      modelVersion: 'gemini-2.0-flash',
    };
    const resp = codec.decodeResponse(raw);
    // message is a plain string (MessageContent::Text serializes to a string, not {text: ...})
    assert.equal(resp.message, 'Hello!');
    // finish_reason is snake_case; value matches FinishReason::Complete serialized as "complete"
    assert.equal(resp.finish_reason, 'complete');
    assert.equal(resp.model, 'gemini-2.0-flash');
    // usage fields are also snake_case
    assert.equal(resp.usage?.prompt_tokens, 5);
  });

  it('decodeResponse maps functionCall id correctly', () => {
    const codec = new GeminiGenerateContentCodec();
    const raw = {
      candidates: [{
        content: {
          role: 'model',
          parts: [{ functionCall: { id: 'call_xyz', name: 'my_fn', args: { x: 1 } } }],
        },
        finishReason: 'STOP',
        index: 0,
      }],
      usageMetadata: {},
    };
    const resp = codec.decodeResponse(raw);
    // tool_calls is snake_case
    assert.ok(Array.isArray(resp.tool_calls), 'tool_calls must be an array');
    assert.equal(resp.tool_calls[0].id, 'call_xyz', 'id must come from functionCall.id, not the function name');
    assert.equal(resp.tool_calls[0].name, 'my_fn');
    // Sanity: confirm the id is NOT the function name
    assert.notEqual(resp.tool_calls[0].id, 'my_fn', 'id must not be the function name');
  });

  it('decode throws for malformed Gemini requests', () => {
    const codec = new GeminiGenerateContentCodec();
    assert.throws(
      () => codec.decode({ headers: {}, content: { contents: 'not an array' } }),
      /contents must be an array/,
    );
  });

  it('encode throws for malformed annotated requests and codec failures', () => {
    const codec = new GeminiGenerateContentCodec();
    const original = {
      headers: {},
      content: {
        contents: [{ role: 'user', parts: [{ text: 'hi' }] }],
      },
    };
    assert.throws(
      () => codec.encode({ messages: 'not an array' }, original),
      /invalid AnnotatedLlmRequest/,
    );

    const annotated = codec.decode(original);
    annotated.messages.push({ role: 'developer', content: 'unsupported role' });
    assert.throws(
      () => codec.encode(annotated, original),
      /no Gemini equivalent/,
    );
  });

  it('decodeResponse throws for malformed Gemini responses', () => {
    const codec = new GeminiGenerateContentCodec();
    assert.throws(
      () => codec.decodeResponse({
        candidates: [{
          content: { role: 'model', parts: [{ text: 42 }] },
          finishReason: 'STOP',
        }],
      }),
      /parts.*text must be a string/,
    );
  });
});
