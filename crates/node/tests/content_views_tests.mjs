// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  codecs,
  OpenAIChatCodec,
  OpenAIResponsesCodec,
  AnthropicMessagesCodec,
  GeminiGenerateContentCodec,
  OCIGenAIChatCodec,
} = require('../dist/index.js');

const parts = () => [
  { type: 'text', text: ' a ' },
  { type: 'refusal', refusal: '' },
  { type: 'tool_use', id: 'inline', name: 'lookup', input: { q: [1] } },
  { type: 'tool_result', tool_use_id: 'inline', content: { text: 'hidden' }, is_error: true },
  { type: 'image_url', image_url: { url: 'https://example.com/i.png' } },
  { type: 'image', image: { text: 'hidden' } },
  { type: 'audio', audio: { data: 'sound' } },
  { type: 'file', file: { id: 'file' } },
  { type: 'provider_native', provider: 'custom', kind: 'extension', value: { refusal: 'fallback' } },
  { type: 'provider_native', provider: 'custom', kind: 'thinking', value: { text: 'hidden' } },
  { type: 'provider_native', provider: 'custom', kind: 'extension', value: { nested: { text: 'hidden' } } },
  { type: 'text', text: 'z\n' },
];

describe('normalized content snapshots', () => {
  for (const [Codec, requestBody, responseBody] of [
    [
      OpenAIChatCodec,
      { messages: [{ role: 'user', content: 'question' }] },
      {
        choices: [
          {
            message: {
              content: 'answer',
              tool_calls: [{ id: 'c', type: 'function', function: { name: 'lookup', arguments: '{"q":1}' } }],
            },
          },
        ],
      },
    ],
    [
      OpenAIResponsesCodec,
      { input: [{ role: 'user', content: [{ type: 'input_text', text: 'question' }] }] },
      {
        output: [
          { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'answer' }] },
          { type: 'function_call', call_id: 'c', name: 'lookup', arguments: '{"q":1}' },
        ],
      },
    ],
    [
      AnthropicMessagesCodec,
      { messages: [{ role: 'user', content: [{ type: 'text', text: 'question' }] }] },
      {
        type: 'message',
        content: [
          { type: 'text', text: 'answer' },
          { type: 'tool_use', id: 'c', name: 'lookup', input: { q: 1 } },
        ],
      },
    ],
    [
      GeminiGenerateContentCodec,
      { contents: [{ role: 'user', parts: [{ text: 'question' }] }] },
      {
        candidates: [
          {
            content: {
              role: 'model',
              parts: [{ text: 'answer' }, { functionCall: { name: 'lookup', args: { q: 1 } } }],
            },
            finishReason: 'STOP',
          },
        ],
      },
    ],
    [
      OCIGenAIChatCodec,
      {
        chatRequest: {
          apiFormat: 'GENERIC',
          messages: [{ role: 'USER', content: [{ type: 'TEXT', text: 'question' }] }],
        },
      },
      {
        chatResponse: {
          apiFormat: 'GENERIC',
          choices: [
            {
              message: {
                role: 'ASSISTANT',
                content: [{ type: 'TEXT', text: 'answer' }],
                toolCalls: [{ id: 'c', type: 'FUNCTION', name: 'lookup', arguments: '{"q":1}' }],
              },
            },
          ],
        },
      },
    ],
  ]) {
    it(`inspects normalized request and response data from ${Codec.name}`, () => {
      const codec = new Codec();
      const request = codec.decode({ headers: {}, content: requestBody });
      assert.equal(codecs.lastMessage(request).text(), 'question');
      assert.equal(codecs.lastUserMessage(request), 'question');
      const response = codec.decodeResponse(responseBody);
      const output = codecs.output(response);
      assert.equal(output.text(), 'answer');
      const calls = [...output.parts()].filter((entry) => entry.kind === 'tool_call');
      assert.equal(calls.length, 1);
      assert.equal(calls[0].toolCall.name, 'lookup');
      assert.equal(calls[0].toolCall.arguments.kind, 'json');
      assert.deepEqual(calls[0].toolCall.arguments.value, { q: 1 });
    });
  }

  it('exposes Anthropic text blocks as already combined by normalization', () => {
    const response = new AnthropicMessagesCodec().decodeResponse({
      type: 'message',
      content: [
        { type: 'text', text: 'first' },
        { type: 'tool_use', id: 'c', name: 'lookup', input: {} },
        { type: 'text', text: 'second' },
      ],
    });
    const output = codecs.output(response);
    assert.deepEqual([...output.textParts()], ['first\nsecond']);
    assert.equal([...output.parts()].length, 2);
  });

  it('retains empty selected items and ignores unrelated annotation metadata', () => {
    const request = {
      instructions: [],
      messages: [{ role: 'user', content: 'earlier' }, { role: 'assistant', content: [] }],
      get metadata() { throw new Error('unrelated metadata was read'); },
    };
    const selected = codecs.lastMessage(request);
    const instructions = codecs.instructionsView(request);
    assert.deepEqual([...selected.parts()], []);
    assert.deepEqual([...instructions.parts()], []);
    assert.equal(selected.text(), null);
    assert.equal(codecs.systemPrompt(request), null);
    request.messages[1].content.push({ type: 'text', text: 'later' });
    request.instructions.push({ type: 'text', text: 'later policy' });
    assert.deepEqual([...selected.parts()], []);
    assert.deepEqual([...instructions.parts()], []);
  });

  it('selects exact history items and instructions with original indices', () => {
    assert.equal(codecs.lastMessage({}), null);
    assert.equal(codecs.instructionsView({}), null);
    const request = {
      instructions: '',
      messages: [
        { role: 'user', content: 'earlier' },
        { role: 'assistant', content: [{ type: 'image', image: {} }] },
      ],
    };
    assert.equal(codecs.lastMessage(request).text(), null);
    assert.equal(codecs.message(request, -2).text(), 'earlier');
    assert.equal(codecs.instructionsView(request).text(), '');
    assert.equal(codecs.systemPrompt(request), '');
    assert.deepEqual([...codecs.lastMessage(request).parts()][0].location, {
      root: 'request_message',
      messageIndex: 1,
      field: 'content',
      index: 0,
    });
    assert.deepEqual([...codecs.instructionsView(request).parts()][0].location, {
      root: 'content',
      messageIndex: null,
      field: 'content',
      index: null,
    });
    for (const index of [2, -3, 0.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
      assert.throws(() => codecs.message(request, index), RangeError);
    }
    assert.throws(() => codecs.message({}, 0), RangeError);
  });

  it('joins only eligible text/refusals while preserving empty strings and whitespace', () => {
    const view = codecs.MessageView.fromContent(parts());
    assert.equal(view.text(), ' a \n\nfallback\nz\n');
    assert.deepEqual([...view.textParts()], [' a ', '', 'fallback', 'z\n']);
    assert.equal(codecs.MessageView.fromContent('').text(), '');
    assert.equal(codecs.MessageView.fromContent('  ').text(), '  ');
    assert.equal(codecs.MessageView.fromContent([]).text(), null);
  });

  it('keeps mixed tool entries typed and appends separate calls after content', () => {
    const view = codecs.MessageView.fromMessage({
      role: 'assistant',
      content: parts(),
      tool_calls: [{ id: 'separate', type: 'function', function: { name: 'search', arguments: '{not JSON}' } }],
    });
    const entries = [...view.parts()];
    assert.deepEqual(
      entries.slice(0, 8).map((entry) => entry.kind),
      ['text', 'refusal', 'tool_call', 'tool_result', 'image_url', 'image', 'audio', 'file'],
    );
    assert.deepEqual(entries[2].toolCall.arguments.value, { q: [1] });
    assert.equal(entries[2].toolCall.arguments.kind, 'json');
    assert.equal(entries[2].toolCall.id, 'inline');
    assert.equal(entries[2].toolCall.callId, null);
    assert.deepEqual(entries[3].toolResult.payload.value, { text: 'hidden' });
    assert.equal(entries[3].toolResult.isError, true);
    const last = entries.at(-1);
    assert.equal(last.kind, 'tool_call');
    assert.equal(last.toolCall.arguments.kind, 'string');
    assert.equal(last.toolCall.arguments.value, '{not JSON}');
    assert.deepEqual(last.location, { root: 'message', messageIndex: null, field: 'tool_calls', index: 0 });
  });

  for (const [message, kind, value, field] of [
    [{ role: 'tool', tool_call_id: 'c', content: 'hidden' }, 'content', 'hidden', 'content'],
    [
      { role: 'tool', tool_call_id: 'c', content: [{ type: 'text', text: 'hidden' }] },
      'content',
      [{ type: 'text', text: 'hidden' }],
      'content',
    ],
    [{ role: 'function', name: 'f', content: 'hidden' }, 'string', 'hidden', 'content'],
    [{ role: 'function', name: 'f', content: null }, 'null', null, 'content'],
    [
      { role: 'tool_result', id: 'item', call_id: 'c', output: { text: 'hidden' } },
      'json',
      { text: 'hidden' },
      'output',
    ],
  ]) {
    it(`preserves whole ${message.role} payloads as ${kind}`, () => {
      const view = codecs.MessageView.fromMessage(message);
      const entry = [...view.parts()][0];
      assert.equal(entry.kind, 'tool_result');
      assert.equal(entry.toolResult.payload.kind, kind);
      assert.deepEqual(entry.toolResult.payload.value, value);
      assert.equal(entry.toolResult.callId, message.role === 'function' ? null : 'c');
      assert.equal(entry.toolResult.name, message.role === 'function' ? 'f' : null);
      assert.equal(entry.location.field, field);
      assert.equal(view.text(), null);
    });
  }

  it('preserves top-level call identifiers and JSON-string arguments', () => {
    const entry = [
      ...codecs.MessageView.fromMessage({
        role: 'tool_call',
        id: 'item',
        call_id: 'correlation',
        name: 'f',
        arguments: '{invalid}',
      }).parts(),
    ][0];
    assert.equal(entry.toolCall.id, 'item');
    assert.equal(entry.toolCall.callId, 'correlation');
    assert.equal(entry.toolCall.arguments.kind, 'json');
    assert.equal(entry.toolCall.arguments.value, '{invalid}');
    assert.equal(entry.location.field, 'tool_call_item');
  });

  it('exposes media and native values independently without recursively scanning them', () => {
    const entries = [...codecs.MessageView.fromContent(parts()).parts()];
    assert.deepEqual(entries[4].value, { url: 'https://example.com/i.png' });
    assert.deepEqual(entries[5].value, { text: 'hidden' });
    assert.deepEqual(entries[6].value, { data: 'sound' });
    assert.deepEqual(entries[7].value, { id: 'file' });
    assert.equal(entries[8].provider, 'custom');
    assert.equal(entries[8].nativeKind, 'extension');
    const value = entries[8].value;
    value.refusal = 'changed';
    assert.deepEqual(entries[8].value, { refusal: 'fallback' });
  });

  it('keeps selected Rust data alive and independent of original and returned payloads', () => {
    let request = { messages: [{ role: 'assistant', content: parts() }] };
    let view = codecs.lastMessage(request);
    let iterator = view.parts();
    let text = view.textParts();
    request.messages[0].content[2].input.q.push(2);
    request.messages = [];
    request = null;
    view = null;
    iterator.next();
    iterator.next();
    const call = iterator.next().value.toolCall;
    iterator = null;
    const value = call.arguments.value;
    value.q.push(99);
    assert.deepEqual(call.arguments.value, { q: [1] });
    assert.deepEqual([...text], [' a ', '', 'fallback', 'z\n']);
    text = null;
    assert(Object.isFrozen(call));
    assert(Object.isFrozen(call.arguments));
    assert.throws(() => {
      call.name = 'changed';
    }, TypeError);
    const entry = [...codecs.output({ message: 'hello' }).parts()][0];
    assert(Object.isFrozen(entry));
    assert(Object.isFrozen(entry.location));
    assert.throws(() => {
      entry.location.index = 9;
    }, TypeError);
  });

  it('excludes unrelated history and response metadata from snapshot conversion', () => {
    const request = { messages: [() => 'unselected', { role: 'user', content: 'selected' }], extra: new Date() };
    assert.equal(codecs.lastMessage(request).text(), 'selected');
    const response = { message: 'output', usage: () => {}, extra: new Date() };
    assert.equal(codecs.output(response).text(), 'output');
  });

  it('handles empty and tool-only output with response locations', () => {
    assert.deepEqual([...codecs.output({}).parts()], []);
    assert.equal(codecs.responseText({}), null);
    const response = { tool_calls: [{ id: 'c', name: 'lookup', arguments: { q: 'Boston' } }] };
    const output = codecs.output(response);
    const entry = [...output.parts()][0];
    assert.equal(output.text(), null);
    assert.equal(entry.toolCall.arguments.kind, 'json');
    assert.deepEqual(entry.toolCall.arguments.value, { q: 'Boston' });
    assert.deepEqual(entry.location, { root: 'response', messageIndex: null, field: 'tool_calls', index: 0 });
  });

  it('extracts decoded Chat and Responses refusals without changing normalized data', () => {
    for (const [codec, raw] of [
      [new OpenAIChatCodec(), { choices: [{ message: { role: 'assistant', content: 'yes', refusal: 'no' } }] }],
      [
        new OpenAIResponsesCodec(),
        {
          output: [
            {
              type: 'message',
              role: 'assistant',
              content: [
                { type: 'output_text', text: 'yes' },
                { type: 'refusal', refusal: 'no' },
              ],
            },
          ],
        },
      ],
    ]) {
      const response = codec.decodeResponse(raw);
      const before = structuredClone(response);
      assert.equal(codecs.responseText(response), 'yes\nno');
      const output = codecs.output(response);
      assert.deepEqual(
        [...output.parts()].map((entry) => entry.kind),
        ['text', 'refusal'],
      );
      assert.equal([...output.parts()][0].location.field, 'message');
      assert.deepEqual(response, before);
      response.message = 'changed';
      assert.equal(output.text(), 'yes\nno');
    }
  });

  for (const [provider, kind, role] of [
    ['openai_chat', 'user', 'user'],
    ['anthropic_messages', 'user', 'user'],
    ['openai_responses', 'message', 'user'],
    ['openai_chat', 'system', 'system'],
    ['openai_responses', 'message', 'developer'],
  ]) {
    it(`recognizes native ${provider}/${role} selection without backward fallback`, () => {
      const native = { role: 'provider_native', provider, kind, value: { role, content: 'known' } };
      const view = codecs.MessageView.fromMessage(native);
      assert.equal(view.text(), 'known');
      assert.equal([...view.parts()][0].location.field, 'native_content');
      native.value.content = [];
      const earlier = { role, content: 'earlier' };
      const request = { messages: role === 'user' ? [earlier, native] : [native, earlier] };
      assert.equal(role === 'user' ? codecs.lastUserMessage(request) : codecs.systemPrompt(request), null);
      request.instructions = [];
      assert.equal(codecs.systemPrompt(request), null);
    });
  }

  it('does not use native extension text when a recognized message has missing content', () => {
    const native = {
      role: 'provider_native',
      provider: 'openai_chat',
      kind: 'user',
      value: { role: 'user', text: 'hidden' },
    };
    assert.equal(codecs.MessageView.fromMessage(native).text(), null);
    const request = { messages: [{ role: 'user', content: 'earlier' }, native] };
    assert.equal(codecs.lastUserMessage(request), null);
    const responses = {
      ...native,
      provider: 'openai_responses',
      kind: 'message',
      value: {
        role: 'user',
        content: [
          { type: 'input_text', text: 'known' },
          { type: 'tool_result', text: 'hidden' },
        ],
      },
    };
    assert.equal(codecs.MessageView.fromMessage(responses).text(), 'known');
  });

  it('retains independent iterators and permanent exhaustion', () => {
    const view = codecs.MessageView.fromContent([
      { type: 'text', text: 'a' },
      { type: 'refusal', refusal: 'b' },
    ]);
    const first = view.parts(),
      second = view.parts(),
      text = view.textParts();
    assert.equal(first[Symbol.iterator](), first);
    assert.equal(first.next().value.text, 'a');
    assert.equal(second.next().value.text, 'a');
    assert.deepEqual([...text], ['a', 'b']);
    assert.equal(first.next().value.text, 'b');
    assert.equal(first.next().done, true);
    assert.equal(first.next().done, true);
  });

  it('rejects invalid selected input without invoking accessors', () => {
    assert.throws(() => codecs.MessageView.fromContent({}), /normalized content/);
    assert.throws(() => codecs.MessageView.fromMessage({ role: 'unknown' }), /normalized message/);
    assert.throws(() => codecs.output({ message: Infinity }), /finite|JSON|number/i);
    assert.throws(() => codecs.lastMessage({ messages: {} }), TypeError);
    assert.throws(() => codecs.lastMessage(null), TypeError);
    const request = {
      get messages() {
        throw new Error('getter was invoked');
      },
    };
    assert.throws(() => codecs.lastMessage(request), /data properties/);
    const messages = [];
    Object.defineProperty(messages, 0, {
      get() {
        throw new Error('getter was invoked');
      },
    });
    assert.throws(() => codecs.lastUserMessage({ messages }), /data entries/);
    assert.throws(() => codecs.systemPrompt({ messages }), /data entries/);
    const cyclic = [];
    cyclic.push(cyclic);
    assert.throws(() => codecs.MessageView.fromContent(cyclic), /cyclic|cycle|JSON/i);
  });
});
