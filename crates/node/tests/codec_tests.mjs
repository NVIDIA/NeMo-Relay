// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const { pushScope, popScope, llmCallExecute, registerLlmRequestIntercept, deregisterLlmRequestIntercept, ScopeType } =
  lib;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeRequest() {
  return {
    headers: {},
    content: {
      messages: [
        {
          role: 'user',
          content: 'Hello',
        },
      ],
      model: 'gpt-4',
      temperature: 0.7,
      custom_field: 'should_roundtrip',
    },
  };
}

// llmCallExecute signature: (name, request, func, handle, attributes, data, metadata, model_name, codec_decode, codec_encode)
// Pass null for unused positional args, codec_decode is the 9th arg, codec_encode is the 10th arg.
function execWithCodec(name, request, func, decodeFn, encodeFn) {
  return llmCallExecute(name, request, func, null, null, null, null, null, decodeFn, encodeFn);
}

function execNoCodec(name, request, func) {
  return llmCallExecute(name, request, func, null, null, null, null, null, null, null);
}

// A mock decode function: extracts messages, model, params from content
function mockDecode(request) {
  const c = request.content;
  return {
    messages: c.messages || [],
    model: c.model || null,
    params: {
      temperature: c.temperature,
    },
    tools: null,
    tool_choice: null,
    extra: {},
  };
}

// A mock encode function: merges annotated back into original
function mockEncode({ annotated, original }) {
  const content = {
    ...original.content,
  };
  content.messages = annotated.messages;
  if (annotated.model) content.model = annotated.model;
  if (annotated.params) Object.assign(content, annotated.params);
  if (annotated.extra) Object.assign(content, annotated.extra);
  return {
    headers: original.headers,
    content,
  };
}

// ===========================================================================
// Codec pipeline integration (direct codec passing)
// ===========================================================================

describe('Codec pipeline integration', () => {
  it('intercept receives annotated when codec is active', async () => {
    let receivedAnnotated = null;
    registerLlmRequestIntercept('codec-pipeline-test', 10, false, ({ name, request, annotated }) => {
      receivedAnnotated = annotated;
      return {
        request,
        annotated,
      };
    });

    const handle = pushScope('pipeline-scope', ScopeType.Agent);
    try {
      await execWithCodec(
        'test-llm',
        makeRequest(),
        async (req) => ({
          choices: [
            {
              message: {
                content: 'Hi',
              },
            },
          ],
        }),
        mockDecode,
        mockEncode,
      );

      assert.notEqual(receivedAnnotated, null, 'annotated should not be null');
      assert.equal(receivedAnnotated.model, 'gpt-4');
      assert.equal(receivedAnnotated.messages.length, 1);
      assert.equal(receivedAnnotated.messages[0].role, 'user');
    } finally {
      popScope(handle);
      deregisterLlmRequestIntercept('codec-pipeline-test');
    }
  });

  it('intercept modifications to annotated flow through encode', async () => {
    let executedContent = null;
    registerLlmRequestIntercept('modify-test', 10, false, ({ name, request, annotated }) => {
      if (annotated) {
        annotated = {
          ...annotated,
          model: 'gpt-4-turbo',
        };
      }
      return {
        request,
        annotated,
      };
    });

    const handle = pushScope('modify-scope', ScopeType.Agent);
    try {
      await execWithCodec(
        'test-llm',
        makeRequest(),
        async (req) => {
          executedContent = req.content;
          return {
            choices: [],
          };
        },
        mockDecode,
        mockEncode,
      );

      assert.equal(executedContent.model, 'gpt-4-turbo', 'model should be modified by intercept');
      assert.equal(executedContent.custom_field, 'should_roundtrip', 'unmodeled fields must survive');
    } finally {
      popScope(handle);
      deregisterLlmRequestIntercept('modify-test');
    }
  });

  it('rejects raw request content edits before provider execution', async () => {
    let providerCalled = false;
    registerLlmRequestIntercept('raw-content-test', 10, false, ({ request, annotated }) => ({
      request: {
        ...request,
        content: {
          ...request.content,
          model: 'raw-model-edit',
        },
      },
      annotated,
    }));

    const handle = pushScope('raw-content-scope', ScopeType.Agent);
    try {
      await assert.rejects(
        () =>
          execWithCodec(
            'test-llm',
            makeRequest(),
            async () => {
              providerCalled = true;
              return { choices: [] };
            },
            mockDecode,
            mockEncode,
          ),
        /request\.content/,
      );
      assert.equal(providerCalled, false);
    } finally {
      popScope(handle);
      deregisterLlmRequestIntercept('raw-content-test');
    }
  });

  it('different codec functions produce different results', async () => {
    let usedModel = null;

    const decodeB = (req) => ({
      messages: [],
      model: 'from-b',
      params: null,
      tools: null,
      tool_choice: null,
      extra: {},
    });
    const encodeB = ({ annotated, original }) => ({
      headers: original.headers,
      content: {
        ...original.content,
        model: annotated.model,
      },
    });

    registerLlmRequestIntercept('override-test', 10, false, ({ name, request, annotated }) => {
      usedModel = annotated ? annotated.model : null;
      return {
        request,
        annotated,
      };
    });

    const handle = pushScope('override-scope', ScopeType.Agent);
    try {
      // Passing decodeB/encodeB should use codec-b's decode.
      await execWithCodec(
        'test-llm',
        makeRequest(),
        async (req) => ({
          choices: [],
        }),
        decodeB,
        encodeB,
      );

      assert.equal(usedModel, 'from-b', 'direct codec functions select codec-b decode');
    } finally {
      popScope(handle);
      deregisterLlmRequestIntercept('override-test');
    }
  });
});
