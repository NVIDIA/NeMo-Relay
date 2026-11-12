// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const lib = require('../dist/index.js');
const typed = require('../dist/typed.js');

for (const kind of ['tool', 'toolAsync', 'llm', 'llmAsync', 'stream', 'typedTool', 'typedLlm', 'typedStream']) {
  it(`per-call stage combinations: ${kind}`, async () => {
    const calls = [];
    const isTool = kind.toLowerCase().includes('tool');
    const isStream = kind.toLowerCase().includes('stream');
    const prefix = isTool ? 'Tool' : 'Llm';
    const executionPrefix = isStream ? 'LlmStream' : prefix;
    lib[`register${prefix}ConditionalExecutionGuardrail`]('options', 0, () => {
      calls.push('conditional');
      return null;
    });
    lib[`register${prefix}RequestIntercept`]('options', 0, false, () => {
      calls.push('request');
      return isTool ? { rewritten: true } : { request: { headers: {}, content: { rewritten: true } } };
    });
    lib[`register${executionPrefix}ExecutionIntercept`]('options', 0, async (...args) => {
      calls.push('execution');
      return isTool ? args[1](args[0].args) : args[2](args[0]);
    });
    try {
      for (let mask = 0; mask < 8; mask++) {
        calls.length = 0;
        const executionOptions = {
          conditionalExecution: !(mask & 1),
          requestIntercepts: !(mask & 2),
          executionIntercepts: !(mask & 4),
        };
        const expected = executionOptions.requestIntercepts ? { rewritten: true } : { original: true };
        const callback = (arg) => {
          calls.push('callback');
          assert.deepEqual(isTool ? arg : arg.content, expected);
          return isTool ? { result: arg, annotation: { opaque: true } } : arg.content;
        };
        async function* stream(req) {
          yield callback(req);
        }
        const req = { headers: {}, content: { original: true } };
        const codec = new typed.JsonPassthrough();
        let result;
        if (kind === 'typedTool') {
          result = await typed.typedToolExecute('options-call', { original: true }, callback, codec, codec, {
            executionOptions,
          });
        } else if (kind === 'typedLlm') {
          result = await typed.typedLlmExecute('options-call', req, callback, codec, { executionOptions });
        } else if (kind === 'typedStream') {
          result = await typed.typedLlmStreamExecute(
            'options-call',
            req,
            stream,
            () => {},
            () => ({}),
            codec,
            codec,
            { executionOptions },
          );
        } else if (isTool) {
          const execute = kind === 'toolAsync' ? lib.toolCallExecuteAsync : lib.toolCallExecute;
          result = await execute(
            'options-call',
            { original: true },
            callback,
            null,
            null,
            null,
            null,
            null,
            executionOptions,
          );
        } else if (isStream) {
          result = await lib.llmStreamCallExecute(
            'options-call',
            req,
            stream,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            executionOptions,
          );
        } else {
          const execute = kind === 'llmAsync' ? lib.llmCallExecuteAsync : lib.llmCallExecute;
          result = await execute(
            'options-call',
            req,
            callback,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            executionOptions,
          );
        }
        if (isStream) {
          const chunks = [];
          for (;;) {
            const chunk = await result.next();
            if (chunk === null) break;
            chunks.push(chunk);
          }
          await result.close();
          assert.deepEqual(chunks, [expected]);
        } else if (isTool) {
          assert.deepEqual(result, { result: expected, annotation: { opaque: true } });
        } else {
          assert.deepEqual(result, expected);
        }
        assert.deepEqual(calls, [
          ...(executionOptions.conditionalExecution ? ['conditional'] : []),
          ...(executionOptions.requestIntercepts ? ['request'] : []),
          ...(executionOptions.executionIntercepts ? ['execution'] : []),
          'callback',
        ]);
      }
    } finally {
      lib[`deregister${prefix}ConditionalExecutionGuardrail`]('options');
      lib[`deregister${prefix}RequestIntercept`]('options');
      lib[`deregister${executionPrefix}ExecutionIntercept`]('options');
    }
  });
}
