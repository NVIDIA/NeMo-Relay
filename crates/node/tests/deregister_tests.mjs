// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const {
  deregisterToolSanitizeRequestGuardrail,
  deregisterToolSanitizeResponseGuardrail,
  deregisterToolConditionalExecutionGuardrail,
  deregisterToolRequestIntercept,
  deregisterToolResponseIntercept,
  deregisterToolExecutionIntercept,
  deregisterLlmSanitizeRequestGuardrail,
  deregisterLlmSanitizeResponseGuardrail,
  deregisterLlmConditionalExecutionGuardrail,
  deregisterLlmRequestIntercept,
  deregisterLlmResponseIntercept,
  deregisterLlmExecutionIntercept,
  deregisterLlmStreamResponseIntercept,
  deregisterLlmStreamExecutionIntercept,
  deregisterSubscriber,
} = lib;

// ===========================================================================
// Deregister nonexistent
// ===========================================================================

describe('Deregister nonexistent', () => {
  it('tool guardrails', () => {
    assert.equal(deregisterToolSanitizeRequestGuardrail('nx'), false);
    assert.equal(deregisterToolSanitizeResponseGuardrail('nx'), false);
    assert.equal(deregisterToolConditionalExecutionGuardrail('nx'), false);
  });

  it('tool intercepts', () => {
    assert.equal(deregisterToolRequestIntercept('nx'), false);
    assert.equal(deregisterToolResponseIntercept('nx'), false);
    assert.equal(deregisterToolExecutionIntercept('nx'), false);
  });

  it('llm guardrails', () => {
    assert.equal(deregisterLlmSanitizeRequestGuardrail('nx'), false);
    assert.equal(deregisterLlmSanitizeResponseGuardrail('nx'), false);
    assert.equal(deregisterLlmConditionalExecutionGuardrail('nx'), false);
  });

  it('llm intercepts', () => {
    assert.equal(deregisterLlmRequestIntercept('nx'), false);
    assert.equal(deregisterLlmResponseIntercept('nx'), false);
    assert.equal(deregisterLlmExecutionIntercept('nx'), false);
    assert.equal(deregisterLlmStreamResponseIntercept('nx'), false);
    assert.equal(deregisterLlmStreamExecutionIntercept('nx'), false);
  });

  it('subscriber', () => {
    assert.equal(deregisterSubscriber('nx'), false);
  });
});
