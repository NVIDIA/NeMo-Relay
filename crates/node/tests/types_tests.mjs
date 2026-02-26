import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const {
  ScopeType, JsLlmRequest,
  SCOPE_ATTR_PARALLEL, SCOPE_ATTR_RELOCATABLE,
  TOOL_ATTR_LOCAL, LLM_ATTR_STATELESS, LLM_ATTR_STREAMING,
} = lib;

// ===========================================================================
// Type constants
// ===========================================================================

describe('Type constants', () => {
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

  it('attribute constants', () => {
    assert.equal(SCOPE_ATTR_PARALLEL, 0b01);
    assert.equal(SCOPE_ATTR_RELOCATABLE, 0b10);
    assert.equal(TOOL_ATTR_LOCAL, 0b01);
    assert.equal(LLM_ATTR_STATELESS, 0b01);
    assert.equal(LLM_ATTR_STREAMING, 0b10);
  });
});

// ===========================================================================
// JsLlmRequest
// ===========================================================================

describe('JsLlmRequest', () => {
  it('construction and getters', () => {
    const req = new JsLlmRequest({ method: 'POST', url: 'https://api.test.com', headers: { 'Content-Type': 'application/json' }, body: { model: 'gpt-4' } });
    assert.equal(req.method, 'POST');
    assert.equal(req.url, 'https://api.test.com');
    assert.deepEqual(req.headers, { 'Content-Type': 'application/json' });
    assert.deepEqual(req.body, { model: 'gpt-4' });
  });
});
