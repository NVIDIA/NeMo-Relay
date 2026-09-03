// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The argument-transform decision matrix.
 *
 * pi validates tool arguments *before* the `tool_call` hook and never
 * re-validates, so a rewrite that violates the schema would execute. The schema
 * is reachable -- `pi.getAllTools()` exposes it -- and deliberately not used,
 * because pi's tool set is per-session mutable and one read goes stale. The
 * shape check is what stands in for validation instead: same keys, same JSON
 * types, recursively. These tests pin both the cases it must allow and the ones
 * it must refuse.
 *
 * Run: node --test integrations/pi/test/*.test.mjs
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

const { applyTransform, decideTransform, refusalReason, shapeViolation } = await import(
  '../src/argument-transform.ts'
);

const CALL = 'call-1';
const envelope = (input, id = CALL) => ({ tool_call: { tool_call_id: id, input } });

describe('transform decision', () => {
  it('has nothing to do when the body carries no transform', () => {
    assert.equal(decideTransform({}, CALL, { path: 'a.txt' }).kind, 'none');
    assert.equal(decideTransform(null, CALL, { path: 'a.txt' }).kind, 'none');
    assert.equal(decideTransform({ tool_call: {} }, CALL, { path: 'a.txt' }).kind, 'none');
  });

  // The use case this exists for: a policy rewriting a path or redacting a value.
  it('applies a value rewrite that preserves the shape', () => {
    const outcome = decideTransform(
      envelope({ path: '.env.example' }),
      CALL,
      { path: '.env' },
    );
    assert.equal(outcome.kind, 'apply');
    assert.deepEqual(outcome.input, { path: '.env.example' });
  });

  it('refuses an added or removed key', () => {
    const added = decideTransform(envelope({ path: 'a.txt', sudo: true }), CALL, { path: 'a.txt' });
    assert.equal(added.kind, 'refuse');
    assert.match(added.reason, /added sudo/);

    const removed = decideTransform(envelope({}), CALL, { path: 'a.txt' });
    assert.equal(removed.kind, 'refuse');
    assert.match(removed.reason, /removed path/);
  });

  // A required string becoming null is exactly the schema violation pi would execute unchecked.
  it('refuses a type change, including to null', () => {
    for (const value of [null, 42, ['a.txt'], { nested: true }]) {
      const outcome = decideTransform(envelope({ path: value }), CALL, { path: 'a.txt' });
      assert.equal(outcome.kind, 'refuse', JSON.stringify(value));
      assert.match(outcome.reason, /input\.path changed type/);
    }
  });

  it('checks nested objects and arrays, naming the path', () => {
    const nested = decideTransform(
      envelope({ opts: { limit: 'ten' } }),
      CALL,
      { opts: { limit: 10 } },
    );
    assert.equal(nested.kind, 'refuse');
    assert.match(nested.reason, /input\.opts\.limit changed type from number to string/);

    const shorter = decideTransform(envelope({ paths: ['a'] }), CALL, { paths: ['a', 'b'] });
    assert.equal(shorter.kind, 'refuse');
    assert.match(shorter.reason, /input\.paths changed length from 2 to 1/);

    const ok = decideTransform(envelope({ paths: ['x', 'y'] }), CALL, { paths: ['a', 'b'] });
    assert.equal(ok.kind, 'apply');
  });

  // A body for a different call means the two sides disagree about what is in flight; applying it
  // would rewrite one tool call with another's arguments.
  it('refuses a transform addressed to a different tool call', () => {
    const outcome = decideTransform(envelope({ path: 'b.txt' }, 'call-2'), CALL, { path: 'a.txt' });
    assert.equal(outcome.kind, 'refuse');
    assert.match(outcome.reason, /call-2, not call-1/);
  });

  // The echoed id is the only thing proving the transform belongs to the call we
  // posted. A missing or non-string one used to skip the check entirely, so a
  // truncated body could rewrite the wrong call's arguments.
  it('refuses a transform whose call id is missing or not a string', () => {
    // Built by hand rather than through `envelope`, whose default parameter would
    // substitute a valid id for the absent case and quietly test nothing.
    const bodies = [
      ['absent', { tool_call: { input: { path: 'b.txt' } } }],
      ['null', { tool_call: { tool_call_id: null, input: { path: 'b.txt' } } }],
      ['number', { tool_call: { tool_call_id: 42, input: { path: 'b.txt' } } }],
      ['object', { tool_call: { tool_call_id: { id: CALL }, input: { path: 'b.txt' } } }],
      ['array', { tool_call: { tool_call_id: [CALL], input: { path: 'b.txt' } } }],
    ];
    for (const [label, body] of bodies) {
      const outcome = decideTransform(body, CALL, { path: 'a.txt' });
      assert.equal(outcome.kind, 'refuse', `a ${label} call id must be refused`);
      assert.match(outcome.reason, /not call-1/);
    }
  });

  it('refuses a non-object transform', () => {
    const outcome = decideTransform(envelope('rm -rf /'), CALL, { path: 'a.txt' });
    assert.equal(outcome.kind, 'refuse');
    assert.match(outcome.reason, /is string, not an object/);
  });
});

describe('what the shape check does not promise', () => {
  // Documented limitation, asserted so it is not mistaken for validation: the check does not
  // consult the tool's schema, so pattern/enum/range violations pass and will execute.
  it('allows a value the schema might still reject', () => {
    assert.equal(shapeViolation({ path: 'a.txt' }, { path: '../../etc/shadow' }), null);
    assert.equal(shapeViolation({ mode: 'read' }, { mode: 'not-an-enum-member' }), null);
  });

  // The check is structural, not content-aware: a shortened string is still a string, so a
  // truncated argument would come back rewritten and be applied verbatim -- a `write` would land
  // on disk cut short. This is why the gated post forwards arguments whole while results are
  // bounded at 2000 characters. See "What Is Not Represented" in the README.
  it('cannot tell a shortened string from the original, which is why arguments are never truncated', () => {
    assert.equal(
      shapeViolation(
        { path: '/work/a.txt', content: 'a much longer original body' },
        { path: '/work/a.txt', content: 'short' },
      ),
      null,
    );
  });
});

describe('applying the transform', () => {
  // In place is required, not stylistic: pi hands the same object to the tool and to later
  // handlers, so replacing the reference would be silently discarded.
  it('mutates the object pi will execute rather than replacing it', () => {
    const input = { path: '.env', encoding: 'utf8' };
    const seenByPi = input;
    applyTransform(input, { path: '.env.example', encoding: 'utf8' });
    assert.equal(seenByPi.path, '.env.example');
    assert.equal(seenByPi, input);
  });
});

describe('the refusal reason', () => {
  // pi hands the reason to the model verbatim, so it has to read as guidance and must not look
  // like the model did something wrong.
  it('names the tool, the cause, and that it is not a judgment of the request', () => {
    const reason = refusalReason('read', 'input added sudo');
    assert.match(reason, /read/);
    assert.match(reason, /input added sudo/);
    assert.match(reason, /not a judgment about your request/);
    assert.match(reason, /blocked rather than run with the original/);
  });
});
