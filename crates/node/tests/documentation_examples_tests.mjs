// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { afterEach, it } from 'node:test';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const relay = require('../index.js');
const examples = fileURLToPath(new URL('../../../docs/instrument-applications/_code/nodejs/', import.meta.url));
const toolExample = require(`${examples}/tool-call.js`);
const modelExample = require(`${examples}/llm-call.js`);
const queryPolicy = require(`${examples}/query-policy.js`);
const toolPolicy = require(`${examples}/tool-policy.js`);
const printer = require(`${examples}/print-event.js`);

afterEach(async () => {
  await relay.flushSubscribers();
  relay.deregisterToolConditionalExecutionGuardrail('search.require_query');
  relay.deregisterToolSanitizeRequestGuardrail('search.redact_api_key');
  relay.deregisterToolExecutionIntercept('search.measure');
});

async function captureEvents(action) {
  const events = [];
  relay.registerSubscriber('documentation-capture', (event) => events.push(event));
  try {
    const result = await action();
    await relay.flushSubscribers();
    return { result, events };
  } finally {
    await relay.flushSubscribers();
    relay.deregisterSubscriber('documentation-capture');
  }
}

function assertPair(events, category, name) {
  const matching = events.filter((event) => event.category === category && event.name === name);
  const starts = matching.filter((event) => event.scope_category === 'start');
  const ends = matching.filter((event) => event.scope_category === 'end');
  assert.equal(starts.length, 1);
  assert.equal(ends.length, 1);
  assert.equal(starts[0].uuid, ends[0].uuid);
  assert.equal(starts[0].parent_uuid, ends[0].parent_uuid);
  return starts[0];
}

for (const [name, example, category, callName] of [
  ['tool', toolExample, 'tool', 'search'],
  ['model', modelExample, 'llm', 'demo-provider'],
]) {
  it(`documented ${name} program returns its result and paired events`, async () => {
    const { result, events } = await captureEvents(() => example.main());
    assert.deepEqual(
      result,
      name === 'tool'
        ? { query: 'runtime instrumentation', hits: [{ title: 'NeMo Relay' }] }
        : { text: 'hello', messages: [{ role: 'user', content: 'hello' }] },
    );
    const agent = assertPair(events, 'agent', 'agent-run');
    assert.equal(assertPair(events, category, callName).parent_uuid, agent.uuid);
  });

  it(`documented ${name} entrypoint runs as a normal Node program`, () => {
    const filename = name === 'tool' ? 'tool-call.js' : 'llm-call.js';
    const output = execFileSync(process.execPath, [`${examples}/${filename}`], { encoding: 'utf8', timeout: 30000 });
    assert.match(output, /scope agent-run/);
    assert.match(output, name === 'tool' ? /NeMo Relay/ : /hello/);
  });

  it(`documented ${name} failure closes its scope and removes its subscriber`, async () => {
    const register =
      name === 'tool'
        ? relay.registerToolConditionalExecutionGuardrail
        : relay.registerLlmConditionalExecutionGuardrail;
    const deregister =
      name === 'tool'
        ? relay.deregisterToolConditionalExecutionGuardrail
        : relay.deregisterLlmConditionalExecutionGuardrail;
    register('documentation.reject', 1, () => 'documentation rejection');
    try {
      const { events } = await captureEvents(() => assert.rejects(example.main(), /documentation rejection/));
      assertPair(events, 'agent', 'agent-run');
      const subscriber = name === 'tool' ? 'instrumentation-check' : 'llm-check';
      assert.ok(
        !relay.listRuntimeRegistrations(['subscriber']).some((registration) => registration.localName === subscriber),
      );
    } finally {
      deregister('documentation.reject');
    }
  });
}

for (const [name, register] of [
  ['first', queryPolicy.registerQueryPolicy],
  ['combined', toolPolicy.registerToolPolicy],
]) {
  for (const query of ['weather', '']) {
    it(`${name} policy ${query ? 'allows the request' : 'rejects before the callback'}`, async () => {
      register();
      const received = [];
      const argumentsValue = { query, api_key: 'demo-value' };
      const { result, events } = await captureEvents(async () => {
        const run = () =>
          relay.toolCallExecute('search', argumentsValue, (toolArguments) => {
            received.push(toolArguments);
            return { result: { query: toolArguments.query } };
          });
        if (!query) {
          await assert.rejects(run(), /search.query is required/);
          return null;
        }
        return run();
      });
      if (!query) {
        assert.deepEqual(received, []);
      } else {
        assert.deepEqual(result.result, { query });
        assert.deepEqual(received, [argumentsValue]);
        if (name === 'combined') assert.equal(assertPair(events, 'tool', 'search').data.api_key, '<redacted>');
      }
    });
  }
}

for (const annotation of [undefined, { verification: 'preserved' }]) {
  it(`combined execution intercept forwards ${annotation === undefined ? 'absent' : 'present'} annotation`, async () => {
    toolPolicy.registerToolPolicy();
    const execution = await relay.toolCallExecute('search', { query: 'weather' }, () =>
      annotation === undefined ? { result: { ok: true } } : { result: { ok: true }, annotation },
    );
    assert.deepEqual(execution.result, { ok: true });
    assert.deepEqual(execution.annotation, annotation);
  });
}

it('JSON subscriber source handles a real delivered mark', async () => {
  relay.registerSubscriber('documentation-json', printer.logEvent);
  try {
    relay.event('documentation-check', null, { sample: 'synthetic' }, null);
    await relay.flushSubscribers();
    assert.equal(relay.getLastCallbackError(), null);
  } finally {
    relay.deregisterSubscriber('documentation-json');
  }
});
