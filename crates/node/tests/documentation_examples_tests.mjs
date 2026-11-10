// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { afterEach, it } from 'node:test';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const relay = require('nemo-relay-node');
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
  it(`${name} policy allows another tool without a query`, async () => {
    register();
    const received = [];
    const { result, events } = await captureEvents(() =>
      relay.toolCallExecute('get_time', { timezone: 'UTC' }, (toolArguments) => {
        received.push(toolArguments);
        return { result: { timezone: toolArguments.timezone, time: '12:00:00' } };
      }),
    );
    assert.deepEqual(received, [{ timezone: 'UTC' }]);
    assert.deepEqual(result.result, { timezone: 'UTC', time: '12:00:00' });
    assertPair(events, 'tool', 'get_time');
  });

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

it('reference provider and stream examples return their documented results', async () => {
  const managed = await import(`${examples}/managed-llm.ts`);
  const streaming = await import(`${examples}/streaming-llm.ts`);
  const execution = await captureEvents(() => managed.runExample());
  assert.deepEqual(execution.result, { text: 'hi', request: { messages: [{ role: 'user', content: 'hello' }] } });
  assertPair(execution.events, 'llm', 'demo-provider');
  const stream = await captureEvents(() => streaming.runExample());
  assert.deepEqual(stream.result, { delta: 'hi' });
  assertPair(stream.events, 'llm', 'demo-provider');
});

for (const query of ['weather', '']) {
  it(`reference manual lifecycle completes for ${query ? 'success' : 'failure'}`, async () => {
    const program = await import(`${examples}/manual-tool-lifecycle.ts`);
    const { events } = await captureEvents(async () => {
      if (query) assert.deepEqual(program.runExample(query), { hits: 2 });
      else assert.throws(() => program.runExample(query), /^Error: query is required$/);
    });
    assertPair(events, 'tool', 'search');
  });
}

it('partial middleware returns transformed values without executing a callback', async () => {
  const program = await import(`${examples}/partial-middleware.ts`);
  const { result, events } = await captureEvents(() => program.runExample());
  assert.deepEqual(result.toolArguments, { query: 'weather' });
  assert.deepEqual(result.request.content, { messages: [{ role: 'user', content: 'hello' }] });
  assert.equal(
    events.some((event) => event.category === 'tool' || event.category === 'llm'),
    false,
  );
});

it('scope-context example emits its mark within the request', async () => {
  const program = await import(`${examples}/scope-context.ts`);
  const previousStack = relay.currentScopeStack();
  try {
    const { events } = await captureEvents(() => program.runExample());
    const parent = assertPair(events, 'agent', 'request');
    assert.equal(events.find((event) => event.name === 'started').parent_uuid, parent.uuid);
  } finally {
    relay.setThreadScopeStack(previousStack);
  }
});

it('global gate example reports a missing required OpenTelemetry subscriber', async () => {
  const program = await import(`${examples}/global-registration-gate.ts`);
  await assert.rejects(
    () => program.runExample(async () => assert.fail('application must not run')),
    /OpenTelemetry subscriber was not found/,
  );
});

it('scope program emits two marks within the agent and removes its subscriber', async () => {
  const program = require(`${examples}/scopes-and-marks.js`);
  const { events } = await captureEvents(() => program.main());
  const parent = assertPair(events, 'agent', 'agent-run');
  const marks = events.filter((event) => event.kind === 'mark');
  assert.deepEqual(
    marks.map((event) => event.name),
    ['planning-started', 'planning-finished'],
  );
  assert.equal(
    marks.every((event) => event.parent_uuid === parent.uuid),
    true,
  );
  assert.equal(
    relay.listRuntimeRegistrations(['subscriber']).some((registration) => registration.localName === 'scope-check'),
    false,
  );
});
