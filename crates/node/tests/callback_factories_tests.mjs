// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { runInThisContext } from 'node:vm';

const factoriesUrl = new URL('../src/callback_factories.js', import.meta.url);
const factories = runInThisContext(readFileSync(factoriesUrl, 'utf8'), { filename: fileURLToPath(factoriesUrl) });

function neverSettles() {
  return new Promise(() => {});
}

function abortedThen(signal, started) {
  return new Promise((resolve) => {
    signal.addEventListener('abort', resolve, { once: true });
    started.resolve();
  });
}

function invoke(wrapper, callId, { abort = false, streamResult = false, scope = undefined } = {}) {
  wrapper(abort, callId, { value: 1 }, false, undefined, false, undefined, scope, streamResult);
}

function rejectSettle(name) {
  return () => {
    throw new Error(`${name} settled a cancelled callback`);
  };
}

describe('promise callback cancellation', () => {
  it('aborts a non-stream callback that never settles', async () => {
    const started = Promise.withResolvers();
    const wrapper = factories.promise(
      (_value, signal) => {
        started.resolve(signal);
        return neverSettles();
      },
      rejectSettle('resolve'),
      rejectSettle('reject'),
      rejectSettle('streamPush'),
      rejectSettle('streamEnd'),
      rejectSettle('next'),
    );

    invoke(wrapper, 1);
    const signal = await started.promise;
    assert.equal(signal.aborted, false);

    invoke(wrapper, 1, { abort: true });
    assert.equal(signal.aborted, true);
  });

  it('keeps the callback scope until an aborted callback settles', async () => {
    const scope = { marker: 'kept' };
    const started = Promise.withResolvers();
    const resolved = Promise.withResolvers();
    const wrapper = factories.promise(
      async (_value, signal) => {
        await abortedThen(signal, started);
        return factories.callbackScope();
      },
      (callId, value) => resolved.resolve([callId, value]),
      rejectSettle('reject'),
      rejectSettle('streamPush'),
      rejectSettle('streamEnd'),
      rejectSettle('next'),
    );

    invoke(wrapper, 2, { scope });
    await started.promise;
    invoke(wrapper, 2, { abort: true });
    const [callId, value] = await resolved.promise;
    assert.equal(callId, 2);
    assert.equal(value.marker, 'kept');
  });

  it('runs a cancelled stream callback through its cleanup', async () => {
    const started = Promise.withResolvers();
    const ended = Promise.withResolvers();
    let cleanedUp = false;
    const wrapper = factories.promise(
      (_value, signal) => ({
        async *[Symbol.asyncIterator]() {
          try {
            await abortedThen(signal, started);
          } finally {
            cleanedUp = true;
          }
        },
      }),
      () => {},
      rejectSettle('reject'),
      rejectSettle('streamPush'),
      (callId) => ended.resolve(callId),
      rejectSettle('next'),
    );

    invoke(wrapper, 3, { streamResult: true });
    await started.promise;
    invoke(wrapper, 3, { abort: true, streamResult: true });
    assert.equal(await ended.promise, 3);
    assert.equal(cleanedUp, true);
  });
});
