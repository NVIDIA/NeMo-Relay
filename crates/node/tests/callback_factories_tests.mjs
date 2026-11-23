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

function invoke(wrapper, callId, { abort = false, streamResult = false, scope = undefined, publication = false } = {}) {
  wrapper(abort, callId, { value: 1 }, false, undefined, publication, undefined, scope, streamResult);
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

describe('failure description', () => {
  function rejectionOf(thrown) {
    const rejected = Promise.withResolvers();
    const wrapper = factories.promise(
      () => {
        throw thrown;
      },
      rejectSettle('resolve'),
      (_callId, exception, message, exceptionType) => rejected.resolve({ exception, message, exceptionType }),
      rejectSettle('streamPush'),
      rejectSettle('streamEnd'),
      rejectSettle('next'),
    );
    invoke(wrapper, 1);
    return rejected.promise;
  }

  it('uses the error name when the message cannot be read', async () => {
    const thrown = new Error('hidden');
    Object.defineProperties(thrown, {
      name: { value: 'GetterError' },
      message: {
        get() {
          throw new Error('message getter boom');
        },
      },
    });
    assert.deepEqual(await rejectionOf(thrown), {
      exception: thrown,
      message: 'GetterError',
      exceptionType: 'GetterError',
    });
  });

  it('describes thrown primitives by value and type', async () => {
    assert.deepEqual(await rejectionOf(42), { exception: 42, message: '42', exceptionType: 'Number' });
    assert.deepEqual(await rejectionOf(null), { exception: null, message: 'null', exceptionType: 'Null' });
  });

  it('describes objects without a name by their type tag', async () => {
    const thrown = { message: '' };
    assert.deepEqual(await rejectionOf(thrown), { exception: thrown, message: 'Object', exceptionType: 'Object' });
  });
});

describe('fork publication selection', { timeout: 3000 }, () => {
  for (const binding of ['withCallbackScope', 'setCallbackScope']) {
    for (const selection of ['ambient', 'absent', 'active']) {
      it(`${binding} retains ${selection} publication ownership`, async () => {
        const completed = Promise.withResolvers();
        const wrapper = factories.promise(
          async () => {
            const originalIdentifier = factories.eventSanitizerCallbackContextId();
            assert.equal(typeof originalIdentifier, 'string');
            const selectedScope = { selection };
            const capturedIdentifier = selection === 'active' ? originalIdentifier : null;
            const captured = selection !== 'ambient';
            const inspect = async () => {
              for (let iteration = 0; iteration < 2; iteration += 1) {
                assert.equal(factories.callbackScope(), selectedScope);
                assert.equal(factories.publicationCallbackActive(), selection !== 'absent');
                assert.equal(
                  factories.eventSanitizerCallbackContextId(),
                  selection === 'absent' ? undefined : originalIdentifier,
                );
                await Promise.resolve();
              }
            };
            if (binding === 'withCallbackScope') {
              await factories.withCallbackScope(selectedScope, inspect, capturedIdentifier, captured);
            } else {
              await factories.withCallbackScope({ parent: true }, async () => {
                assert.equal(factories.setCallbackScope(selectedScope, capturedIdentifier, captured), true);
                await inspect();
              });
            }
            assert.equal(factories.eventSanitizerCallbackContextId(), originalIdentifier);
          },
          () => completed.resolve(),
          (_identifier, error) => completed.reject(error),
          rejectSettle('streamPush'),
          rejectSettle('streamEnd'),
          rejectSettle('next'),
        );
        invoke(wrapper, 1, { publication: true, scope: { parent: true } });
        await completed.promise;
      });
    }
  }
});

describe('callback context boundary', () => {
  function scopedResult(scope) {
    assert.equal(factories.setCallbackScope(scope, null, true), true);
    return {
      get marker() {
        return factories.callbackScope()?.marker;
      },
    };
  }

  it('encodes a synchronous finalizer result inside the callback context', () => {
    const envelope = factories.finalizer(() => scopedResult({ marker: 'sync finalizer' }))(
      () => assert.fail('direct settlement must not call complete'),
      true,
    );
    assert.equal(envelope.ok, true);
    assert.equal(envelope.value.marker, 'sync finalizer');
  });

  it('encodes a Promise-returning finalizer result inside the callback context', async () => {
    const completed = Promise.withResolvers();
    const returned = factories.finalizer(async () => {
      const result = scopedResult({ marker: 'async finalizer' });
      await Promise.resolve();
      return result;
    })(completed.resolve, false);
    assert.equal(returned, undefined);
    const envelope = await completed.promise;
    assert.equal(envelope.ok, true);
    assert.equal(envelope.value.marker, 'async finalizer');
  });

  it('adopts a thenable returned to withCallbackScope inside the callback context', async () => {
    const observed = Promise.withResolvers();
    const value = await factories.withCallbackScope(
      { marker: 'thenable' },
      () => ({
        then(resolve) {
          observed.resolve(factories.callbackScope()?.marker);
          resolve('settled');
        },
      }),
      null,
      false,
    );
    assert.equal(value, 'settled');
    assert.equal(await observed.promise, 'thenable');
  });
});
