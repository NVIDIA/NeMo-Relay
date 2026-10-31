// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(() => {
  const { AsyncLocalStorage } = process.getBuiltinModule('node:async_hooks');
  const eventSanitizerContext = new AsyncLocalStorage();
  const publicationStates = new Map();
  const streamQueues = new WeakMap();
  const streamCloses = new WeakMap();
  const pinnedStreams = new Map();
  let nextPublicationContextId = 0;

  function callbackStore(publicationState, publicationContextId, scope) {
    const lifecycle = { expired: false, stores: new Set() };
    const store = { lifecycle, publicationState, publicationContextId, scope };
    lifecycle.stores.add(store);
    return store;
  }

  function replacementCallbackStore(current, scope) {
    const store = {
      lifecycle: current.lifecycle,
      publicationState: current.publicationState,
      publicationContextId: current.publicationContextId,
      scope,
    };
    current.lifecycle.stores.add(store);
    if (current.lifecycle.expired) {
      store.scope = undefined;
    }
    return store;
  }

  function expireCallbackStore(store) {
    if (store === undefined) {
      return;
    }
    store.lifecycle.expired = true;
    for (const current of store.lifecycle.stores) {
      current.scope = undefined;
    }
    store.lifecycle.stores.clear();
  }

  function jsonValue(value, seen = new Set()) {
    if (value === null || typeof value === 'string' || typeof value === 'boolean') {
      return value;
    }
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) {
        throw new TypeError('JavaScript callback returned a non-finite number that cannot be converted to JSON');
      }
      return value;
    }
    if (typeof value !== 'object') {
      throw new TypeError(
        `JavaScript callback returned an unsupported ${typeof value} value that cannot be converted to JSON`,
      );
    }
    if (seen.has(value)) {
      throw new TypeError('JavaScript callback returned a circular value that cannot be converted to JSON');
    }
    seen.add(value);
    if (Array.isArray(value)) {
      const length = value.length;
      const result = new Array(length);
      for (let index = 0; index < length; index += 1) {
        result[index] = jsonValue(value[index], seen);
      }
      seen.delete(value);
      return result;
    }

    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      seen.delete(value);
      throw new TypeError('JavaScript callback returned an unsupported object value that cannot be converted to JSON');
    }

    const result = Object.create(null);
    for (const key of Object.keys(value)) {
      result[key] = jsonValue(value[key], seen);
    }
    seen.delete(value);
    return result;
  }

  function failureEnvelope(error, fallbackMessage) {
    let message = fallbackMessage;
    let exceptionType = 'Error';
    try {
      if (typeof error === 'string') {
        message = error;
      } else if (error === null || (typeof error !== 'object' && typeof error !== 'function')) {
        message = String(error);
      } else {
        const errorMessage = error.message;
        if (typeof errorMessage === 'string' && errorMessage.length > 0) {
          message = errorMessage;
        }
      }
    } catch {}
    try {
      const errorName = error?.name;
      if (typeof errorName === 'string' && errorName.length > 0) {
        exceptionType = errorName;
      }
    } catch {}
    return { ok: false, exception: error, message, exceptionType };
  }

  function cleanupFailure(error, cleanup) {
    const primary = failureEnvelope(error, 'unknown error');
    const secondary = failureEnvelope(cleanup, 'unknown error');
    return new AggregateError([error, cleanup], `${primary.message}; stream cleanup also failed: ${secondary.message}`);
  }

  function isThenable(value) {
    return (
      value !== null && (typeof value === 'object' || typeof value === 'function') && typeof value.then === 'function'
    );
  }

  function messageFor(error) {
    try {
      return String(error?.message ?? error);
    } catch {
      return 'JavaScript callback failed';
    }
  }

  // Return a synchronous callback's envelope to a direct caller so a chunk is delivered in
  // the same poll.
  function settleEnvelope(fn, encode, complete, fallbackMessage, direct) {
    const envelopeFor = (value) => {
      try {
        return { ok: true, value: encode(value) };
      } catch (error) {
        return failureEnvelope(error, fallbackMessage);
      }
    };
    let envelope;
    try {
      const value = fn();
      if (isThenable(value)) {
        Promise.resolve(value).then(
          (resolved) => complete(envelopeFor(resolved)),
          (error) => complete(failureEnvelope(error, fallbackMessage)),
        );
        return undefined;
      }
      envelope = envelopeFor(value);
    } catch (error) {
      envelope = failureEnvelope(error, fallbackMessage);
    }
    if (direct) {
      return envelope;
    }
    complete(envelope);
    return undefined;
  }

  function driveStream(poll, close) {
    const step = poll(close);
    if (!(step instanceof Promise)) {
      return step;
    }
    return step.then((settled) => (settled === null ? driveStream(poll, close) : close ? undefined : null));
  }

  // Keep a handle reachable while its next() or close() is pending so collection does
  // not cut a read short.
  function pinStream(handle, run) {
    pinnedStreams.set(handle, (pinnedStreams.get(handle) ?? 0) + 1);
    const unpin = () => {
      const left = pinnedStreams.get(handle) - 1;
      if (left > 0) {
        pinnedStreams.set(handle, left);
      } else {
        pinnedStreams.delete(handle);
      }
    };
    let outcome;
    try {
      outcome = run();
    } catch (error) {
      unpin();
      throw error;
    }
    if (!(outcome instanceof Promise)) {
      unpin();
      return outcome;
    }
    return outcome.then(
      (value) => {
        unpin();
        return value;
      },
      (error) => {
        unpin();
        throw error;
      },
    );
  }

  function callPromise(
    fn,
    settle,
    controllers,
    callId,
    arg0,
    spread,
    next,
    publication,
    publicationContextId,
    scope,
    streamResult,
  ) {
    const controller = new AbortController();
    controllers.set(callId, controller);
    // Downstream streams from `next()`. A wrapper the intercept iterates reports
    // the close itself. The rest are closed before this call settles.
    const downstreams = new Map();
    const closeDownstream = (downstream) => {
      const state = downstreams.get(downstream);
      state.closing ??= downstream.close();
      return state.closing;
    };
    const settleDownstreams = async (failure) => {
      for (const [downstream, state] of downstreams) {
        if (state.reported) continue;
        state.reported = true;
        try {
          await closeDownstream(downstream);
        } catch (cleanup) {
          failure = { error: failure === undefined ? cleanup : cleanupFailure(failure.error, cleanup) };
        }
      }
      return failure;
    };
    controller.signal.addEventListener(
      'abort',
      () => {
        for (const downstream of downstreams.keys()) {
          closeDownstream(downstream).catch(() => {});
        }
      },
      { once: true },
    );
    let ownsPublicationState = false;
    let publicationState;
    if (publicationContextId !== undefined) {
      publicationState = publicationStates.get(publicationContextId);
      if (publicationState === undefined && publication) {
        publicationContextId = String(++nextPublicationContextId);
        publicationState = { active: true };
        publicationStates.set(publicationContextId, publicationState);
        ownsPublicationState = true;
      } else if (publicationState === undefined) {
        publicationState = { active: false };
      }
    } else if (publication) {
      publicationContextId = String(++nextPublicationContextId);
      publicationState = { active: true };
      publicationStates.set(publicationContextId, publicationState);
      ownsPublicationState = true;
    } else {
      publicationState = { active: false };
    }
    const token = callbackStore(publicationState, publicationContextId, scope);
    const settlePublication = () => {
      if (ownsPublicationState) {
        publicationState.active = false;
        publicationStates.delete(publicationContextId);
      }
      expireCallbackStore(token);
      controllers.delete(callId);
    };
    const invokeNext = (value) =>
      settle.next(next, jsonValue(value === undefined ? null : value), eventSanitizerContext.getStore()?.scope);
    const safeNext =
      next === undefined
        ? undefined
        : !streamResult
          ? invokeNext
          : async (value) => {
              const downstream = await invokeNext(value);
              const state = { closing: undefined, reported: false };
              downstreams.set(downstream, state);
              if (controller.signal.aborted) {
                closeDownstream(downstream).catch(() => {});
              }
              return {
                async *[Symbol.asyncIterator]() {
                  try {
                    for (;;) {
                      const chunk = await downstream.next();
                      if (chunk === null) return;
                      yield chunk;
                    }
                  } finally {
                    state.reported = true;
                    await closeDownstream(downstream);
                  }
                },
              };
            };
    const invoke = () => {
      Promise.resolve()
        .then(() =>
          safeNext === undefined
            ? spread
              ? fn(...arg0, controller.signal)
              : fn(arg0, controller.signal)
            : spread
              ? fn(...arg0, safeNext, controller.signal)
              : fn(arg0, safeNext, controller.signal),
        )
        .then(async (value) => {
          if (!streamResult) {
            settlePublication();
            settle.resolve(callId, jsonValue(value === undefined ? null : value));
            return;
          }
          if (value == null || typeof value[Symbol.asyncIterator] !== 'function') {
            throw new TypeError('streaming callback must return an AsyncIterable');
          }
          settle.resolve(callId);
          const iterator = value[Symbol.asyncIterator]();
          // Like `for await`, a finished or failed `next()` leaves nothing for `return()` to clean up.
          let finished = false;
          let failure;
          try {
            while (!controller.signal.aborted) {
              let item;
              try {
                item = await iterator.next();
              } catch (error) {
                finished = true;
                throw error;
              }
              if (item.done) {
                finished = true;
                break;
              }
              await settle.streamPush(callId, jsonValue(item.value === undefined ? null : item.value));
            }
          } catch (error) {
            failure = { error };
          }
          if (!finished && typeof iterator.return === 'function') {
            try {
              await iterator.return();
            } catch (cleanup) {
              failure = { error: failure === undefined ? cleanup : cleanupFailure(failure.error, cleanup) };
            }
          }
          failure = await settleDownstreams(failure);
          if (failure !== undefined) {
            throw failure.error;
          }
          settle.streamEnd(callId);
          settlePublication();
        })
        .catch(async (error) => {
          const failure = await settleDownstreams({ error });
          settlePublication();
          const envelope = failureEnvelope(failure.error, 'unknown error');
          settle.reject(callId, failure.error, envelope.message, envelope.exceptionType);
        });
    };
    eventSanitizerContext.run(token, invoke);
  }

  return {
    noop() {},

    execution(fn) {
      return function __nemo_relay_execution_wrapper(...args) {
        try {
          const value = fn(...args);
          return { ok: true, value: jsonValue(value === undefined ? null : value) };
        } catch (error) {
          return failureEnvelope(error, 'JavaScript callback failed');
        }
      };
    },

    middleware(fn) {
      return function __nemo_relay_middleware_wrapper(...args) {
        try {
          const value = fn(...args);
          return { ok: true, value: value === undefined ? null : value };
        } catch (error) {
          return failureEnvelope(error, 'JavaScript callback threw');
        }
      };
    },

    conditionalGate(fn) {
      return function __nemo_relay_conditional_gate_wrapper(...args) {
        try {
          const returned = fn(...args);
          const value = returned === undefined ? null : returned;
          if (value === null || typeof value === 'string') {
            return { ok: true, value };
          }
          return { ok: false, error: 'conditional middleware guardrail must return a string or null' };
        } catch (error) {
          return failureEnvelope(error, 'JavaScript callback threw');
        }
      };
    },

    subscriber(fn, complete) {
      return function __nemo_relay_subscriber_wrapper(error, event, callbackId) {
        // A failed argument build already completed the callback from Rust.
        if (error != null) {
          return;
        }
        Promise.resolve()
          .then(() => fn(event))
          .then(
            () => complete(callbackId),
            (error) => complete(callbackId, messageFor(error)),
          );
      };
    },

    // Chaining through a resolved promise stores the queue tail before a poll runs, so a
    // next() from inside a poll queues behind it.
    streamNext(poll, handle) {
      const run = () => pinStream(handle, () => driveStream(poll, false));
      const current = (streamQueues.get(poll) ?? Promise.resolve()).then(run, run);
      streamQueues.set(poll, current);
      return current;
    },

    pinWhile(handle, promise) {
      return pinStream(handle, () => promise);
    },

    aggregate(errors, message) {
      return new AggregateError(errors, message);
    },

    streamClose(poll, handle) {
      const existing = streamCloses.get(poll);
      if (existing !== undefined) {
        return existing;
      }
      const closing = Promise.resolve().then(() => pinStream(handle, () => driveStream(poll, true)));
      streamCloses.set(poll, closing);
      return closing;
    },

    collector(fn) {
      return function __nemo_relay_collector_wrapper(chunk, complete, direct) {
        // A direct call runs inside a consumer's context. Pin a bare store like the
        // threadsafe path does.
        const token = callbackStore({ active: false }, undefined, undefined);
        const envelope = eventSanitizerContext.run(token, () =>
          settleEnvelope(
            () => fn(chunk),
            () => null,
            (settled) => {
              expireCallbackStore(token);
              complete(settled);
            },
            'JavaScript collector callback failed',
            direct,
          ),
        );
        if (envelope !== undefined) {
          expireCallbackStore(token);
        }
        return envelope;
      };
    },

    finalizer(fn) {
      return function __nemo_relay_finalizer_wrapper(complete, direct) {
        // The finalizer is part of the END publication, so a flush inside it is re-entrant.
        const token = callbackStore({ active: true }, undefined, undefined);
        const envelope = eventSanitizerContext.run(token, () =>
          settleEnvelope(
            fn,
            (value) => jsonValue(value === undefined ? null : value),
            (settled) => {
              expireCallbackStore(token);
              complete(settled);
            },
            'JavaScript finalizer callback failed',
            direct,
          ),
        );
        if (envelope !== undefined) {
          expireCallbackStore(token);
        }
        return envelope;
      };
    },

    promise(fn, resolve, reject, streamPush, streamEnd, next) {
      const settle = { resolve, reject, streamPush, streamEnd, next };
      const controllers = new Map();
      return function __nemo_relay_promise_wrapper(
        error,
        abort,
        callId,
        arg0,
        spread,
        next,
        publication,
        publicationContextId,
        scope,
        streamResult,
      ) {
        // A failed argument build already settled the call from Rust.
        if (error != null) {
          return;
        }
        if (abort) {
          controllers.get(callId)?.abort();
          return;
        }
        callPromise(
          fn,
          settle,
          controllers,
          callId,
          arg0,
          spread,
          next,
          publication,
          publicationContextId,
          scope,
          streamResult,
        );
      };
    },

    scoped(fn) {
      return function __nemo_relay_scoped_wrapper(arg, scope) {
        const current = eventSanitizerContext.getStore();
        const token = callbackStore(
          current?.publicationState ?? { active: false },
          current?.publicationContextId,
          scope,
        );
        try {
          return eventSanitizerContext.run(token, () => fn(arg));
        } finally {
          expireCallbackStore(token);
        }
      };
    },

    publicationCallbackActive() {
      return eventSanitizerContext.getStore()?.publicationState.active === true;
    },

    eventSanitizerCallbackContextId() {
      const current = eventSanitizerContext.getStore();
      return current?.publicationState.active === true ? current.publicationContextId : undefined;
    },

    callbackScope() {
      return eventSanitizerContext.getStore()?.scope;
    },

    withCallbackScope(scope, fn) {
      const current = eventSanitizerContext.getStore();
      const token = callbackStore(current?.publicationState ?? { active: false }, current?.publicationContextId, scope);
      const expire = () => expireCallbackStore(token);
      let value;
      try {
        value = eventSanitizerContext.run(token, fn);
      } catch (error) {
        expire();
        throw error;
      }
      if (isThenable(value)) {
        return Promise.resolve(value).finally(expire);
      }
      expire();
      return value;
    },

    setCallbackScope(scope) {
      const current = eventSanitizerContext.getStore();
      if (current === undefined) {
        return false;
      }
      eventSanitizerContext.enterWith(replacementCallbackStore(current, scope));
      return true;
    },
  };
})();
