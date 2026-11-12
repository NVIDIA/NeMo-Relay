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
    nextState,
    publication,
    publicationContextId,
    scope,
    streamResult,
  ) {
    const controller = new AbortController();
    controllers.set(callId, controller);
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
    let publicationState = publicationContextId === undefined ? undefined : publicationStates.get(publicationContextId);
    let ownsPublicationState = false;
    if (publicationState === undefined) {
      if (publication) {
        publicationContextId = String(++nextPublicationContextId);
        publicationState = { active: true };
        publicationStates.set(publicationContextId, publicationState);
        ownsPublicationState = true;
      } else {
        publicationState = { active: false };
      }
    }
    const token = callbackStore(publicationState, publicationContextId, scope);
    const releaseCall = () => {
      if (ownsPublicationState) {
        publicationState.active = false;
        publicationStates.delete(publicationContextId);
      }
      controllers.delete(callId);
    };
    const settlePublication = () => {
      releaseCall();
      expireCallbackStore(token);
    };
    controller.signal.addEventListener(
      'abort',
      () => {
        for (const downstream of downstreams.keys()) {
          closeDownstream(downstream).catch(() => {});
        }
        if (!streamResult) {
          releaseCall();
        }
      },
      { once: true },
    );
    const invokeNext = (value) =>
      settle.next(nextState, jsonValue(value === undefined ? null : value), eventSanitizerContext.getStore()?.scope);
    const safeNext =
      nextState === undefined
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
        .then(() => {
          const args = spread ? [...arg0] : [arg0];
          if (safeNext !== undefined) {
            args.push(safeNext);
          }
          args.push(controller.signal);
          return fn(...args);
        })
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

    errorFirst(fn) {
      return function __nemo_relay_error_first(error, ...args) {
        if (error !== null) {
          return undefined;
        }
        return fn(...args);
      };
    },

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
          return { ok: true, value: jsonValue(value === undefined ? null : value) };
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
      return function __nemo_relay_subscriber_wrapper(event, callbackId) {
        Promise.resolve()
          .then(() => fn(event))
          .then(
            () => complete(callbackId),
            (error) => complete(callbackId, messageFor(error)),
          );
      };
    },

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
        abort,
        callId,
        arg0,
        spread,
        nextState,
        publication,
        publicationContextId,
        scope,
        streamResult,
      ) {
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
          nextState,
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

    withCallbackScope(scope, fn, capturedPublicationContextId) {
      const current = eventSanitizerContext.getStore();
      const publicationContextId = capturedPublicationContextId ?? current?.publicationContextId;
      const publicationState =
        capturedPublicationContextId == null
          ? (current?.publicationState ?? { active: false })
          : (publicationStates.get(capturedPublicationContextId) ?? { active: false });
      const token = callbackStore(publicationState, publicationContextId, scope);
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
