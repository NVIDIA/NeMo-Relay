// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cached JavaScript callback wrapper factories for the Node binding.

use napi::bindgen_prelude::FromNapiValue;
use napi::{Env, JsFunction, JsObject, JsUnknown, NapiRaw, NapiValue, ValueType};
use nemo_relay::api::runtime::ScopeStackHandle;
use nemo_relay::api::runtime::subscriber_dispatcher::PublicationBuffer;

use crate::types::ScopeStack;

const CALLBACK_FACTORIES_PROPERTY: &str = "__nemo_relay_callback_factories_v27";

const CALLBACK_FACTORIES_SOURCE: &str = r#"(() => {
  const { AsyncLocalStorage } = process.getBuiltinModule('node:async_hooks');
  const eventSanitizerContext = new AsyncLocalStorage();
  const publicationStates = new Map();
  const streamQueues = new WeakMap();
  const streamCloses = new WeakMap();
  const pinnedStreams = new Map();
  let nextPublicationContextId = 0;

  function callbackStore(
    publicationState,
    publicationContextId,
    scopeStack,
    propagationParentUuid,
    propagationContext,
  ) {
    const lifecycle = { expired: false, stores: new Set() };
    const store = {
      lifecycle,
      publicationState,
      publicationContextId,
      scopeStack,
      propagationParentUuid,
      propagationContext,
    };
    lifecycle.stores.add(store);
    return store;
  }

  function replacementCallbackStore(current, scopeStack) {
    const store = {
      lifecycle: current.lifecycle,
      publicationState: current.publicationState,
      publicationContextId: current.publicationContextId,
      scopeStack,
      propagationParentUuid: undefined,
      propagationContext: undefined,
    };
    current.lifecycle.stores.add(store);
    if (current.lifecycle.expired) {
      store.scopeStack = null;
      store.propagationParentUuid = undefined;
      store.propagationContext = undefined;
    }
    return store;
  }

  function expireCallbackStore(store) {
    if (store === undefined) {
      return;
    }
    store.lifecycle.expired = true;
    for (const current of store.lifecycle.stores) {
      current.scopeStack = null;
      current.propagationParentUuid = undefined;
      current.propagationContext = undefined;
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
      throw new TypeError(`JavaScript callback returned an unsupported ${typeof value} value that cannot be converted to JSON`);
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
    return { ok: false, error: message, exceptionType };
  }

  function cleanupFailure(error, cleanup) {
    const primary = failureEnvelope(error, 'unknown error');
    const secondary = failureEnvelope(cleanup, 'unknown error');
    const combined = new Error(`${primary.error}; stream cleanup also failed: ${secondary.error}`);
    combined.name = primary.exceptionType;
    return combined;
  }

  function isThenable(value) {
    return (
      value !== null
      && (typeof value === 'object' || typeof value === 'function')
      && typeof value.then === 'function'
    );
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

  async function driveStream(poll, close) {
    for (;;) {
      const step = poll(close);
      if (!(step instanceof Promise)) {
        return step;
      }
      const settled = await step;
      if (settled === null) {
        continue;
      }
      if (settled.error != null) {
        throw new Error(settled.error);
      }
      return close ? undefined : null;
    }
  }

  function callPromise(
    fn,
    arg0,
    spread,
    next,
    resolve,
    reject,
    publication,
    publicationContextId,
    scopeStack,
    propagationParentUuid,
    propagationContext,
    registerAbort,
    streamResult,
    streamPush,
    streamEnd,
  ) {
    const controller = new AbortController();
    registerAbort(() => controller.abort());
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
    const token = callbackStore(
      publicationState,
      publicationContextId,
      scopeStack,
      propagationParentUuid,
      propagationContext,
    );
    const settlePublication = () => {
      if (ownsPublicationState) {
        publicationState.active = false;
        publicationStates.delete(publicationContextId);
      }
      expireCallbackStore(token);
    };
    const invokeNext = (value) => next(
      jsonValue(value === undefined ? null : value),
      eventSanitizerContext.getStore()?.scopeStack,
    );
    const safeNext = next === undefined
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
      Promise.resolve().then(() => (
        safeNext === undefined
          ? (spread ? fn(...arg0, controller.signal) : fn(arg0, controller.signal))
          : (spread ? fn(...arg0, safeNext, controller.signal) : fn(arg0, safeNext, controller.signal))
      )).then(async (value) => {
        if (!streamResult) {
          settlePublication();
          resolve(jsonValue(value === undefined ? null : value));
          return;
        }
        if (value == null || typeof value[Symbol.asyncIterator] !== 'function') {
          throw new TypeError('streaming callback must return an AsyncIterable');
        }
        resolve();
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
            await streamPush(jsonValue(item.value === undefined ? null : item.value));
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
        streamEnd();
        settlePublication();
      }).catch(async (error) => {
        const failure = await settleDownstreams({ error });
        settlePublication();
        const envelope = failureEnvelope(failure.error, 'unknown error');
        reject(envelope.error, envelope.exceptionType);
      });
    };
    eventSanitizerContext.run(token, invoke);
  }

  return {
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

    // Chaining through a resolved promise stores the queue tail before a poll runs, so a
    // next() from inside a poll queues behind it.
    streamNext(poll) {
      const run = () => driveStream(poll, false);
      const current = (streamQueues.get(poll) ?? Promise.resolve()).then(run, run);
      streamQueues.set(poll, current);
      return current;
    },

    streamClose(poll) {
      const existing = streamCloses.get(poll);
      if (existing !== undefined) {
        return existing;
      }
      const closing = Promise.resolve().then(() => driveStream(poll, true));
      streamCloses.set(poll, closing);
      return closing;
    },

    // Keep a handle reachable while its next() or close() is pending so collection does
    // not cut a read short.
    pinStreamMethods(prototype) {
      for (const name of ['next', 'close']) {
        const native = prototype[name];
        Object.defineProperty(prototype, name, {
          configurable: true,
          writable: true,
          value: {
            [name]() {
              const promise = native.call(this);
              pinnedStreams.set(this, (pinnedStreams.get(this) ?? 0) + 1);
              const unpin = () => {
                const left = pinnedStreams.get(this) - 1;
                if (left > 0) {
                  pinnedStreams.set(this, left);
                } else {
                  pinnedStreams.delete(this);
                }
              };
              // Return a derived promise so an ignored rejection stays unhandled.
              return promise.then(
                (value) => {
                  unpin();
                  return value;
                },
                (error) => {
                  unpin();
                  throw error;
                },
              );
            },
          }[name],
        });
      }
    },

    collector(fn) {
      return function __nemo_relay_collector_wrapper(chunk, complete, direct) {
        // A direct call runs inside a consumer's context. Pin a bare store like the
        // threadsafe path does.
        const token = callbackStore({ active: false }, undefined, undefined, undefined, undefined);
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
        const token = callbackStore({ active: true }, undefined, undefined, undefined, undefined);
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

    promise(fn) {
      return function __nemo_relay_promise_wrapper(
        error,
        arg0,
        spread,
        next,
        resolve,
        reject,
        publication,
        publicationContextId,
        scopeStack,
        propagationParentUuid,
        propagationContext,
        registerAbort,
        streamResult,
        streamPush,
        streamEnd,
      ) {
        if (error != null) {
          if (typeof reject === 'function') {
            const failure = failureEnvelope(error, 'unknown error');
            reject(failure.error, failure.exceptionType);
          }
          return;
        }
        callPromise(
          fn,
          arg0,
          spread,
          next,
          resolve,
          reject,
          publication,
          publicationContextId,
          scopeStack,
          propagationParentUuid,
          propagationContext,
          registerAbort,
          streamResult,
          streamPush,
          streamEnd,
        );
      };
    },

    scoped(fn) {
      return function __nemo_relay_scoped_wrapper(
        arg,
        scopeStack,
        propagationParentUuid,
        propagationContext,
      ) {
        const current = eventSanitizerContext.getStore();
        const token = callbackStore(
          current?.publicationState ?? { active: false },
          current?.publicationContextId,
          scopeStack,
          propagationParentUuid,
          propagationContext,
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
      return current?.publicationState.active === true
        ? current.publicationContextId
        : undefined;
    },

    callbackScopeStack() {
      return eventSanitizerContext.getStore()?.scopeStack;
    },

    callbackPropagationParentUuid() {
      return eventSanitizerContext.getStore()?.propagationParentUuid;
    },

    callbackPropagationContext() {
      return eventSanitizerContext.getStore()?.propagationContext;
    },

    withCallbackScopeStack(scopeStack, fn) {
      const current = eventSanitizerContext.getStore();
      const token = callbackStore(
        current?.publicationState ?? { active: false },
        current?.publicationContextId,
        scopeStack,
        undefined,
        undefined,
      );
      const expire = () => expireCallbackStore(token);
      let value;
      try {
        value = eventSanitizerContext.run(token, fn);
      } catch (error) {
        expire();
        throw error;
      }
      if (
        value !== null
        && (typeof value === 'object' || typeof value === 'function')
        && typeof value.then === 'function'
      ) {
        return { active: true, value: Promise.resolve(value).finally(expire) };
      }
      expire();
      return { active: true, value };
    },

    setCallbackScopeStack(scopeStack) {
      const current = eventSanitizerContext.getStore();
      if (current === undefined) {
        return false;
      }
      eventSanitizerContext.enterWith(replacementCallbackStore(current, scopeStack));
      return true;
    },

  };
})()"#;

fn as_unknown<T: NapiRaw>(env: &Env, value: &T) -> JsUnknown {
    unsafe { JsUnknown::from_raw_unchecked(env.raw(), value.raw()) }
}

fn callback_factories(env: &Env) -> napi::Result<JsObject> {
    let global = env.get_global()?;
    if global.has_own_property(CALLBACK_FACTORIES_PROPERTY)? {
        return global.get_named_property(CALLBACK_FACTORIES_PROPERTY);
    }

    let factories: JsObject = env.run_script(CALLBACK_FACTORIES_SOURCE)?;
    let object: JsFunction = global.get_named_property("Object")?;
    let object = unsafe { JsObject::from_raw_unchecked(env.raw(), object.raw()) };
    let define_property: JsFunction = object.get_named_property("defineProperty")?;
    let property = env.create_string(CALLBACK_FACTORIES_PROPERTY)?;
    let mut descriptor = env.create_object()?;
    descriptor.set_named_property("value", factories)?;
    define_property.call(
        None,
        &[
            as_unknown(env, &global),
            as_unknown(env, &property),
            as_unknown(env, &descriptor),
        ],
    )?;

    global.get_named_property(CALLBACK_FACTORIES_PROPERTY)
}

fn wrap_callback(env: &Env, func: &JsFunction, factory_name: &str) -> napi::Result<JsFunction> {
    let factories = callback_factories(env)?;
    let factory: JsFunction = factories.get_named_property(factory_name)?;
    let wrapper = factory.call(None, &[as_unknown(env, func)])?;
    Ok(unsafe { wrapper.cast::<JsFunction>() })
}

fn drive_stream(env: &Env, helper: &str, poll: &JsFunction) -> napi::Result<JsObject> {
    let factories = callback_factories(env)?;
    let helper: JsFunction = factories.get_named_property(helper)?;
    let promise = helper.call(None, &[as_unknown(env, poll)])?;
    Ok(unsafe { promise.cast::<JsObject>() })
}

pub(crate) fn stream_next(env: &Env, poll: &JsFunction) -> napi::Result<JsObject> {
    drive_stream(env, "streamNext", poll)
}

pub(crate) fn stream_close(env: &Env, poll: &JsFunction) -> napi::Result<JsObject> {
    drive_stream(env, "streamClose", poll)
}

pub(crate) fn pin_stream_methods(env: &Env, prototype: &JsObject) -> napi::Result<()> {
    let factories = callback_factories(env)?;
    let helper: JsFunction = factories.get_named_property("pinStreamMethods")?;
    helper.call(None, &[as_unknown(env, prototype)])?;
    Ok(())
}

pub(crate) fn wrap_execution_callback(env: &Env, func: &JsFunction) -> napi::Result<JsFunction> {
    wrap_callback(env, func, "execution")
}

pub(crate) fn wrap_promise_callback(env: &Env, func: &JsFunction) -> napi::Result<JsFunction> {
    wrap_callback(env, func, "promise")
}

pub(crate) fn wrap_collector_callback(env: &Env, func: &JsFunction) -> napi::Result<JsFunction> {
    wrap_callback(env, func, "collector")
}

pub(crate) fn wrap_finalizer_callback(env: &Env, func: &JsFunction) -> napi::Result<JsFunction> {
    wrap_callback(env, func, "finalizer")
}

pub(crate) fn wrap_scoped_callback(env: &Env, func: &JsFunction) -> napi::Result<JsFunction> {
    wrap_callback(env, func, "scoped")
}

pub(crate) fn publication_callback_active(env: &Env) -> napi::Result<bool> {
    let factories = callback_factories(env)?;
    let callback: JsFunction = factories.get_named_property("publicationCallbackActive")?;
    callback
        .call::<JsUnknown>(None, &[])?
        .coerce_to_bool()?
        .get_value()
}

pub(crate) fn event_sanitizer_callback_context_id(env: &Env) -> napi::Result<Option<String>> {
    let factories = callback_factories(env)?;
    let callback: JsFunction = factories.get_named_property("eventSanitizerCallbackContextId")?;
    let value = callback.call::<JsUnknown>(None, &[])?;
    if matches!(value.get_type()?, ValueType::Undefined | ValueType::Null) {
        return Ok(None);
    }
    value
        .coerce_to_string()?
        .into_utf8()?
        .into_owned()
        .map(Some)
}

pub(crate) fn callback_scope_stack(
    env: &Env,
) -> napi::Result<Option<(ScopeStackHandle, Option<PublicationBuffer>)>> {
    let factories = callback_factories(env)?;
    let callback: JsFunction = factories.get_named_property("callbackScopeStack")?;
    let value = callback.call::<JsUnknown>(None, &[])?;
    if matches!(value.get_type()?, ValueType::Undefined | ValueType::Null) {
        return Ok(None);
    }
    let stack = unsafe { <&ScopeStack as FromNapiValue>::from_napi_value(env.raw(), value.raw())? };
    Ok(Some((
        stack.inner.clone(),
        stack.publication_buffer.clone(),
    )))
}

pub(crate) fn callback_propagation_parent_uuid(env: &Env) -> napi::Result<Option<String>> {
    let factories = callback_factories(env)?;
    let callback: JsFunction = factories.get_named_property("callbackPropagationParentUuid")?;
    let value = callback.call::<JsUnknown>(None, &[])?;
    if matches!(value.get_type()?, ValueType::Undefined | ValueType::Null) {
        return Ok(None);
    }
    value
        .coerce_to_string()?
        .into_utf8()?
        .into_owned()
        .map(Some)
}

pub(crate) fn callback_propagation_context(
    env: &Env,
) -> napi::Result<Option<crate::api::PropagationContext>> {
    let factories = callback_factories(env)?;
    let callback: JsFunction = factories.get_named_property("callbackPropagationContext")?;
    let value = callback.call::<JsUnknown>(None, &[])?;
    if matches!(value.get_type()?, ValueType::Undefined | ValueType::Null) {
        return Ok(None);
    }
    unsafe {
        <crate::api::PropagationContext as FromNapiValue>::from_napi_value(env.raw(), value.raw())
            .map(Some)
    }
}

pub(crate) fn with_callback_scope_stack(
    env: &Env,
    stack: &ScopeStack,
    callback: &JsFunction,
) -> napi::Result<Option<JsUnknown>> {
    let factories = callback_factories(env)?;
    let with_stack: JsFunction = factories.get_named_property("withCallbackScopeStack")?;
    let publication_buffer = callback_scope_stack(env)?
        .and_then(|(_, buffer)| buffer)
        .or_else(|| stack.publication_buffer.clone());
    let stack = ScopeStack {
        inner: stack.inner.clone(),
        publication_buffer,
    }
    .into_instance(*env)?;
    let outcome = with_stack.call(None, &[as_unknown(env, &stack), as_unknown(env, callback)])?;
    let outcome = unsafe { JsObject::from_raw_unchecked(env.raw(), outcome.raw()) };
    if !outcome.get_named_property::<bool>("active")? {
        return Ok(None);
    }
    outcome.get_named_property("value").map(Some)
}

pub(crate) fn set_callback_scope_stack(env: &Env, stack: &ScopeStack) -> napi::Result<bool> {
    let factories = callback_factories(env)?;
    let set_stack: JsFunction = factories.get_named_property("setCallbackScopeStack")?;
    let publication_buffer = callback_scope_stack(env)?
        .and_then(|(_, buffer)| buffer)
        .or_else(|| stack.publication_buffer.clone());
    let stack = ScopeStack {
        inner: stack.inner.clone(),
        publication_buffer,
    }
    .into_instance(*env)?;
    set_stack
        .call::<JsUnknown>(None, &[as_unknown(env, &stack)])?
        .coerce_to_bool()?
        .get_value()
}
