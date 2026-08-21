// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cached JavaScript callback wrapper factories for the Node binding.

use napi::bindgen_prelude::FromNapiValue;
use napi::{Env, JsFunction, JsObject, JsUnknown, NapiRaw, NapiValue, ValueType};
use nemo_relay::api::runtime::ScopeStackHandle;
use nemo_relay::api::runtime::subscriber_dispatcher::PublicationBuffer;

use crate::types::ScopeStack;

const CALLBACK_FACTORIES_PROPERTY: &str = "__nemo_relay_callback_factories_v12";

const CALLBACK_FACTORIES_SOURCE: &str = r#"(() => {
  const { AsyncLocalStorage } = process.getBuiltinModule('node:async_hooks');
  const eventSanitizerContext = new AsyncLocalStorage();
  const publicationStates = new Map();
  let nextPublicationContextId = 0;

  function callbackStore(publicationState, publicationContextId, scopeStack, propagationParentUuid) {
    const lifecycle = { expired: false, stores: new Set() };
    const store = {
      lifecycle,
      publicationState,
      publicationContextId,
      scopeStack,
      propagationParentUuid,
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
      propagationParentUuid: current.propagationParentUuid,
    };
    current.lifecycle.stores.add(store);
    if (current.lifecycle.expired) {
      store.scopeStack = null;
      store.propagationParentUuid = undefined;
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
    registerAbort,
    streamResult,
    streamPush,
    streamEnd,
  ) {
    const controller = new AbortController();
    registerAbort(() => controller.abort());
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
        return {
          async *[Symbol.asyncIterator]() {
            try {
              for (;;) {
                const chunk = await downstream.next();
                if (chunk === null) return;
                yield chunk;
              }
            } finally {
              await downstream.close();
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
          throw new TypeError('streaming execution intercept must return an AsyncIterable');
        }
        resolve();
        const iterator = value[Symbol.asyncIterator]();
        let exhausted = false;
        try {
          while (!controller.signal.aborted) {
            const item = await iterator.next();
            if (item.done) {
              exhausted = true;
              break;
            }
            await streamPush(jsonValue(item.value === undefined ? null : item.value));
          }
        } finally {
          if (!exhausted && typeof iterator.return === 'function') {
            await iterator.return();
          }
        }
        streamEnd();
        settlePublication();
      }).catch((error) => {
        settlePublication();
        let message = 'unknown error';
        let exceptionType = 'Error';
        try {
          if (typeof error === 'string') {
            message = error;
          } else if (error === null || (typeof error !== 'object' && typeof error !== 'function')) {
            message = String(error);
          } else if (error != null) {
            const errorMessage = error.message;
            if (typeof errorMessage === 'string') {
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
        reject(message, exceptionType);
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
          let message = 'JavaScript callback failed';
          let exceptionType = 'Error';
          try {
            const errorMessage = error?.message;
            if (typeof errorMessage === 'string') {
              message = errorMessage;
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
        registerAbort,
        streamResult,
        streamPush,
        streamEnd,
      ) {
        if (error != null) {
          let message = 'unknown error';
          let exceptionType = 'Error';
          try {
            message = String(error?.message ?? error);
          } catch {}
          try {
            const errorName = error?.name;
            if (typeof errorName === 'string' && errorName.length > 0) {
              exceptionType = errorName;
            }
          } catch {}
          if (typeof reject === 'function') {
            reject(message, exceptionType);
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
          registerAbort,
          streamResult,
          streamPush,
          streamEnd,
        );
      };
    },

    scopedStream(fn) {
      return function __nemo_relay_scoped_stream_wrapper(arg, scopeStack, propagationParentUuid) {
        const current = eventSanitizerContext.getStore();
        const token = callbackStore(
          current?.publicationState ?? { active: false },
          current?.publicationContextId,
          scopeStack,
          propagationParentUuid,
        );
        return eventSanitizerContext.run(token, () => fn(arg));
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

    withCallbackScopeStack(scopeStack, fn) {
      const current = eventSanitizerContext.getStore();
      const token = callbackStore(
        current?.publicationState ?? { active: false },
        current?.publicationContextId,
        scopeStack,
        current?.propagationParentUuid,
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

    expireCallbackContext() {
      const current = eventSanitizerContext.getStore();
      if (current === undefined) {
        return;
      }
      expireCallbackStore(current);
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

pub(crate) fn wrap_execution_callback(env: &Env, func: &JsFunction) -> napi::Result<JsFunction> {
    wrap_callback(env, func, "execution")
}

pub(crate) fn wrap_promise_callback(env: &Env, func: &JsFunction) -> napi::Result<JsFunction> {
    wrap_callback(env, func, "promise")
}

pub(crate) fn wrap_scoped_stream_callback(
    env: &Env,
    func: &JsFunction,
) -> napi::Result<JsFunction> {
    wrap_callback(env, func, "scopedStream")
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

pub(crate) fn expire_callback_context(env: &Env) -> napi::Result<()> {
    let factories = callback_factories(env)?;
    let expire: JsFunction = factories.get_named_property("expireCallbackContext")?;
    expire.call::<JsUnknown>(None, &[])?;
    Ok(())
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
