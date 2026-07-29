// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cached JavaScript callback wrapper factories for the Node binding.

use napi::bindgen_prelude::FromNapiValue;
use napi::{Env, JsFunction, JsObject, JsUnknown, NapiRaw, NapiValue, ValueType};
use nemo_relay::api::runtime::ScopeStackHandle;

use crate::types::ScopeStack;

const CALLBACK_FACTORIES_PROPERTY: &str = "__nemo_relay_callback_factories_v3";

const CALLBACK_FACTORIES_SOURCE: &str = r#"(() => {
  const { AsyncLocalStorage } = process.getBuiltinModule('node:async_hooks');
  const eventSanitizerContext = new AsyncLocalStorage();

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

  function callPromise(fn, arg0, spread, next, resolve, reject, publication, scopeStack) {
    const token = { publicationState: { active: publication }, scopeStack };
    const invoke = () => {
      Promise.resolve().then(() => (
        next === undefined
          ? (spread ? fn(...arg0) : fn(arg0))
          : (spread ? fn(...arg0, next) : fn(arg0, next))
      )).then((value) => jsonValue(value === undefined ? null : value)).then((value) => {
        token.publicationState.active = false;
        token.scopeStack = null;
        resolve(value);
      }, (error) => {
        token.publicationState.active = false;
        token.scopeStack = null;
        let message = 'unknown error';
        try {
          if (typeof error === 'string') {
            message = error;
          } else if (error === null || (typeof error !== 'object' && typeof error !== 'function')) {
            message = String(error);
          } else if (error != null && typeof error.message === 'string') {
            message = error.message;
          }
        } catch {}
        reject(message);
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
          try {
            message = String(error?.message ?? error);
          } catch {}
          return { ok: false, error: message };
        }
      };
    },

    promise(fn) {
      return function __nemo_relay_promise_wrapper(error, arg0, spread, next, resolve, reject, publication, scopeStack) {
        if (error != null) {
          let message = 'unknown error';
          try {
            message = String(error?.message ?? error);
          } catch {}
          reject(message);
          return;
        }
        callPromise(fn, arg0, spread, next, resolve, reject, publication, scopeStack);
      };
    },

    eventSanitizerCallbackActive() {
      return eventSanitizerContext.getStore()?.publicationState.active === true;
    },

    callbackScopeStack() {
      return eventSanitizerContext.getStore()?.scopeStack;
    },

    withCallbackScopeStack(scopeStack, fn) {
      const current = eventSanitizerContext.getStore();
      if (current === undefined) {
        return { active: false };
      }
      const token = { publicationState: current.publicationState, scopeStack };
      try {
        return { active: true, value: eventSanitizerContext.run(token, fn) };
      } finally {
        token.scopeStack = null;
      }
    },

    setCallbackScopeStack(scopeStack) {
      const current = eventSanitizerContext.getStore();
      if (current === undefined) {
        return false;
      }
      current.scopeStack = scopeStack;
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

pub(crate) fn wrap_execution_callback(env: &Env, func: &JsFunction) -> napi::Result<JsFunction> {
    wrap_callback(env, func, "execution")
}

pub(crate) fn wrap_promise_callback(env: &Env, func: &JsFunction) -> napi::Result<JsFunction> {
    wrap_callback(env, func, "promise")
}

pub(crate) fn event_sanitizer_callback_active(env: &Env) -> napi::Result<bool> {
    let factories = callback_factories(env)?;
    let callback: JsFunction = factories.get_named_property("eventSanitizerCallbackActive")?;
    callback
        .call::<JsUnknown>(None, &[])?
        .coerce_to_bool()?
        .get_value()
}

pub(crate) fn callback_scope_stack(env: &Env) -> napi::Result<Option<ScopeStackHandle>> {
    let factories = callback_factories(env)?;
    let callback: JsFunction = factories.get_named_property("callbackScopeStack")?;
    let value = callback.call::<JsUnknown>(None, &[])?;
    if matches!(value.get_type()?, ValueType::Undefined | ValueType::Null) {
        return Ok(None);
    }
    let stack = unsafe { <&ScopeStack as FromNapiValue>::from_napi_value(env.raw(), value.raw())? };
    Ok(Some(stack.inner.clone()))
}

pub(crate) fn with_callback_scope_stack(
    env: &Env,
    stack: &ScopeStack,
    callback: &JsFunction,
) -> napi::Result<Option<JsUnknown>> {
    let factories = callback_factories(env)?;
    let with_stack: JsFunction = factories.get_named_property("withCallbackScopeStack")?;
    let stack = ScopeStack::from(stack.inner.clone()).into_instance(*env)?;
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
    let stack = ScopeStack::from(stack.inner.clone()).into_instance(*env)?;
    set_stack
        .call::<JsUnknown>(None, &[as_unknown(env, &stack)])?
        .coerce_to_bool()?
        .get_value()
}
