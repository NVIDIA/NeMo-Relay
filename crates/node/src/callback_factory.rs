// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cached JavaScript callback wrapper factories for the Node binding.

use napi::{Env, JsFunction, JsObject, JsUnknown, NapiRaw, NapiValue};

const CALLBACK_FACTORIES_PROPERTY: &str = "__nemo_relay_callback_factories_v1";

const CALLBACK_FACTORIES_SOURCE: &str = r#"(() => {
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

    const result = Object.create(null);
    for (const key of Object.keys(value)) {
      result[key] = jsonValue(value[key], seen);
    }
    seen.delete(value);
    return result;
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
      return function __nemo_relay_promise_wrapper(error, arg0, next, resolve, reject) {
        if (error != null) {
          reject(error);
          return;
        }
        Promise.resolve().then(() => (
          next === undefined ? fn(arg0) : fn(arg0, next)
        )).then((value) => jsonValue(value === undefined ? null : value)).then(resolve, reject);
      };
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
