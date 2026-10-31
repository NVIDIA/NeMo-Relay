// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversion utilities for bridging between NeMo Relay core types and NAPI types.
//!
//! Provides helpers to convert errors and optional JSON values between the core
//! runtime representation and the NAPI binding layer.

use std::sync::{LazyLock, Mutex};

use chrono::{DateTime, Utc};
use napi::bindgen_prelude::{
    FnArgs, FromNapiValue, Function, FunctionCallContext, FunctionRef, JsObjectValue, JsValue,
    JsValuesTupleIntoVec, Object, ToNapiValue, Unknown,
};
use napi::threadsafe_function::ThreadsafeFunction;
use napi::{Env, Status, ValueType};
use serde_json::Value as Json;

use crate::error::{Error, Result};

/// JavaScript value passed to or returned from a `Function` call without a scope lifetime.
///
/// Closure and threadsafe callback signatures are fixed at creation. A value built later
/// on the JavaScript thread travels as this carrier. A carrier is valid only inside the
/// JavaScript call or callback that created it.
#[derive(Clone, Copy)]
pub(crate) struct JsArg {
    env: napi::sys::napi_env,
    value: napi::sys::napi_value,
}

impl JsArg {
    /// # Safety
    /// `self` must come from the JavaScript call or callback running now, and `env` must be
    /// that call's environment.
    pub(crate) unsafe fn into_unknown<'env>(self, env: &'env Env) -> Unknown<'env> {
        debug_assert!(
            self.env == env.raw(),
            "JsArg borrowed in a different environment"
        );
        // SAFETY: the caller keeps `self` inside its creating call, so the handle is live, and
        // the result cannot outlive the `env` borrow.
        unsafe { Unknown::from_raw_unchecked(self.env, self.value) }
    }
}

impl ToNapiValue for JsArg {
    unsafe fn to_napi_value(
        env: napi::sys::napi_env,
        val: Self,
    ) -> napi::Result<napi::sys::napi_value> {
        debug_assert!(env == val.env, "JsArg passed to a different environment");
        Ok(val.value)
    }
}

impl FromNapiValue for JsArg {
    unsafe fn from_napi_value(
        env: napi::sys::napi_env,
        value: napi::sys::napi_value,
    ) -> napi::Result<Self> {
        Ok(Self { env, value })
    }
}

pub(crate) struct Callback<'env>(Function<'env, JsArg, JsArg>);

impl<'env> std::ops::Deref for Callback<'env> {
    type Target = Function<'env, JsArg, JsArg>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl FromNapiValue for Callback<'_> {
    unsafe fn from_napi_value(
        env: napi::sys::napi_env,
        value: napi::sys::napi_value,
    ) -> napi::Result<Self> {
        let unknown = unsafe { Unknown::from_raw_unchecked(env, value) };
        match unknown.get_type()? {
            ValueType::Function => Ok(Self(unsafe { Function::from_napi_value(env, value) }?)),
            other => Err(napi::Error::new(
                Status::InvalidArg,
                format!("callback must be a function, got {other}"),
            )),
        }
    }
}

pub(crate) type JsArgs2 = FnArgs<(JsArg, JsArg)>;
pub(crate) type JsArgs3 = FnArgs<(JsArg, JsArg, JsArg)>;

/// Weak threadsafe function. JavaScript receives the call arguments only.
pub(crate) type Tsfn<T, Args = T, Return = ()> =
    ThreadsafeFunction<T, Return, Args, Status, false, true>;

/// Weak threadsafe function. JavaScript receives an error or null before the call arguments.
pub(crate) type HandledTsfn<T, Args = T, Return = ()> =
    ThreadsafeFunction<T, Return, Args, Status, true, true>;

/// JavaScript function kept alive for later direct calls on the creating thread.
pub(crate) struct PersistentJsFunction<Args: JsValuesTupleIntoVec, Return> {
    env: Env,
    reference: FunctionRef<Args, Return>,
}

// SAFETY: `env` is read on the creating thread only. Callers check the current thread
// before borrowing the function back. `FunctionRef` is `Send` and `Sync` and routes a
// release from another thread back to the JavaScript thread.
unsafe impl<Args: JsValuesTupleIntoVec, Return> Send for PersistentJsFunction<Args, Return> {}
// SAFETY: as for `Send`. Both fields are immutable after construction.
unsafe impl<Args: JsValuesTupleIntoVec, Return> Sync for PersistentJsFunction<Args, Return> {}

impl<Args: JsValuesTupleIntoVec, Return> PersistentJsFunction<Args, Return> {
    pub(crate) fn new(env: &Env, function: &Function<'_, Args, Return>) -> napi::Result<Self> {
        Ok(Self {
            env: *env,
            reference: function.create_ref()?,
        })
    }

    pub(crate) fn env(&self) -> Env {
        self.env
    }

    pub(crate) fn function(&self) -> napi::Result<Function<'_, Args, Return>> {
        self.reference.borrow_back(&self.env)
    }
}

pub(crate) fn js_value<T: ToNapiValue>(env: &Env, value: T) -> napi::Result<JsArg> {
    // SAFETY: `env` is the live environment of the calling JavaScript thread.
    let value = unsafe { T::to_napi_value(env.raw(), value) }?;
    Ok(JsArg {
        env: env.raw(),
        value,
    })
}

pub(crate) fn detach<'env>(env: &Env, value: &impl JsValue<'env>) -> JsArg {
    JsArg {
        env: env.raw(),
        value: value.raw(),
    }
}

pub(crate) fn closure_function<'env, R, F>(
    env: &'env Env,
    name: &str,
    callback: F,
) -> napi::Result<Function<'env, JsArg, R>>
where
    R: ToNapiValue,
    F: 'static + Fn(FunctionCallContext) -> Result<R>,
{
    env.create_function_from_closure(name, move |ctx| {
        let env = *ctx.env;
        callback(ctx).map_err(|error| error.throw_into(&env))
    })
}

pub(crate) fn abort_tsfn<T, Return, Args, const CALLEE_HANDLED: bool, const WEAK: bool>(
    tsfn: &ThreadsafeFunction<T, Return, Args, Status, CALLEE_HANDLED, WEAK>,
) -> napi::Result<()>
where
    T: 'static,
    Return: 'static + FromNapiValue,
    Args: 'static + JsValuesTupleIntoVec,
{
    tsfn.handle.with_write_aborted(|mut aborted| {
        if !*aborted {
            // SAFETY: the handle is live until this release.
            napi::check_status!(unsafe {
                napi::sys::napi_release_threadsafe_function(
                    tsfn.handle.get_raw(),
                    napi::sys::ThreadsafeFunctionReleaseMode::abort,
                )
            })?;
            *aborted = true;
        }
        Ok(())
    })
}

static LAST_CALLBACK_ERROR: LazyLock<Mutex<Option<String>>> = LazyLock::new(|| Mutex::new(None));

/// Record the most recent callback error observed inside the Node binding.
pub fn record_callback_error(message: impl Into<String>) {
    let message = message.into();
    eprintln!("{message}");
    if let Ok(mut guard) = LAST_CALLBACK_ERROR.lock() {
        *guard = Some(message);
    }
}

/// Read the most recent callback error observed inside the Node binding.
pub fn get_last_callback_error() -> Option<String> {
    LAST_CALLBACK_ERROR
        .lock()
        .ok()
        .and_then(|guard| guard.clone())
}

/// Clear the most recent callback error observed inside the Node binding.
pub fn clear_last_callback_error() {
    if let Ok(mut guard) = LAST_CALLBACK_ERROR.lock() {
        *guard = None;
    }
}

/// Filter an optional JSON value, converting explicit `null` values to `None`.
///
/// NAPI's serde-json feature handles most conversion automatically, but JavaScript
/// may pass `null` where Rust expects `None`. This normalizes that case.
pub fn opt_json(val: Option<Json>) -> Option<Json> {
    val.filter(|v| !v.is_null())
}

/// Convert JSON input. Throw a TypeError for a BigInt instead of coercing it.
/// Read properties once so getters cannot bypass the BigInt check.
pub(crate) fn checked_json(env: &Env, value: Unknown) -> Result<Json> {
    match value.get_type()? {
        ValueType::BigInt => {
            env.throw_type_error("BigInt cannot be represented as JSON", None)?;
            Err(Error::Napi(napi::Error::new(
                Status::PendingException,
                "BigInt cannot be represented as JSON",
            )))
        }
        ValueType::Object => {
            let object = Object::from_unknown(value)?;
            if object.is_array()? {
                let mut values = Vec::new();
                for index in 0..object.get_array_length()? {
                    values.push(checked_json(env, object.get_element(index)?)?);
                }
                Ok(Json::Array(values))
            } else {
                let mut values = serde_json::Map::new();
                for key in Object::keys(&object)? {
                    if let Some(value) = object.get::<Unknown>(&key)? {
                        values.insert(key, checked_json(env, value)?);
                    }
                }
                Ok(Json::Object(values))
            }
        }
        _ => Ok(Json::from_unknown(value)?),
    }
}

/// Parse optional Unix microseconds since epoch into a UTC timestamp.
pub fn parse_timestamp_micros(value: Option<f64>) -> Result<Option<DateTime<Utc>>> {
    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    value
        .map(|timestamp| {
            if !timestamp.is_finite()
                || timestamp.fract() != 0.0
                || timestamp.abs() > MAX_SAFE_INTEGER
            {
                return Err(Error::invalid_argument(
                    "timestamp must be a safe integer number of Unix microseconds",
                ));
            }
            DateTime::<Utc>::from_timestamp_micros(timestamp as i64).ok_or_else(|| {
                Error::invalid_argument(
                    "invalid timestamp: unix microseconds are outside supported range",
                )
            })
        })
        .transpose()
}
