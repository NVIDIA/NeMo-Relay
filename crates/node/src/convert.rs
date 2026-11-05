// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::pin::Pin;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;

use chrono::{DateTime, Utc};
use napi::bindgen_prelude::{
    FnArgs, FromNapiValue, Function, FunctionCallContext, FunctionRef, JsObjectValue, JsValue,
    JsValuesTupleIntoVec, Object, ToNapiValue, Unknown,
};
use napi::threadsafe_function::ThreadsafeFunction;
use napi::{Env, Status, ValueType};
use serde::de::DeserializeOwned;
use serde_json::Value as Json;

use crate::error::{Error, Result};

pub(crate) type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

pub(crate) type JsBuilder = Box<dyn FnOnce(&Env) -> napi::Result<JsArg> + Send>;

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
        match napi::type_of!(env, value)? {
            // SAFETY: the value was just typed as a function in `env`.
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

pub(crate) type Tsfn<T, Args = T, Return = ()> =
    ThreadsafeFunction<T, Return, Args, Status, false, true>;

pub(crate) type HandledTsfn<T, Args = T, Return = ()> =
    ThreadsafeFunction<T, Return, Args, Status, true, true>;

pub(crate) const JAVASCRIPT_MAX_SAFE_INTEGER: u64 = (1 << 53) - 1;

const MAX_JSON_DEPTH: u32 = 128;

const MAX_JSON_NODES: usize = 1_000_000;

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) struct PersistentJsFunction<Args: JsValuesTupleIntoVec, Return> {
    env: Env,
    thread: ThreadId,
    reference: FunctionRef<Args, Return>,
}

// SAFETY: `env` is dereferenced only by `borrow`, which refuses every thread but the
// creating one. `FunctionRef` is `Send` and `Sync` and routes a release from another
// thread back to the JavaScript thread.
unsafe impl<Args: JsValuesTupleIntoVec, Return> Send for PersistentJsFunction<Args, Return> {}
// SAFETY: as for `Send`. All fields are immutable after construction.
unsafe impl<Args: JsValuesTupleIntoVec, Return> Sync for PersistentJsFunction<Args, Return> {}

impl<Args: JsValuesTupleIntoVec, Return> PersistentJsFunction<Args, Return> {
    pub(crate) fn new(env: &Env, function: &Function<'_, Args, Return>) -> napi::Result<Self> {
        Ok(Self {
            env: *env,
            thread: std::thread::current().id(),
            reference: function.create_ref()?,
        })
    }

    pub(crate) fn on_creating_thread(&self) -> bool {
        std::thread::current().id() == self.thread
    }

    pub(crate) fn borrow(&self) -> napi::Result<(&Env, Function<'_, Args, Return>)> {
        if !self.on_creating_thread() {
            return Err(napi::Error::new(
                Status::GenericFailure,
                "JavaScript function borrowed outside its creating thread",
            ));
        }
        Ok((&self.env, self.reference.borrow_back(&self.env)?))
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
    env.create_function_from_closure(name, move |ctx| callback(ctx).map_err(Error::into_napi))
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

pub(crate) async fn settled<T: Clone>(
    mut receiver: tokio::sync::watch::Receiver<Option<T>>,
) -> Option<T> {
    receiver
        .wait_for(Option::is_some)
        .await
        .ok()
        .and_then(|value| value.clone())
}

pub(crate) fn report_queue_status(status: Status, call: &str) {
    if status != Status::Ok && status != Status::Closing {
        record_callback_error(format!("nemo_relay: failed to queue {call}: {status:?}"));
    }
}

static LAST_CALLBACK_ERROR: Mutex<Option<String>> = Mutex::new(None);

pub(crate) fn record_callback_error(message: impl Into<String>) {
    let message = message.into();
    eprintln!("{message}");
    *lock(&LAST_CALLBACK_ERROR) = Some(message);
}

pub(crate) fn get_last_callback_error() -> Option<String> {
    lock(&LAST_CALLBACK_ERROR).clone()
}

pub(crate) fn clear_last_callback_error() {
    *lock(&LAST_CALLBACK_ERROR) = None;
}

pub(crate) fn opt_json(val: Option<Json>) -> Option<Json> {
    val.filter(|v| !v.is_null())
}

pub(crate) fn parse_json<T: DeserializeOwned>(value: Json, what: &str) -> Result<T> {
    serde_json::from_value(value)
        .map_err(|error| Error::invalid_argument(format!("invalid {what}: {error}")))
}

pub(crate) fn checked_json(env: &Env, value: Unknown) -> Result<Json> {
    let mut remaining_nodes = MAX_JSON_NODES;
    checked_json_within(env, value, MAX_JSON_DEPTH, &mut remaining_nodes)
}

fn checked_json_within(
    env: &Env,
    value: Unknown,
    remaining_depth: u32,
    remaining_nodes: &mut usize,
) -> Result<Json> {
    *remaining_nodes = remaining_nodes
        .checked_sub(1)
        .ok_or_else(|| Error::invalid_argument("Exceeded maximum JSON value count"))?;
    match value.get_type()? {
        ValueType::BigInt => {
            env.throw_type_error("BigInt cannot be represented as JSON", None)?;
            Err(Error::Napi(napi::Error::new(
                Status::PendingException,
                "BigInt cannot be represented as JSON",
            )))
        }
        ValueType::Object => {
            let Some(remaining_depth) = remaining_depth.checked_sub(1) else {
                return Err(Error::invalid_argument(
                    "Exceeded maximum JSON nesting depth",
                ));
            };
            let object = Object::from_unknown(value)?;
            if object.is_array()? {
                let mut values = Vec::new();
                for index in 0..object.get_array_length()? {
                    values.push(checked_json_within(
                        env,
                        object.get_element(index)?,
                        remaining_depth,
                        remaining_nodes,
                    )?);
                }
                Ok(Json::Array(values))
            } else {
                let keys = Object::keys(&object)?;
                let mut values = serde_json::Map::with_capacity(keys.len());
                for key in keys {
                    if let Some(value) = object.get::<Unknown>(&key)? {
                        values.insert(
                            key,
                            checked_json_within(env, value, remaining_depth, remaining_nodes)?,
                        );
                    }
                }
                Ok(Json::Object(values))
            }
        }
        _ => Ok(Json::from_unknown(value)?),
    }
}

pub(crate) fn parse_timestamp_micros(value: Option<f64>) -> Result<Option<DateTime<Utc>>> {
    value
        .map(|timestamp| {
            if !timestamp.is_finite()
                || timestamp.fract() != 0.0
                || timestamp.abs() > JAVASCRIPT_MAX_SAFE_INTEGER as f64
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
