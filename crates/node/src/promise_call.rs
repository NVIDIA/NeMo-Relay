// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Promise-aware JS function calling for NeMo Relay NAPI bindings.
//!
//! This module wraps JS middleware callbacks so Rust can call them from any thread
//! and await either synchronous return values or Promise-returning callbacks.
//!
//! The previous implementation used a raw `napi_threadsafe_function` with a custom
//! `call_js_cb`. That path was prone to lifecycle issues under `node --test`.
//! This implementation keeps the same surface API but delegates the underlying
//! TSFN lifecycle to `napi-rs`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use napi::bindgen_prelude::ToNapiValue;
use napi::threadsafe_function::{ThreadSafeCallContext, ThreadsafeFunction};
use napi::{Env, JsFunction, JsUnknown, NapiRaw, NapiValue};
use serde_json::Value as Json;

use nemo_relay::error::{FlowError, Result as FlowResult};

pub type JsonNextFn =
    Arc<dyn Fn(Json) -> Pin<Box<dyn Future<Output = FlowResult<Json>> + Send>> + Send + Sync>;
pub type JsonStreamNextFn =
    Arc<dyn Fn(Json) -> Pin<Box<dyn Future<Output = FlowResult<Vec<Json>>> + Send>> + Send + Sync>;

#[derive(Clone)]
enum NextFn {
    Json(JsonNextFn),
    Stream(JsonStreamNextFn),
}

/// Builds the first JS callback argument on the Node main thread.
///
/// Some callback arguments, such as `#[napi]` class instances, cannot cross the
/// threadsafe-function boundary as plain JSON. This builder runs inside the
/// threadsafe-function call (on the JS thread), so it can materialize those
/// values directly instead of serializing them.
pub type Arg0Builder = Box<dyn FnOnce(&Env) -> napi::Result<JsUnknown> + Send>;

/// The first argument passed to the wrapped JS callback.
enum PrimaryArg {
    /// A plain JSON value converted on the JS thread.
    Json(Json),
    /// A value materialized on the JS thread by a builder closure.
    Build(Arg0Builder),
}

struct CallArgs {
    arg0: PrimaryArg,
    next: Option<NextFn>,
    completion: CallCompletion,
}

#[derive(Clone)]
struct CallCompletion {
    sender: Arc<std::sync::Mutex<Option<tokio::sync::oneshot::Sender<FlowResult<Json>>>>>,
}

impl CallCompletion {
    fn new(sender: tokio::sync::oneshot::Sender<FlowResult<Json>>) -> Self {
        Self {
            sender: Arc::new(std::sync::Mutex::new(Some(sender))),
        }
    }

    fn send(&self, value: FlowResult<Json>) {
        if let Some(sender) = self.sender.lock().unwrap().take() {
            let _ = sender.send(value);
        }
    }
}

fn rejection_message(
    string_result: napi::Result<String>,
    object_message_result: Option<napi::Result<String>>,
) -> String {
    if let Ok(value) = string_result {
        value
    } else if let Some(message_result) = object_message_result {
        message_result.unwrap_or_else(|_| "unknown error".to_string())
    } else {
        "unknown error".to_string()
    }
}

fn closed_tsfn_error() -> FlowError {
    FlowError::Internal("PromiseAwareFn threadsafe function closed".into())
}

fn queue_status_result(status: napi::Status) -> FlowResult<()> {
    if status == napi::Status::Ok {
        Ok(())
    } else {
        Err(FlowError::Internal(format!(
            "failed to queue threadsafe function call: {status:?}",
        )))
    }
}

fn json_to_unknown(env: &Env, value: Json) -> napi::Result<JsUnknown> {
    let raw = unsafe { <Json as ToNapiValue>::to_napi_value(env.raw(), value) }?;
    Ok(unsafe { JsUnknown::from_raw_unchecked(env.raw(), raw) })
}

fn function_to_unknown(env: &Env, value: &JsFunction) -> JsUnknown {
    unsafe { JsUnknown::from_raw_unchecked(env.raw(), value.raw()) }
}

fn undefined_to_unknown(env: &Env) -> napi::Result<JsUnknown> {
    let value = env.get_undefined()?;
    Ok(unsafe { JsUnknown::from_raw_unchecked(env.raw(), value.raw()) })
}

fn build_next_unknown(env: &Env, next: NextFn) -> napi::Result<JsUnknown> {
    let next_fn = match next {
        NextFn::Json(next) => {
            env.create_function_from_closure("__nemo_relay_next", move |ctx| {
                let arg = ctx.get::<Json>(0).unwrap_or(Json::Null);
                let next = next.clone();
                ctx.env.execute_tokio_future(
                    async move {
                        next(arg)
                            .await
                            .map_err(|e| napi::Error::from_reason(e.to_string()))
                    },
                    |_env, value| Ok(value),
                )
            })?
        }
        NextFn::Stream(next) => {
            env.create_function_from_closure("__nemo_relay_next", move |ctx| {
                let arg = ctx.get::<Json>(0).unwrap_or(Json::Null);
                let next = next.clone();
                ctx.env.execute_tokio_future(
                    async move {
                        next(arg)
                            .await
                            .map_err(|e| napi::Error::from_reason(e.to_string()))
                    },
                    |_env, value| Ok(value),
                )
            })?
        }
    };

    Ok(function_to_unknown(env, &next_fn))
}

fn build_completion_unknowns(
    env: &Env,
    completion: CallCompletion,
) -> napi::Result<(JsUnknown, JsUnknown)> {
    let resolve_completion = completion.clone();
    let resolve = env.create_function_from_closure("__nemo_relay_resolve", move |ctx| {
        let value = ctx.get::<Json>(0).map_err(|error| {
            FlowError::Internal(format!(
                "JavaScript callback result could not be converted to JSON: {error}"
            ))
        });
        resolve_completion.send(value);
        ctx.env.get_undefined()
    })?;

    let reject = env.create_function_from_closure("__nemo_relay_reject", move |ctx| {
        let message = rejection_message(
            ctx.get::<String>(0),
            ctx.get::<napi::JsObject>(0)
                .ok()
                .map(|value| value.get_named_property::<String>("message")),
        );
        completion.send(Err(FlowError::Internal(message)));
        ctx.env.get_undefined()
    })?;

    Ok((
        function_to_unknown(env, &resolve),
        function_to_unknown(env, &reject),
    ))
}

fn create_promise_wrapper(env: &Env, callable: &JsFunction) -> napi::Result<JsFunction> {
    let factory: JsFunction = env.run_script(
        r#"((fn) => {
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

  return function __nemo_relay_promise_wrapper(error, arg0, next, resolve, reject) {
  if (error != null) {
    reject(error);
    return;
  }
  Promise.resolve().then(() => (
    next === undefined ? fn(arg0) : fn(arg0, next)
  )).then((value) => jsonValue(value === undefined ? null : value)).then(resolve, reject);
  };
})"#,
    )?;
    let wrapper_unknown: JsUnknown = factory.call(None, &[function_to_unknown(env, callable)])?;
    Ok(unsafe { wrapper_unknown.cast::<JsFunction>() })
}

/// A wrapper around a JS function that can be called from any thread and
/// transparently handles both synchronous and Promise return values.
pub struct PromiseAwareFn {
    tsfn: std::sync::Mutex<Option<ThreadsafeFunction<CallArgs>>>,
}

impl PromiseAwareFn {
    /// Create a new `PromiseAwareFn` wrapping the given JS function.
    ///
    /// Must be called on the JS main thread (i.e., in a sync `#[napi]` function).
    pub fn new(env: &Env, func: &JsFunction) -> napi::Result<Self> {
        let wrapper = create_promise_wrapper(env, func)?;
        let mut tsfn =
            env.create_threadsafe_function(&wrapper, 0, |ctx: ThreadSafeCallContext<CallArgs>| {
                let next = match ctx.value.next {
                    Some(next) => build_next_unknown(&ctx.env, next)?,
                    None => undefined_to_unknown(&ctx.env)?,
                };
                let (resolve, reject) = build_completion_unknowns(&ctx.env, ctx.value.completion)?;
                let arg0 = match ctx.value.arg0 {
                    PrimaryArg::Json(value) => json_to_unknown(&ctx.env, value)?,
                    PrimaryArg::Build(build) => build(&ctx.env)?,
                };

                let args = vec![arg0, next, resolve, reject];
                Ok(args)
            })?;

        // The callback should not keep the Node event loop alive on its own.
        tsfn.unref(env)?;

        Ok(Self {
            tsfn: std::sync::Mutex::new(Some(tsfn)),
        })
    }

    /// Call the JS function with the given args and await the result.
    pub async fn call(&self, args: Json) -> FlowResult<Json> {
        self.call_inner(PrimaryArg::Json(args), None).await
    }

    /// Call the JS function with a builder-constructed first argument and await
    /// the result.
    ///
    /// The builder runs on the Node main thread, so it can construct values that
    /// cannot cross the threadsafe-function boundary as plain JSON, such as a
    /// `#[napi]` class instance.
    pub async fn call_with_arg0(&self, build_arg0: Arg0Builder) -> FlowResult<Json> {
        self.call_inner(PrimaryArg::Build(build_arg0), None).await
    }

    /// Call the JS function with a middleware-style `next(arg)` callback that
    /// resolves to a JSON result.
    pub async fn call_with_json_next(&self, args: Json, next: JsonNextFn) -> FlowResult<Json> {
        self.call_inner(PrimaryArg::Json(args), Some(NextFn::Json(next)))
            .await
    }

    /// Call the JS function with a middleware-style `next(arg)` callback that
    /// resolves to an array of downstream stream chunks.
    pub async fn call_with_stream_next(
        &self,
        args: Json,
        next: JsonStreamNextFn,
    ) -> FlowResult<Json> {
        self.call_inner(PrimaryArg::Json(args), Some(NextFn::Stream(next)))
            .await
    }

    /// Release the underlying threadsafe function so it does not outlive its registration.
    pub fn close(&self) {
        if let Some(tsfn) = self.tsfn.lock().unwrap().take() {
            let _ = tsfn.abort();
        }
    }

    async fn call_inner(&self, arg0: PrimaryArg, next: Option<NextFn>) -> FlowResult<Json> {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let tsfn = self
            .tsfn
            .lock()
            .unwrap()
            .as_ref()
            .cloned()
            .ok_or_else(closed_tsfn_error)?;
        let status = tsfn.call(
            Ok(CallArgs {
                arg0,
                next,
                completion: CallCompletion::new(sender),
            }),
            napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
        );
        queue_status_result(status)?;

        receiver
            .await
            .map_err(|e| FlowError::Internal(e.to_string()))?
    }
}

impl Drop for PromiseAwareFn {
    fn drop(&mut self) {
        if let Some(tsfn) = self.tsfn.get_mut().unwrap().take() {
            let _ = tsfn.abort();
        }
    }
}
