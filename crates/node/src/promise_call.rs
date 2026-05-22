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

struct CallArgs {
    args: Json,
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
        let value = ctx.get::<Json>(0).unwrap_or(Json::Null);
        resolve_completion.send(Ok(value));
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
        r#"((fn) => function __nemo_relay_promise_wrapper(error, arg0, next, resolve, reject) {
  if (error != null) {
    reject(error);
    return;
  }
  Promise.resolve().then(() => (
    next === undefined ? fn(arg0) : fn(arg0, next)
  )).then(resolve, reject);
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

                let args = vec![
                    json_to_unknown(&ctx.env, ctx.value.args)?,
                    next,
                    resolve,
                    reject,
                ];
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
        self.call_inner(args, None).await
    }

    /// Call the JS function with a middleware-style `next(arg)` callback that
    /// resolves to a JSON result.
    pub async fn call_with_json_next(&self, args: Json, next: JsonNextFn) -> FlowResult<Json> {
        self.call_inner(args, Some(NextFn::Json(next))).await
    }

    /// Call the JS function with a middleware-style `next(arg)` callback that
    /// resolves to an array of downstream stream chunks.
    pub async fn call_with_stream_next(
        &self,
        args: Json,
        next: JsonStreamNextFn,
    ) -> FlowResult<Json> {
        self.call_inner(args, Some(NextFn::Stream(next))).await
    }

    /// Release the underlying threadsafe function so it does not outlive its registration.
    pub fn close(&self) {
        if let Some(tsfn) = self.tsfn.lock().unwrap().take() {
            let _ = tsfn.abort();
        }
    }

    async fn call_inner(&self, args: Json, next: Option<NextFn>) -> FlowResult<Json> {
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
                args,
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
