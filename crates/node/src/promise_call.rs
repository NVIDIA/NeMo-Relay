// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Promise-aware JS function calling for Nexus NAPI bindings.
//!
//! Provides [`PromiseAwareFn`], a cross-thread callable wrapper around a JS function
//! that transparently handles both synchronous return values and Promise-returning
//! (async) callbacks. This solves the fundamental limitation of NAPI's
//! `call_with_return_value` which cannot resolve JS Promises — they serialize to `{}`
//! because Promise properties are non-enumerable.
//!
//! The implementation uses a raw `napi_threadsafe_function` with a custom `call_js_cb`
//! that inspects the return value via `napi_is_promise` and either:
//! - Converts synchronous results directly to JSON
//! - Attaches `.then(resolve, reject)` handlers to Promises that push resolved values
//!   through a oneshot channel

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use napi::bindgen_prelude::{FromNapiValue, ToNapiValue};
use napi::{Env, JsFunction, NapiRaw};
use serde_json::Value as Json;

use nvidia_nat_nexus_core::{NexusError, Result as NexusResult};

// ---------------------------------------------------------------------------
// Channel registry for pending call results
// ---------------------------------------------------------------------------

static NEXT_CALL_ID: AtomicU64 = AtomicU64::new(0);
type ResultSender = tokio::sync::oneshot::Sender<NexusResult<Json>>;

static PENDING_CALLS: std::sync::LazyLock<Mutex<HashMap<u64, ResultSender>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

fn register_call(id: u64, tx: ResultSender) {
    PENDING_CALLS.lock().unwrap().insert(id, tx);
}

fn send_result(call_id: u64, value: Json) {
    if let Some(tx) = PENDING_CALLS.lock().unwrap().remove(&call_id) {
        let _ = tx.send(Ok(value));
    }
}

fn send_error(call_id: u64, msg: String) {
    if let Some(tx) = PENDING_CALLS.lock().unwrap().remove(&call_id) {
        let _ = tx.send(Err(NexusError::Internal(msg)));
    }
}

// ---------------------------------------------------------------------------
// Data passed through the threadsafe function
// ---------------------------------------------------------------------------

struct CallData {
    call_id: u64,
    args_json: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Custom call_js_cb — runs on the JS main thread
// ---------------------------------------------------------------------------

/// The custom callback invoked by `napi_call_threadsafe_function` on the JS thread.
///
/// This function:
/// 1. Deserializes the args from JSON
/// 2. Calls the JS callback function
/// 3. Checks if the return value is a Promise (`napi_is_promise`)
/// 4. For sync values: converts to JSON and sends through the channel
/// 5. For Promises: attaches `.then(resolve, reject)` handlers
unsafe extern "C" fn promise_call_js_cb(
    raw_env: napi::sys::napi_env,
    js_callback: napi::sys::napi_value,
    _context: *mut c_void,
    data: *mut c_void,
) {
    use napi::sys;

    if data.is_null() || raw_env.is_null() {
        return;
    }

    let call_data = unsafe { Box::from_raw(data as *mut CallData) };
    let call_id = call_data.call_id;

    // Deserialize args
    let args: Json = match serde_json::from_slice(&call_data.args_json) {
        Ok(v) => v,
        Err(e) => {
            send_error(call_id, format!("failed to deserialize args: {e}"));
            return;
        }
    };

    // Convert Json to napi_value
    let js_args = match unsafe { <Json as ToNapiValue>::to_napi_value(raw_env, args) } {
        Ok(v) => v,
        Err(e) => {
            send_error(call_id, format!("failed to convert args to JS: {e}"));
            return;
        }
    };

    // Call the JS function
    let mut result: sys::napi_value = std::ptr::null_mut();
    let mut global: sys::napi_value = std::ptr::null_mut();
    unsafe {
        sys::napi_get_global(raw_env, &mut global);
    }

    let status =
        unsafe { sys::napi_call_function(raw_env, global, js_callback, 1, &js_args, &mut result) };

    // Check for exceptions
    if status != sys::Status::napi_ok {
        let mut is_exception = false;
        unsafe { sys::napi_is_exception_pending(raw_env, &mut is_exception) };
        if is_exception {
            let mut error_val: sys::napi_value = std::ptr::null_mut();
            unsafe { sys::napi_get_and_clear_last_exception(raw_env, &mut error_val) };

            // Try to extract error message
            let msg = extract_error_message(raw_env, error_val)
                .unwrap_or_else(|| "JS function threw an exception".to_string());
            send_error(call_id, msg);
        } else {
            send_error(call_id, "JS function call failed".to_string());
        }
        return;
    }

    // Handle null/undefined result
    if result.is_null() {
        send_result(call_id, Json::Null);
        return;
    }

    let mut value_type: sys::napi_valuetype = sys::ValueType::napi_undefined;
    unsafe { sys::napi_typeof(raw_env, result, &mut value_type) };
    if value_type == sys::ValueType::napi_undefined || value_type == sys::ValueType::napi_null {
        send_result(call_id, Json::Null);
        return;
    }

    // Check if result is a Promise
    let mut is_promise = false;
    unsafe { sys::napi_is_promise(raw_env, result, &mut is_promise) };

    if !is_promise {
        // Synchronous result — convert directly to JSON
        match unsafe { <Json as FromNapiValue>::from_napi_value(raw_env, result) } {
            Ok(json) => send_result(call_id, json),
            Err(e) => send_error(call_id, format!("failed to convert result to JSON: {e}")),
        }
        return;
    }

    // Promise result — attach .then(resolve, reject)
    let env = unsafe { Env::from_raw(raw_env) };

    let resolve_id = call_id;
    let resolve_fn = match env.create_function_from_closure("__nat_nexus_resolve", move |ctx| {
        let val: Json = ctx.get(0).unwrap_or(Json::Null);
        send_result(resolve_id, val);
        ctx.env.get_undefined()
    }) {
        Ok(f) => f,
        Err(e) => {
            send_error(call_id, format!("failed to create resolve callback: {e}"));
            return;
        }
    };

    let reject_id = call_id;
    let reject_fn = match env.create_function_from_closure("__nat_nexus_reject", move |ctx| {
        // Try to extract error message from the rejection reason.
        // Could be an Error object, string, or any value.
        // Error.message is non-enumerable so JSON serialization loses it —
        // use get_named_property which accesses all properties.
        let msg = if let Ok(s) = ctx.get::<String>(0) {
            s
        } else if let Ok(obj) = ctx.get::<napi::JsObject>(0) {
            obj.get_named_property::<String>("message")
                .unwrap_or_else(|_| "unknown error".to_string())
        } else {
            "unknown error".to_string()
        };
        send_error(reject_id, msg);
        ctx.env.get_undefined()
    }) {
        Ok(f) => f,
        Err(e) => {
            send_error(call_id, format!("failed to create reject callback: {e}"));
            return;
        }
    };

    // Call promise.then(resolve, reject)
    unsafe {
        let then_key = "then\0";
        let mut then_prop: sys::napi_value = std::ptr::null_mut();
        sys::napi_get_named_property(
            raw_env,
            result,
            then_key.as_ptr() as *const i8,
            &mut then_prop,
        );

        let args = [resolve_fn.raw(), reject_fn.raw()];
        let mut then_result: sys::napi_value = std::ptr::null_mut();
        sys::napi_call_function(
            raw_env,
            result,
            then_prop,
            2,
            args.as_ptr(),
            &mut then_result,
        );
    }
}

/// Try to extract a string message from a JS error value.
fn extract_error_message(
    raw_env: napi::sys::napi_env,
    error_val: napi::sys::napi_value,
) -> Option<String> {
    use napi::sys;

    if error_val.is_null() {
        return None;
    }

    // Check type — if it's a string, use it directly
    let mut value_type: sys::napi_valuetype = sys::ValueType::napi_undefined;
    unsafe { sys::napi_typeof(raw_env, error_val, &mut value_type) };

    if value_type == sys::ValueType::napi_string {
        return unsafe { <String as FromNapiValue>::from_napi_value(raw_env, error_val) }.ok();
    }

    // If it's an object, try .message
    if value_type == sys::ValueType::napi_object {
        let msg_key = "message\0";
        let mut msg_val: sys::napi_value = std::ptr::null_mut();
        unsafe {
            sys::napi_get_named_property(
                raw_env,
                error_val,
                msg_key.as_ptr() as *const i8,
                &mut msg_val,
            );
        }
        if !msg_val.is_null() {
            let mut msg_type: sys::napi_valuetype = sys::ValueType::napi_undefined;
            unsafe { sys::napi_typeof(raw_env, msg_val, &mut msg_type) };
            if msg_type == sys::ValueType::napi_string {
                return unsafe { <String as FromNapiValue>::from_napi_value(raw_env, msg_val) }
                    .ok();
            }
        }
    }

    // Fallback: coerce to string
    let mut str_val: sys::napi_value = std::ptr::null_mut();
    let status = unsafe { sys::napi_coerce_to_string(raw_env, error_val, &mut str_val) };
    if status == sys::Status::napi_ok && !str_val.is_null() {
        return unsafe { <String as FromNapiValue>::from_napi_value(raw_env, str_val) }.ok();
    }

    None
}

// ---------------------------------------------------------------------------
// PromiseAwareFn — cross-thread callable with Promise support
// ---------------------------------------------------------------------------

/// A wrapper around a JS function that can be called from any thread and
/// transparently handles both synchronous and Promise return values.
pub struct PromiseAwareFn {
    tsfn: napi::sys::napi_threadsafe_function,
}

// SAFETY: napi_threadsafe_function is explicitly designed for cross-thread use.
unsafe impl Send for PromiseAwareFn {}
unsafe impl Sync for PromiseAwareFn {}

impl PromiseAwareFn {
    /// Create a new `PromiseAwareFn` wrapping the given JS function.
    ///
    /// Must be called on the JS main thread (i.e., in a sync `#[napi]` function).
    pub fn new(env: &Env, func: &JsFunction) -> napi::Result<Self> {
        use napi::sys;

        let raw_env = env.raw();
        let mut tsfn: sys::napi_threadsafe_function = std::ptr::null_mut();

        // Create resource name
        let name = "nat_nexus_promise_aware_fn\0";
        let mut resource_name: sys::napi_value = std::ptr::null_mut();
        let status = unsafe {
            sys::napi_create_string_utf8(
                raw_env,
                name.as_ptr() as *const i8,
                name.len() - 1, // exclude null terminator
                &mut resource_name,
            )
        };
        if status != sys::Status::napi_ok {
            return Err(napi::Error::from_reason(
                "failed to create resource name string",
            ));
        }

        let status = unsafe {
            sys::napi_create_threadsafe_function(
                raw_env,
                func.raw(),
                std::ptr::null_mut(), // async_resource
                resource_name,
                0,                        // max_queue_size (0 = unlimited)
                1,                        // initial_thread_count
                std::ptr::null_mut(),     // thread_finalize_data
                None,                     // thread_finalize_cb
                std::ptr::null_mut(),     // context
                Some(promise_call_js_cb), // call_js_cb
                &mut tsfn,
            )
        };

        if status != sys::Status::napi_ok {
            return Err(napi::Error::from_reason(
                "failed to create threadsafe function",
            ));
        }

        Ok(PromiseAwareFn { tsfn })
    }

    /// Call the JS function with the given args and await the result.
    ///
    /// Transparently handles both synchronous returns and Promise returns.
    /// Can be called from any thread (e.g., tokio runtime).
    pub async fn call(&self, args: Json) -> NexusResult<Json> {
        use napi::sys;

        let call_id = NEXT_CALL_ID.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = tokio::sync::oneshot::channel();
        register_call(call_id, tx);

        let args_json = serde_json::to_vec(&args).map_err(|e| {
            // Clean up on serialization failure
            PENDING_CALLS.lock().unwrap().remove(&call_id);
            NexusError::Internal(format!("failed to serialize args: {e}"))
        })?;

        let call_data = Box::new(CallData { call_id, args_json });

        let status = unsafe {
            sys::napi_call_threadsafe_function(
                self.tsfn,
                Box::into_raw(call_data) as *mut c_void,
                sys::ThreadsafeFunctionCallMode::nonblocking,
            )
        };

        if status != sys::Status::napi_ok {
            PENDING_CALLS.lock().unwrap().remove(&call_id);
            return Err(NexusError::Internal(
                "failed to queue threadsafe function call".to_string(),
            ));
        }

        rx.await.map_err(|e| NexusError::Internal(e.to_string()))?
    }
}

impl Drop for PromiseAwareFn {
    fn drop(&mut self) {
        unsafe {
            napi::sys::napi_release_threadsafe_function(
                self.tsfn,
                napi::sys::ThreadsafeFunctionReleaseMode::release,
            );
        }
    }
}
