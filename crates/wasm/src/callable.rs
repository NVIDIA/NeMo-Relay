#![allow(clippy::type_complexity)]
//! Wrappers that adapt JavaScript callback functions into Rust closures.
//!
//! Each wrapper takes a `js_sys::Function`, wraps it with `SendWrapper` (since
//! JS functions are not `Send`), and returns a boxed closure matching the
//! signature expected by the core runtime for guardrails, intercepts,
//! execution functions, and event subscribers.

use std::future::Future;
use std::pin::Pin;

use js_sys::Function;
use send_wrapper::SendWrapper;
use serde::Serialize;
use serde_json::Value as Json;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use nvagentrt_core::types::{LLMRequest, SseEvent};
use nvagentrt_core::{AgentRtError, Result};

use crate::convert::{js_to_json, json_to_js};
use crate::types::WasmEvent;

/// Wrap a JS function `(name, args) => result` for tool sanitize/intercept.
pub fn wrap_js_tool_fn(func: Function) -> Box<dyn Fn(&str, Json) -> Json + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |name: &str, args: Json| {
        let js_name = JsValue::from_str(name);
        let js_args = json_to_js(&args);
        match func.call2(&JsValue::NULL, &js_name, &js_args) {
            Ok(result) => js_to_json(&result).unwrap_or(Json::Null),
            Err(_) => Json::Null,
        }
    })
}

/// Wrap a JS function `(name, args) => string | null` for tool conditional.
pub fn wrap_js_tool_conditional_fn(
    func: Function,
) -> Box<dyn Fn(&str, &Json) -> Option<String> + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |name: &str, args: &Json| {
        let js_name = JsValue::from_str(name);
        let js_args = json_to_js(args);
        match func.call2(&JsValue::NULL, &js_name, &js_args) {
            Ok(result) => {
                if result.is_null() || result.is_undefined() {
                    None
                } else {
                    result.as_string()
                }
            }
            Err(_) => None,
        }
    })
}

/// Wrap a JS function `(name, args) => boolean` for tool exec conditional.
pub fn wrap_js_tool_exec_conditional_fn(
    func: Function,
) -> Box<dyn Fn(&str, &Json) -> bool + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |name: &str, args: &Json| {
        let js_name = JsValue::from_str(name);
        let js_args = json_to_js(args);
        match func.call2(&JsValue::NULL, &js_name, &js_args) {
            Ok(result) => result.as_bool().unwrap_or(false),
            Err(_) => false,
        }
    })
}

/// Wrap a JS function `(args) => result | Promise<result>` for tool execution.
pub fn wrap_js_tool_exec_fn(
    func: Function,
) -> Box<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |args: Json| {
        let js_args = json_to_js(&args);
        let result = func.call1(&JsValue::NULL, &js_args);
        Box::pin(SendWrapper::new(async move {
            match result {
                Ok(val) => {
                    // Check if it's a Promise
                    if let Some(promise) = val.dyn_ref::<js_sys::Promise>() {
                        match JsFuture::from(promise.clone()).await {
                            Ok(resolved) => js_to_json(&resolved)
                                .map_err(|e| AgentRtError::Internal(format!("{e:?}"))),
                            Err(e) => Err(AgentRtError::Internal(format!("{e:?}"))),
                        }
                    } else {
                        js_to_json(&val).map_err(|e| AgentRtError::Internal(format!("{e:?}")))
                    }
                }
                Err(e) => Err(AgentRtError::Internal(format!("{e:?}"))),
            }
        }))
    })
}

/// Wrap a JS function `(value) => value` for JSON transform.
pub fn wrap_js_json_fn(func: Function) -> Box<dyn Fn(Json) -> Json + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |value: Json| {
        let js_val = json_to_js(&value);
        match func.call1(&JsValue::NULL, &js_val) {
            Ok(result) => js_to_json(&result).unwrap_or(Json::Null),
            Err(_) => Json::Null,
        }
    })
}

/// Wrap a JS function for LLM sanitize request: `(request) => request`.
pub fn wrap_js_llm_sanitize_request_fn(
    func: Function,
) -> Box<dyn Fn(LLMRequest) -> LLMRequest + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |request: LLMRequest| {
        let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
        let js_req = json_to_js(&req_json);
        match func.call1(&JsValue::NULL, &js_req) {
            Ok(result) => {
                let result_json = js_to_json(&result).unwrap_or(Json::Null);
                serde_json::from_value(result_json).unwrap_or(request)
            }
            Err(_) => request,
        }
    })
}

/// Wrap a JS function for LLM conditional: `(request) => string | null`.
pub fn wrap_js_llm_conditional_fn(
    func: Function,
) -> Box<dyn Fn(&LLMRequest) -> Option<String> + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |request: &LLMRequest| {
        let req_json = serde_json::to_value(request).unwrap_or(Json::Null);
        let js_req = json_to_js(&req_json);
        match func.call1(&JsValue::NULL, &js_req) {
            Ok(result) => {
                if result.is_null() || result.is_undefined() {
                    None
                } else {
                    result.as_string()
                }
            }
            Err(_) => None,
        }
    })
}

/// Wrap a JS function for LLM exec conditional: `(request) => boolean`.
pub fn wrap_js_llm_exec_conditional_fn(
    func: Function,
) -> Box<dyn Fn(&LLMRequest) -> bool + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |request: &LLMRequest| {
        let req_json = serde_json::to_value(request).unwrap_or(Json::Null);
        let js_req = json_to_js(&req_json);
        match func.call1(&JsValue::NULL, &js_req) {
            Ok(result) => result.as_bool().unwrap_or(false),
            Err(_) => false,
        }
    })
}

/// Wrap a JS function for LLM execution: `(request) => result | Promise<result>`.
pub fn wrap_js_llm_exec_fn(
    func: Function,
) -> Box<dyn Fn(LLMRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |request: LLMRequest| {
        let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
        let js_req = json_to_js(&req_json);
        let result = func.call1(&JsValue::NULL, &js_req);
        Box::pin(SendWrapper::new(async move {
            match result {
                Ok(val) => {
                    if let Some(promise) = val.dyn_ref::<js_sys::Promise>() {
                        match JsFuture::from(promise.clone()).await {
                            Ok(resolved) => js_to_json(&resolved)
                                .map_err(|e| AgentRtError::Internal(format!("{e:?}"))),
                            Err(e) => Err(AgentRtError::Internal(format!("{e:?}"))),
                        }
                    } else {
                        js_to_json(&val).map_err(|e| AgentRtError::Internal(format!("{e:?}")))
                    }
                }
                Err(e) => Err(AgentRtError::Internal(format!("{e:?}"))),
            }
        }))
    })
}

/// Wrap a JS function for SSE intercept: `(event) => event`.
pub fn wrap_js_sse_intercept_fn(func: Function) -> Box<dyn Fn(SseEvent) -> SseEvent + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |event: SseEvent| {
        let event_json = serde_json::to_value(&event).unwrap_or(Json::Null);
        let js_event = json_to_js(&event_json);
        match func.call1(&JsValue::NULL, &js_event) {
            Ok(result) => {
                let result_json = js_to_json(&result).unwrap_or(Json::Null);
                serde_json::from_value(result_json).unwrap_or(event)
            }
            Err(_) => event,
        }
    })
}

/// Wrap a JS function for event subscriber: `(event) => void`.
pub fn wrap_js_event_subscriber(
    func: Function,
) -> Box<dyn Fn(&nvagentrt_core::Event) + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |event: &nvagentrt_core::Event| {
        let wasm_event = WasmEvent::from(event);
        let js_event = wasm_event
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .unwrap_or(JsValue::NULL);
        let _ = func.call1(&JsValue::NULL, &js_event);
    })
}
