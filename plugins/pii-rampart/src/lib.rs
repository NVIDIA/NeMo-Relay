// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in native Rampart PII redaction plugin.

use std::ffi::c_void;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::ptr;
use std::sync::Arc;

use futures::FutureExt;
use nemo_relay::api::event::{Event, EventSanitizeFields};
use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::{
    BuiltinLlmCodec, EventSanitizeFn, LlmCodecIdentity, LlmSanitizeRequestContext,
    LlmSanitizeRequestFn, LlmSanitizeResponseContext, LlmSanitizeResponseFn, ToolSanitizeFn,
};
use nemo_relay_pii_redaction::rampart::{
    RAMPART_PII_PLUGIN_KIND, RampartMiddlewareCallbacks, load_middleware, validate_config,
};
use nemo_relay_plugin::{
    Json, NativePlugin, NemoRelayNativeAsyncCallbackState, NemoRelayNativeAsyncCompletion,
    NemoRelayNativeAsyncMiddlewareKind, NemoRelayNativeAsyncNext, NemoRelayNativeHostApiV1,
    NemoRelayNativeHostApiV3, NemoRelayNativeString, NemoRelayStatus, PluginContext,
};
use serde::Deserialize;
use serde_json::Map;
use tokio::runtime::{Builder, Runtime};

type CallbackFuture = Pin<Box<dyn Future<Output = Result<Json, String>> + Send>>;
type Callback = Arc<dyn Fn(Json) -> CallbackFuture + Send + Sync>;

struct RampartNativePlugin;

struct AsyncCallbackState {
    host: NemoRelayNativeHostApiV3,
    runtime: Arc<AsyncRuntime>,
    callback: Callback,
}

struct AsyncRuntime {
    runtime: Option<Runtime>,
}

impl AsyncRuntime {
    fn new() -> Result<Self, String> {
        Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("nemo-relay-rampart-async")
            .enable_time()
            .build()
            .map(|runtime| Self {
                runtime: Some(runtime),
            })
            .map_err(|error| format!("failed to start Rampart async runtime: {error}"))
    }

    fn handle(&self) -> tokio::runtime::Handle {
        self.runtime
            .as_ref()
            .expect("Rampart runtime remains available until final drop")
            .handle()
            .clone()
    }
}

impl Drop for AsyncRuntime {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

#[derive(Deserialize)]
struct ToolInvocation {
    name: String,
    value: Json,
}

#[derive(Deserialize)]
struct LlmRequestInvocation {
    request: LlmRequest,
    context: CodecContext,
}

#[derive(Deserialize)]
struct LlmResponseInvocation {
    response: Json,
    context: CodecContext,
}

#[derive(Deserialize)]
struct EventInvocation {
    event: Event,
    fields: EventSanitizeFields,
}

#[derive(Deserialize)]
struct CodecContext {
    codec_kind: String,
    codec_id: Option<String>,
}

impl CodecContext {
    fn into_identity(self) -> LlmCodecIdentity {
        match (self.codec_kind.as_str(), self.codec_id.as_deref()) {
            ("none", _) => LlmCodecIdentity::None,
            ("builtin", Some("openai_chat")) => {
                LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
            }
            ("builtin", Some("openai_responses")) => {
                LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiResponses)
            }
            ("builtin", Some("anthropic_messages")) => {
                LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::AnthropicMessages)
            }
            ("runtime", Some(id)) => LlmCodecIdentity::Runtime(id.to_owned()),
            _ => LlmCodecIdentity::Opaque,
        }
    }
}

impl NativePlugin for RampartNativePlugin {
    fn plugin_kind(&self) -> &str {
        RAMPART_PII_PLUGIN_KIND
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(
        &self,
        plugin_config: &Map<String, Json>,
    ) -> Vec<nemo_relay_plugin::ConfigDiagnostic> {
        validate_config(plugin_config)
    }

    fn register(
        &mut self,
        plugin_config: &Map<String, Json>,
        ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        let middleware = load_middleware(plugin_config).map_err(|error| error.to_string())?;
        let runtime = Arc::new(AsyncRuntime::new()?);
        register_middleware(ctx, runtime, middleware)
    }
}

fn register_middleware(
    ctx: &mut PluginContext<'_>,
    runtime: Arc<AsyncRuntime>,
    middleware: RampartMiddlewareCallbacks,
) -> nemo_relay_plugin::Result<()> {
    let priority = middleware.priority();
    if let Some(callback) = middleware.mark() {
        register_async(
            ctx,
            &runtime,
            NemoRelayNativeAsyncMiddlewareKind::MarkSanitize,
            "mark",
            priority,
            event_callback(callback),
        )?;
    }
    if let Some(callback) = middleware.tool_input() {
        register_async(
            ctx,
            &runtime,
            NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeRequest,
            "tool_input",
            priority,
            tool_callback(callback),
        )?;
    }
    if let Some(callback) = middleware.tool_output() {
        register_async(
            ctx,
            &runtime,
            NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeResponse,
            "tool_output",
            priority,
            tool_callback(callback),
        )?;
    }
    if let Some(callback) = middleware.input() {
        register_async(
            ctx,
            &runtime,
            NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeRequest,
            "input",
            priority,
            llm_request_callback(callback),
        )?;
    }
    if let Some(callback) = middleware.scope_start() {
        register_async(
            ctx,
            &runtime,
            NemoRelayNativeAsyncMiddlewareKind::ScopeSanitizeStart,
            "scope_start",
            priority,
            event_callback(callback),
        )?;
    }
    if let Some(callback) = middleware.output() {
        register_async(
            ctx,
            &runtime,
            NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeResponse,
            "output",
            priority,
            llm_response_callback(callback),
        )?;
    }
    if let Some(callback) = middleware.scope_end() {
        register_async(
            ctx,
            &runtime,
            NemoRelayNativeAsyncMiddlewareKind::ScopeSanitizeEnd,
            "scope_end",
            priority,
            event_callback(callback),
        )?;
    }
    Ok(())
}

fn register_async(
    ctx: &mut PluginContext<'_>,
    runtime: &Arc<AsyncRuntime>,
    kind: NemoRelayNativeAsyncMiddlewareKind,
    name: &str,
    priority: i32,
    callback: Callback,
) -> nemo_relay_plugin::Result<()> {
    let host = ctx.host_api();
    if host.abi_version < nemo_relay_plugin::NEMO_RELAY_NATIVE_ABI_VERSION_ASYNC_MIDDLEWARE
        || host.struct_size < std::mem::size_of::<NemoRelayNativeHostApiV3>()
    {
        return Err("Rampart requires Relay native ABI v3 async middleware".into());
    }
    let state = Box::new(AsyncCallbackState {
        host: unsafe { *(host as *const _ as *const NemoRelayNativeHostApiV3) },
        runtime: Arc::clone(runtime),
        callback,
    });
    let state = Box::into_raw(state).cast();
    let status = unsafe {
        ctx.register_async_middleware_raw(
            kind,
            name,
            priority,
            false,
            async_sanitizer_callback,
            state,
            Some(drop_async_callback_state),
        )
    };
    if status == NemoRelayStatus::Ok {
        Ok(())
    } else {
        unsafe { drop(Box::from_raw(state as *mut AsyncCallbackState)) };
        Err(format!(
            "failed to register Rampart {name} middleware: {status:?}"
        ))
    }
}

fn tool_callback(callback: ToolSanitizeFn) -> Callback {
    Arc::new(move |invocation| {
        let callback = Arc::clone(&callback);
        Box::pin(async move {
            let invocation: ToolInvocation = decode_invocation(invocation)?;
            callback(invocation.name, invocation.value)
                .await
                .map_err(|error| error.to_string())
        })
    })
}

fn llm_request_callback(callback: LlmSanitizeRequestFn) -> Callback {
    Arc::new(move |invocation| {
        let callback = Arc::clone(&callback);
        Box::pin(async move {
            let invocation: LlmRequestInvocation = decode_invocation(invocation)?;
            let context =
                LlmSanitizeRequestContext::with_identity(invocation.context.into_identity());
            callback(invocation.request, context)
                .await
                .map(|value| {
                    value.map_or(Json::Null, |request| {
                        serde_json::to_value(request).expect("LLM requests serialize")
                    })
                })
                .map_err(|error| error.to_string())
        })
    })
}

fn llm_response_callback(callback: LlmSanitizeResponseFn) -> Callback {
    Arc::new(move |invocation| {
        let callback = Arc::clone(&callback);
        Box::pin(async move {
            let invocation: LlmResponseInvocation = decode_invocation(invocation)?;
            let context =
                LlmSanitizeResponseContext::with_identity(invocation.context.into_identity());
            callback(invocation.response, context)
                .await
                .map(|value| value.unwrap_or(Json::Null))
                .map_err(|error| error.to_string())
        })
    })
}

fn event_callback(callback: EventSanitizeFn) -> Callback {
    Arc::new(move |invocation| {
        let callback = Arc::clone(&callback);
        Box::pin(async move {
            let invocation: EventInvocation = decode_invocation(invocation)?;
            let fields = callback(Arc::new(invocation.event), invocation.fields)
                .await
                .map_err(|error| error.to_string())?;
            serde_json::to_value(fields)
                .map_err(|error| format!("failed to serialize Rampart event fields: {error}"))
        })
    })
}

fn decode_invocation<T: for<'de> Deserialize<'de>>(value: Json) -> Result<T, String> {
    serde_json::from_value(value).map_err(|error| format!("invalid Rampart invocation: {error}"))
}

unsafe extern "C" fn async_sanitizer_callback(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    _next: *const NemoRelayNativeAsyncNext,
    completion: *const NemoRelayNativeAsyncCompletion,
) -> u32 {
    let Some(state) = (unsafe { (user_data as *const AsyncCallbackState).as_ref() }) else {
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    };
    if completion.is_null() {
        return NemoRelayNativeAsyncCallbackState::Complete as u32;
    }
    let invocation = match read_json(&state.host.v1, invocation_json) {
        Ok(invocation) => invocation,
        Err(error) => {
            reject_completion(&state.host, completion, &error);
            return NemoRelayNativeAsyncCallbackState::Complete as u32;
        }
    };

    let host = state.host;
    let callback = Arc::clone(&state.callback);
    let runtime = Arc::clone(&state.runtime);
    let completion = completion as usize;
    let handle = runtime.handle();
    handle.spawn(async move {
        let _runtime = runtime;
        let completion = PendingCompletion {
            host,
            raw: completion,
        };
        if completion.is_cancelled() {
            return;
        }
        let result = AssertUnwindSafe(callback(invocation)).catch_unwind().await;
        if completion.is_cancelled() {
            return;
        }
        match result {
            Ok(Ok(value)) => completion.resolve(&value),
            Ok(Err(error)) => completion.reject(&error),
            Err(_) => completion.reject("Rampart async sanitizer panicked"),
        }
    });
    NemoRelayNativeAsyncCallbackState::Pending as u32
}

struct PendingCompletion {
    host: NemoRelayNativeHostApiV3,
    raw: usize,
}

impl PendingCompletion {
    fn as_ptr(&self) -> *const NemoRelayNativeAsyncCompletion {
        self.raw as *const NemoRelayNativeAsyncCompletion
    }

    fn is_cancelled(&self) -> bool {
        unsafe { (self.host.async_completion_is_cancelled)(self.as_ptr()) }
    }

    fn resolve(&self, value: &Json) {
        match serde_json::to_string(value) {
            Ok(value) => {
                let Some(value) = host_string(&self.host.v1, &value) else {
                    self.reject("failed to allocate Rampart completion result");
                    return;
                };
                unsafe {
                    let _ = (self.host.async_completion_resolve_json)(self.as_ptr(), value);
                    (self.host.v1.string_free)(value);
                }
            }
            Err(error) => self.reject(&format!("failed to serialize Rampart result: {error}")),
        }
    }

    fn reject(&self, message: &str) {
        reject_completion(&self.host, self.as_ptr(), message);
    }
}

impl Drop for PendingCompletion {
    fn drop(&mut self) {
        unsafe { (self.host.async_completion_release)(self.as_ptr()) };
    }
}

unsafe extern "C" fn drop_async_callback_state(user_data: *mut c_void) {
    if !user_data.is_null() {
        unsafe { drop(Box::from_raw(user_data as *mut AsyncCallbackState)) };
    }
}

fn read_json(
    host: &NemoRelayNativeHostApiV1,
    value: *const NemoRelayNativeString,
) -> Result<Json, String> {
    if value.is_null() {
        return Err("Rampart invocation was null".into());
    }
    let len = unsafe { (host.string_len)(value) };
    let data = unsafe { (host.string_data)(value) };
    if data.is_null() && len > 0 {
        return Err("Rampart invocation contained invalid UTF-8".into());
    }
    let bytes = if len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }
    };
    let value = std::str::from_utf8(bytes)
        .map_err(|_| "Rampart invocation contained invalid UTF-8".to_string())?;
    serde_json::from_str(value).map_err(|error| format!("invalid Rampart invocation: {error}"))
}

fn reject_completion(
    host: &NemoRelayNativeHostApiV3,
    completion: *const NemoRelayNativeAsyncCompletion,
    message: &str,
) {
    let Some(message) = host_string(&host.v1, message) else {
        return;
    };
    unsafe {
        let _ = (host.async_completion_reject)(completion, message);
        (host.v1.string_free)(message);
    }
}

fn host_string(host: &NemoRelayNativeHostApiV1, value: &str) -> Option<*mut NemoRelayNativeString> {
    let mut output = ptr::null_mut();
    let status = unsafe { (host.string_new)(value.as_ptr(), value.len(), &mut output) };
    (status == NemoRelayStatus::Ok && !output.is_null()).then_some(output)
}

nemo_relay_plugin::nemo_relay_plugin!(nemo_relay_register_plugin, || RampartNativePlugin);
