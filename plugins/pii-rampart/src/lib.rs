// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in native Rampart PII redaction plugin.

use std::sync::Arc;

use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::{
    BuiltinLlmCodec, LlmCodecIdentity, LlmSanitizeRequestContext, LlmSanitizeResponseContext,
};
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::response::AnnotatedLlmResponse;
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay::error::{FlowError, Result as FlowResult};
use nemo_relay_pii_redaction::rampart::{
    RAMPART_PII_PLUGIN_KIND, RampartMiddlewareCallbacks, load_middleware, validate_config,
};
use nemo_relay_plugin::{
    BuiltinLlmCodec as NativeBuiltinLlmCodec, ConfigDiagnostic, DiagnosticLevel, Json,
    LlmCodecIdentity as NativeLlmCodecIdentity,
    LlmSanitizeRequestContext as NativeLlmSanitizeRequestContext,
    LlmSanitizeResponseContext as NativeLlmSanitizeResponseContext, NativeExecutorConfig,
    NativePlugin, PluginContext,
};
use serde_json::Map;

struct RampartNativePlugin;

impl NativePlugin for RampartNativePlugin {
    fn plugin_kind(&self) -> &str {
        RAMPART_PII_PLUGIN_KIND
    }

    fn executor_config(&self) -> NativeExecutorConfig {
        NativeExecutorConfig { worker_threads: 1 }
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        let mut diagnostics = validate_config(&rampart_config(plugin_config));
        if let Some(Json::Object(executor)) = plugin_config.get("executor") {
            for field in executor.keys().filter(|field| *field != "worker_threads") {
                diagnostics.push(ConfigDiagnostic {
                    level: DiagnosticLevel::Error,
                    code: "pii_rampart.unknown_executor_field".into(),
                    component: Some(RAMPART_PII_PLUGIN_KIND.into()),
                    field: Some(format!("executor.{field}")),
                    message: format!("unknown executor field '{field}'"),
                });
            }
        }
        if let Err(message) = self.executor_config_for_component(plugin_config) {
            diagnostics.push(ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "pii_rampart.invalid_executor".into(),
                component: Some(RAMPART_PII_PLUGIN_KIND.into()),
                field: Some("executor.worker_threads".into()),
                message,
            });
        }
        diagnostics
    }

    fn register(
        &mut self,
        plugin_config: &Map<String, Json>,
        ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        let middleware =
            load_middleware(&rampart_config(plugin_config)).map_err(|error| error.to_string())?;
        register_middleware(ctx, middleware)
    }
}

fn rampart_config(plugin_config: &Map<String, Json>) -> Map<String, Json> {
    let mut config = plugin_config.clone();
    config.remove("executor");
    config
}

fn register_middleware(
    ctx: &mut PluginContext<'_>,
    middleware: RampartMiddlewareCallbacks,
) -> nemo_relay_plugin::Result<()> {
    let priority = middleware.priority();
    if let Some(callback) = middleware.mark() {
        ctx.register_mark_sanitize_guardrail("mark", priority, move |event, fields| {
            let callback = Arc::clone(&callback);
            async move {
                callback(event, fields)
                    .await
                    .map_err(|error| error.to_string())
            }
        })?;
    }
    if let Some(callback) = middleware.tool_input() {
        ctx.register_tool_sanitize_request_guardrail(
            "tool_input",
            priority,
            move |name, value| {
                let callback = Arc::clone(&callback);
                async move {
                    callback(name, value)
                        .await
                        .map_err(|error| error.to_string())
                }
            },
        )?;
    }
    if let Some(callback) = middleware.tool_output() {
        ctx.register_tool_sanitize_response_guardrail(
            "tool_output",
            priority,
            move |name, value| {
                let callback = Arc::clone(&callback);
                async move {
                    callback(name, value)
                        .await
                        .map_err(|error| error.to_string())
                }
            },
        )?;
    }
    if let Some(callback) = middleware.input() {
        ctx.register_llm_sanitize_request_guardrail("input", priority, move |request, context| {
            let callback = Arc::clone(&callback);
            async move {
                callback(request, core_request_context(context))
                    .await
                    .map_err(|error| error.to_string())
            }
        })?;
    }
    if let Some(callback) = middleware.scope_start() {
        ctx.register_scope_sanitize_start_guardrail(
            "scope_start",
            priority,
            move |event, fields| {
                let callback = Arc::clone(&callback);
                async move {
                    callback(event, fields)
                        .await
                        .map_err(|error| error.to_string())
                }
            },
        )?;
    }
    if let Some(callback) = middleware.output() {
        ctx.register_llm_sanitize_response_guardrail(
            "output",
            priority,
            move |response, context| {
                let callback = Arc::clone(&callback);
                async move {
                    callback(response, core_response_context(context))
                        .await
                        .map_err(|error| error.to_string())
                }
            },
        )?;
    }
    if let Some(callback) = middleware.scope_end() {
        ctx.register_scope_sanitize_end_guardrail("scope_end", priority, move |event, fields| {
            let callback = Arc::clone(&callback);
            async move {
                callback(event, fields)
                    .await
                    .map_err(|error| error.to_string())
            }
        })?;
    }
    Ok(())
}

struct NativeRequestCodecBridge {
    identity: LlmCodecIdentity,
    context: NativeLlmSanitizeRequestContext<'static>,
}

impl LlmCodec for NativeRequestCodecBridge {
    fn codec_identity(&self) -> LlmCodecIdentity {
        self.identity.clone()
    }

    fn decode(&self, request: &LlmRequest) -> FlowResult<AnnotatedLlmRequest> {
        self.context
            .resolve_codec()
            .ok_or_else(missing_request_codec)?
            .decode(request)
            .map_err(FlowError::Internal)
    }

    fn encode(
        &self,
        annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> FlowResult<LlmRequest> {
        self.context
            .resolve_codec()
            .ok_or_else(missing_request_codec)?
            .encode(annotated, original)
            .map_err(FlowError::Internal)
    }
}

struct NativeResponseCodecBridge {
    identity: LlmCodecIdentity,
    context: NativeLlmSanitizeResponseContext<'static>,
}

impl LlmResponseCodec for NativeResponseCodecBridge {
    fn codec_identity(&self) -> LlmCodecIdentity {
        self.identity.clone()
    }

    fn decode_response(&self, response: &Json) -> FlowResult<AnnotatedLlmResponse> {
        self.context
            .resolve_codec()
            .ok_or_else(missing_response_codec)?
            .decode(response)
            .map_err(FlowError::Internal)
    }
}

fn missing_request_codec() -> FlowError {
    FlowError::Internal("native request codec capability was unavailable".into())
}

fn missing_response_codec() -> FlowError {
    FlowError::Internal("native response codec capability was unavailable".into())
}

fn core_request_context(
    context: NativeLlmSanitizeRequestContext<'static>,
) -> LlmSanitizeRequestContext {
    let identity = core_codec_identity(&context.codec);
    if context.resolve_codec().is_some() {
        LlmSanitizeRequestContext::for_request_codec(Some(Arc::new(NativeRequestCodecBridge {
            identity,
            context,
        })))
    } else {
        LlmSanitizeRequestContext::with_identity(identity)
    }
}

fn core_response_context(
    context: NativeLlmSanitizeResponseContext<'static>,
) -> LlmSanitizeResponseContext {
    let identity = core_codec_identity(&context.codec);
    if context.resolve_codec().is_some() {
        LlmSanitizeResponseContext::for_response_codec(Some(Arc::new(NativeResponseCodecBridge {
            identity,
            context,
        })))
    } else {
        LlmSanitizeResponseContext::with_identity(identity)
    }
}

fn core_codec_identity(identity: &NativeLlmCodecIdentity) -> LlmCodecIdentity {
    match identity {
        NativeLlmCodecIdentity::None => LlmCodecIdentity::None,
        NativeLlmCodecIdentity::BuiltIn(codec) => LlmCodecIdentity::BuiltIn(match codec {
            NativeBuiltinLlmCodec::OpenAiChat => BuiltinLlmCodec::OpenAiChat,
            NativeBuiltinLlmCodec::OpenAiResponses => BuiltinLlmCodec::OpenAiResponses,
            NativeBuiltinLlmCodec::AnthropicMessages => BuiltinLlmCodec::AnthropicMessages,
            NativeBuiltinLlmCodec::OCIGenAI => BuiltinLlmCodec::OCIGenAI,
            NativeBuiltinLlmCodec::GeminiGenerateContent => BuiltinLlmCodec::GeminiGenerateContent,
        }),
        NativeLlmCodecIdentity::Runtime(id) => LlmCodecIdentity::Runtime(id.clone()),
        NativeLlmCodecIdentity::Opaque => LlmCodecIdentity::Opaque,
    }
}

#[cfg(test)]
#[path = "../tests/unit/lib_tests.rs"]
mod tests;

nemo_relay_plugin::nemo_relay_plugin!(nemo_relay_register_plugin, || RampartNativePlugin);
