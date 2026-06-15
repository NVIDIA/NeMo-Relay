// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use chrono::{DateTime, Utc};
use libc::c_char;
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::response::Usage;
use nemo_relay_adaptive::acg::{
    AgentIdentity, CacheRequestFacts, CacheTelemetryEvent, CacheTelemetryProvider,
};
use nemo_relay_adaptive::context_helpers::set_latency_sensitivity;
use nemo_relay_adaptive::{AdaptiveConfig, AdaptiveRuntime};
use serde::Deserialize;
use serde_json::Value as Json;
use uuid::Uuid;

use crate::convert::{c_str_to_json, json_to_c_string};
use crate::error::{NemoRelayStatus, clear_last_error, set_last_error};
use crate::types::{FfiAdaptiveRuntime, FfiScopeHandle};

use super::tokio_runtime;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CacheTelemetryOptions {
    provider: String,
    #[serde(alias = "request_id")]
    request_id: String,
    usage: Option<Json>,
    #[serde(default, alias = "request_facts")]
    request_facts: Option<Json>,
    #[serde(alias = "agent_id")]
    agent_id: String,
    #[serde(alias = "template_version")]
    template_version: String,
    #[serde(alias = "toolset_hash")]
    toolset_hash: String,
    #[serde(alias = "model_family")]
    model_family: String,
    #[serde(alias = "tenant_scope")]
    tenant_scope: String,
    timestamp: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CacheRequestFactsOptions {
    provider: String,
    #[serde(alias = "request_id")]
    request_id: String,
    #[serde(alias = "annotated_request")]
    annotated_request: Json,
    #[serde(alias = "agent_id")]
    agent_id: String,
    timestamp: Option<String>,
}

fn parse_adaptive_config(config_json: *const c_char) -> Result<AdaptiveConfig, NemoRelayStatus> {
    let config_json = c_str_to_json(config_json).ok_or(NemoRelayStatus::InvalidJson)?;
    serde_json::from_value(config_json).map_err(|error| {
        set_last_error(&format!("invalid adaptive config: {error}"));
        NemoRelayStatus::InvalidJson
    })
}

fn parse_provider(provider: &str) -> Result<CacheTelemetryProvider, NemoRelayStatus> {
    match provider {
        "anthropic" => Ok(CacheTelemetryProvider::Anthropic),
        "openai" => Ok(CacheTelemetryProvider::OpenAI),
        other => {
            set_last_error(&format!("unsupported provider: {other}"));
            Err(NemoRelayStatus::InvalidArg)
        }
    }
}

fn parse_request_id(request_id: &str) -> Result<Uuid, NemoRelayStatus> {
    Uuid::parse_str(request_id).map_err(|error| {
        set_last_error(&format!("invalid request_id UUID: {error}"));
        NemoRelayStatus::InvalidArg
    })
}

fn parse_timestamp(timestamp: Option<&str>) -> Result<DateTime<Utc>, NemoRelayStatus> {
    match timestamp {
        Some(value) => DateTime::parse_from_rfc3339(value)
            .map(|value| value.with_timezone(&Utc))
            .map_err(|error| {
                set_last_error(&format!("invalid timestamp: {error}"));
                NemoRelayStatus::InvalidArg
            }),
        None => Ok(Utc::now()),
    }
}

fn lock_runtime(
    runtime: *mut FfiAdaptiveRuntime,
) -> Result<std::sync::MutexGuard<'static, Option<AdaptiveRuntime>>, NemoRelayStatus> {
    if runtime.is_null() {
        set_last_error("adaptive runtime pointer is null");
        return Err(NemoRelayStatus::NullPointer);
    }
    unsafe { &*runtime }.0.lock().map_err(|error| {
        set_last_error(&format!("adaptive runtime lock poisoned: {error}"));
        NemoRelayStatus::Internal
    })
}

fn with_runtime_mut<R>(
    runtime: *mut FfiAdaptiveRuntime,
    f: impl FnOnce(&mut AdaptiveRuntime) -> Result<R, NemoRelayStatus>,
) -> Result<R, NemoRelayStatus> {
    let mut guard = lock_runtime(runtime)?;
    let Some(runtime) = guard.as_mut() else {
        set_last_error("adaptive runtime already shut down");
        return Err(NemoRelayStatus::InvalidArg);
    };
    f(runtime)
}

/// Validate an adaptive config document and return the diagnostics report as JSON.
///
/// # Safety
/// `config_json` must be a valid C string and `out_json` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_adaptive_validate_config(
    config_json: *const c_char,
    out_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out_json.is_null() {
        set_last_error("out_json pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let config = match parse_adaptive_config(config_json) {
        Ok(config) => config,
        Err(status) => return status,
    };
    let report = AdaptiveRuntime::validate_config(&config);
    match serde_json::to_value(&report) {
        Ok(value) => {
            unsafe { *out_json = json_to_c_string(&value) };
            NemoRelayStatus::Ok
        }
        Err(error) => {
            set_last_error(&error.to_string());
            NemoRelayStatus::Internal
        }
    }
}

/// Create an owned adaptive runtime handle from config JSON.
///
/// # Safety
/// `config_json` must be a valid C string and `out` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_adaptive_runtime_create(
    config_json: *const c_char,
    out: *mut *mut FfiAdaptiveRuntime,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let config = match parse_adaptive_config(config_json) {
        Ok(config) => config,
        Err(status) => return status,
    };
    match tokio_runtime().block_on(AdaptiveRuntime::new(config)) {
        Ok(runtime) => {
            unsafe {
                *out = Box::into_raw(Box::new(FfiAdaptiveRuntime(std::sync::Mutex::new(Some(
                    runtime,
                )))))
            };
            NemoRelayStatus::Ok
        }
        Err(error) => {
            set_last_error(&error.to_string());
            NemoRelayStatus::InvalidArg
        }
    }
}

/// Register configured adaptive runtime features.
///
/// # Safety
/// `runtime` must be a valid adaptive runtime pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_adaptive_runtime_register(
    runtime: *mut FfiAdaptiveRuntime,
) -> NemoRelayStatus {
    clear_last_error();
    match with_runtime_mut(runtime, |runtime| {
        tokio_runtime()
            .block_on(runtime.register())
            .map_err(|error| {
                set_last_error(&error.to_string());
                NemoRelayStatus::Internal
            })
    }) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(status) => status,
    }
}

/// Deregister configured adaptive runtime features.
///
/// # Safety
/// `runtime` must be a valid adaptive runtime pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_adaptive_runtime_deregister(
    runtime: *mut FfiAdaptiveRuntime,
) -> NemoRelayStatus {
    clear_last_error();
    match with_runtime_mut(runtime, |runtime| {
        runtime.deregister().map_err(|error| {
            set_last_error(&error.to_string());
            NemoRelayStatus::Internal
        })
    }) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(status) => status,
    }
}

/// Shut down the adaptive runtime and consume its Rust runtime state.
///
/// # Safety
/// `runtime` must be a valid adaptive runtime pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_adaptive_runtime_shutdown(
    runtime: *mut FfiAdaptiveRuntime,
) -> NemoRelayStatus {
    clear_last_error();
    let mut guard = match lock_runtime(runtime) {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let Some(runtime) = guard.take() else {
        set_last_error("adaptive runtime already shut down");
        return NemoRelayStatus::InvalidArg;
    };
    match tokio_runtime().block_on(runtime.shutdown()) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => {
            set_last_error(&error.to_string());
            NemoRelayStatus::Internal
        }
    }
}

/// Wait until the adaptive telemetry drain has processed pending events.
///
/// # Safety
/// `runtime` must be a valid adaptive runtime pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_adaptive_runtime_wait_for_idle(
    runtime: *mut FfiAdaptiveRuntime,
) -> NemoRelayStatus {
    clear_last_error();
    match with_runtime_mut(runtime, |runtime| {
        runtime.wait_for_idle();
        Ok(())
    }) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(status) => status,
    }
}

/// Return the runtime construction report as JSON.
///
/// # Safety
/// `runtime` and `out_json` must be valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_adaptive_runtime_report_json(
    runtime: *mut FfiAdaptiveRuntime,
    out_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out_json.is_null() {
        set_last_error("out_json pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    match with_runtime_mut(runtime, |runtime| {
        serde_json::to_value(runtime.report()).map_err(|error| {
            set_last_error(&error.to_string());
            NemoRelayStatus::Internal
        })
    }) {
        Ok(value) => {
            unsafe { *out_json = json_to_c_string(&value) };
            NemoRelayStatus::Ok
        }
        Err(status) => status,
    }
}

/// Bind ACG request rewrites to a scope.
///
/// # Safety
/// `runtime` and `scope` must be valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_adaptive_runtime_bind_scope(
    runtime: *mut FfiAdaptiveRuntime,
    scope: *const FfiScopeHandle,
) -> NemoRelayStatus {
    clear_last_error();
    if scope.is_null() {
        set_last_error("scope handle pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let scope_uuid = unsafe { (*scope).0.uuid };
    match with_runtime_mut(runtime, |runtime| {
        runtime.bind_scope(scope_uuid).map_err(|error| {
            set_last_error(&error.to_string());
            NemoRelayStatus::Internal
        })
    }) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(status) => status,
    }
}

/// Build cache request facts as JSON.
///
/// # Safety
/// `runtime`, `options_json`, and `out_json` must be valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_adaptive_runtime_build_cache_request_facts(
    runtime: *mut FfiAdaptiveRuntime,
    options_json: *const c_char,
    out_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out_json.is_null() {
        set_last_error("out_json pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let options_json = match c_str_to_json(options_json) {
        Some(value) => value,
        None => return NemoRelayStatus::InvalidJson,
    };
    let options: CacheRequestFactsOptions = match serde_json::from_value(options_json) {
        Ok(options) => options,
        Err(error) => {
            set_last_error(&format!("invalid cache request facts options: {error}"));
            return NemoRelayStatus::InvalidJson;
        }
    };
    if let Err(status) = parse_request_id(&options.request_id) {
        return status;
    }
    if let Err(status) = parse_timestamp(options.timestamp.as_deref()) {
        return status;
    }
    let annotated_request: AnnotatedLlmRequest =
        match serde_json::from_value(options.annotated_request) {
            Ok(request) => request,
            Err(error) => {
                set_last_error(&format!("invalid annotated_request: {error}"));
                return NemoRelayStatus::InvalidJson;
            }
        };
    match with_runtime_mut(runtime, |runtime| {
        Ok(runtime.build_cache_request_facts(
            &options.agent_id,
            &options.provider,
            &annotated_request,
        ))
    }) {
        Ok(facts) => match serde_json::to_value(facts) {
            Ok(value) => {
                unsafe { *out_json = json_to_c_string(&value) };
                NemoRelayStatus::Ok
            }
            Err(error) => {
                set_last_error(&error.to_string());
                NemoRelayStatus::Internal
            }
        },
        Err(status) => status,
    }
}

/// Build one cache telemetry event as JSON.
///
/// # Safety
/// `options_json` and `out_json` must be valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_adaptive_build_cache_telemetry_event(
    options_json: *const c_char,
    out_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out_json.is_null() {
        set_last_error("out_json pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let options_json = match c_str_to_json(options_json) {
        Some(value) => value,
        None => return NemoRelayStatus::InvalidJson,
    };
    let options: CacheTelemetryOptions = match serde_json::from_value(options_json) {
        Ok(options) => options,
        Err(error) => {
            set_last_error(&format!("invalid cache telemetry options: {error}"));
            return NemoRelayStatus::InvalidJson;
        }
    };
    let provider = match parse_provider(&options.provider) {
        Ok(provider) => provider,
        Err(status) => return status,
    };
    let request_id = match parse_request_id(&options.request_id) {
        Ok(request_id) => request_id,
        Err(status) => return status,
    };
    let timestamp = match parse_timestamp(options.timestamp.as_deref()) {
        Ok(timestamp) => timestamp,
        Err(status) => return status,
    };
    let Some(usage_json) = options.usage else {
        unsafe { *out_json = json_to_c_string(&Json::Null) };
        return NemoRelayStatus::Ok;
    };
    let usage: Usage = match serde_json::from_value(usage_json) {
        Ok(usage) => usage,
        Err(error) => {
            set_last_error(&format!("invalid usage: {error}"));
            return NemoRelayStatus::InvalidJson;
        }
    };
    let request_facts: Option<CacheRequestFacts> = match options.request_facts {
        Some(value) => match serde_json::from_value(value) {
            Ok(facts) => Some(facts),
            Err(error) => {
                set_last_error(&format!("invalid request_facts: {error}"));
                return NemoRelayStatus::InvalidJson;
            }
        },
        None => None,
    };
    let agent_identity = AgentIdentity {
        agent_id: options.agent_id,
        template_version: options.template_version,
        toolset_hash: options.toolset_hash,
        model_family: options.model_family,
        tenant_scope: options.tenant_scope,
    };
    let event = CacheTelemetryEvent::from_usage(
        request_id,
        agent_identity,
        provider,
        &usage,
        timestamp,
        request_facts.as_ref(),
    );
    match serde_json::to_value(event) {
        Ok(value) => {
            unsafe { *out_json = json_to_c_string(&value) };
            NemoRelayStatus::Ok
        }
        Err(error) => {
            set_last_error(&error.to_string());
            NemoRelayStatus::Internal
        }
    }
}

/// Set manual latency sensitivity on the current scope.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_relay_adaptive_set_latency_sensitivity(value: u32) -> NemoRelayStatus {
    clear_last_error();
    if value == 0 {
        set_last_error("sensitivity must be positive (> 0)");
        return NemoRelayStatus::InvalidArg;
    }
    match set_latency_sensitivity(value) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => {
            set_last_error(&error);
            NemoRelayStatus::Internal
        }
    }
}
