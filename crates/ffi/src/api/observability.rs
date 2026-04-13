// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ---------------------------------------------------------------------------
// ATIF exporter
// ---------------------------------------------------------------------------

/// Creates a new ATIF exporter.
///
/// # Parameters
/// - `session_id`: Session identifier string (required, non-null).
/// - `agent_name`: Agent name string (required, non-null).
/// - `agent_version`: Agent version string (required, non-null).
/// - `model_name`: Default model name (nullable).
/// - `out`: On success, receives a heap-allocated `FfiAtifExporter`.
///
/// # Safety
/// All non-null string pointers must be valid C strings. `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_atif_exporter_create(
    session_id: *const c_char,
    agent_name: *const c_char,
    agent_version: *const c_char,
    model_name: *const c_char,
    out: *mut *mut FfiAtifExporter,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let session_id = match c_str_to_string(session_id) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let agent_name = match c_str_to_string(agent_name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let agent_version = match c_str_to_string(agent_version) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };

    let agent_info = nemo_flow::atif::AtifAgentInfo {
        name: agent_name,
        version: agent_version,
        model_name: model_name_opt,
        tool_definitions: None,
        extra: None,
    };

    let exporter = nemo_flow::atif::AtifExporter::new(session_id, agent_info);
    unsafe { *out = Box::into_raw(Box::new(FfiAtifExporter(exporter))) };
    NemoFlowStatus::Ok
}

/// Registers the exporter as an event subscriber.
///
/// # Parameters
/// - `exporter`: The exporter handle.
/// - `name`: Subscriber name (required, non-null).
///
/// # Safety
/// `exporter` and `name` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_atif_exporter_register(
    exporter: *const FfiAtifExporter,
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let subscriber = unsafe { &*exporter }.0.subscriber();
    match core_subscriber_api::register_subscriber(&name, subscriber) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregisters the exporter subscriber.
///
/// # Parameters
/// - `name`: Subscriber name (required, non-null).
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_atif_exporter_deregister(name: *const c_char) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_subscriber_api::deregister_subscriber(&name) {
        Ok(_) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Exports collected events as an ATIF trajectory JSON string.
///
/// # Parameters
/// - `exporter`: The exporter handle.
/// - `out`: On success, receives a JSON string (caller must free with
///   `nemo_flow_string_free`).
///
/// # Safety
/// `exporter` and `out` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_atif_exporter_export(
    exporter: *const FfiAtifExporter,
    out: *mut *mut c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let trajectory = unsafe { &*exporter }.0.export();
    match serde_json::to_string(&trajectory) {
        Ok(json_str) => {
            unsafe { *out = str_to_c_string(&json_str) };
            NemoFlowStatus::Ok
        }
        Err(e) => {
            set_last_error(&format!("failed to serialize trajectory: {e}"));
            NemoFlowStatus::Internal
        }
    }
}

/// Clears all collected events from the exporter.
///
/// # Parameters
/// - `exporter`: The exporter handle.
///
/// # Safety
/// `exporter` must be a valid, non-null `FfiAtifExporter` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_atif_exporter_clear(
    exporter: *const FfiAtifExporter,
) -> NemoFlowStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    unsafe { &*exporter }.0.clear();
    NemoFlowStatus::Ok
}

// ---------------------------------------------------------------------------
// OpenTelemetry subscriber
// ---------------------------------------------------------------------------

fn parse_string_map_json(
    json_ptr: *const c_char,
    field_name: &str,
) -> Result<std::collections::HashMap<String, String>, NemoFlowStatus> {
    if json_ptr.is_null() {
        return Ok(std::collections::HashMap::new());
    }

    let json_string = c_str_to_string(json_ptr)?;
    let value: serde_json::Value = serde_json::from_str(&json_string).map_err(|e| {
        set_last_error(&format!("invalid {field_name} JSON: {e}"));
        NemoFlowStatus::InvalidJson
    })?;

    let serde_json::Value::Object(map) = value else {
        set_last_error(&format!(
            "{field_name} must be a JSON object of string values"
        ));
        return Err(NemoFlowStatus::InvalidArg);
    };

    let mut out = std::collections::HashMap::with_capacity(map.len());
    for (key, value) in map {
        let serde_json::Value::String(value) = value else {
            set_last_error(&format!(
                "{field_name} must be a JSON object of string values"
            ));
            return Err(NemoFlowStatus::InvalidArg);
        };
        out.insert(key, value);
    }
    Ok(out)
}

/// Creates a new OpenTelemetry subscriber.
///
/// Nullable strings use crate defaults when omitted. `headers_json` and
/// `resource_attributes_json` must be JSON objects of string values when
/// provided.
///
/// # Safety
/// Any non-null C strings must be valid and `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_otel_subscriber_create(
    transport: *const c_char,
    endpoint: *const c_char,
    headers_json: *const c_char,
    resource_attributes_json: *const c_char,
    service_name: *const c_char,
    service_namespace: *const c_char,
    service_version: *const c_char,
    instrumentation_scope: *const c_char,
    timeout_millis: u64,
    out: *mut *mut FfiOpenTelemetrySubscriber,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }

    let transport = if transport.is_null() {
        "http_binary".to_string()
    } else {
        match c_str_to_string(transport) {
            Ok(value) => value,
            Err(status) => return status,
        }
    };

    let service_name = if service_name.is_null() {
        "nemo-flow".to_string()
    } else {
        match c_str_to_string(service_name) {
            Ok(value) => value,
            Err(status) => return status,
        }
    };

    let mut config = match transport.as_str() {
        "http_binary" => {
            nemo_flow::observability::otel::OpenTelemetryConfig::http_binary(service_name)
        }
        "grpc" => nemo_flow::observability::otel::OpenTelemetryConfig::grpc(service_name),
        other => {
            set_last_error(&format!(
                "transport must be 'http_binary' or 'grpc', got {other:?}"
            ));
            return NemoFlowStatus::InvalidArg;
        }
    };

    if !endpoint.is_null() {
        let endpoint = match c_str_to_string(endpoint) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_endpoint(endpoint);
    }
    if !service_namespace.is_null() {
        let namespace = match c_str_to_string(service_namespace) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_service_namespace(namespace);
    }
    if !service_version.is_null() {
        let version = match c_str_to_string(service_version) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_service_version(version);
    }
    if !instrumentation_scope.is_null() {
        let scope = match c_str_to_string(instrumentation_scope) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_instrumentation_scope(scope);
    }
    if timeout_millis != 0 {
        config = config.with_timeout(Duration::from_millis(timeout_millis));
    }

    for (key, value) in match parse_string_map_json(headers_json, "headers") {
        Ok(map) => map,
        Err(status) => return status,
    } {
        config = config.with_header(key, value);
    }
    for (key, value) in match parse_string_map_json(resource_attributes_json, "resource_attributes")
    {
        Ok(map) => map,
        Err(status) => return status,
    } {
        config = config.with_resource_attribute(key, value);
    }

    let _runtime_guard = tokio_runtime().enter();
    let subscriber_result = nemo_flow::observability::otel::OpenTelemetrySubscriber::new(config);
    match subscriber_result {
        Ok(subscriber) => {
            unsafe { *out = Box::into_raw(Box::new(FfiOpenTelemetrySubscriber(subscriber))) };
            NemoFlowStatus::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}

/// Registers the OpenTelemetry subscriber as an event subscriber.
///
/// # Safety
/// `subscriber` and `name` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_otel_subscriber_register(
    subscriber: *const FfiOpenTelemetrySubscriber,
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };

    match unsafe { &*subscriber }.0.register(&name) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}

/// Deregisters the OpenTelemetry subscriber by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_otel_subscriber_deregister(
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };

    match core_subscriber_api::deregister_subscriber(&name) {
        Ok(_) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Forces a flush of finished spans through the exporter.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_otel_subscriber_force_flush(
    subscriber: *const FfiOpenTelemetrySubscriber,
) -> NemoFlowStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoFlowStatus::NullPointer;
    }

    match unsafe { &*subscriber }.0.force_flush() {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}

/// Shuts down the underlying tracer provider.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_otel_subscriber_shutdown(
    subscriber: *const FfiOpenTelemetrySubscriber,
) -> NemoFlowStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoFlowStatus::NullPointer;
    }

    match unsafe { &*subscriber }.0.shutdown() {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}

/// Creates a new OpenInference subscriber.
///
/// Nullable strings use crate defaults when omitted. `headers_json` and
/// `resource_attributes_json` must be JSON objects of string values when
/// provided.
///
/// # Safety
/// Any non-null C strings must be valid and `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_openinference_subscriber_create(
    transport: *const c_char,
    endpoint: *const c_char,
    headers_json: *const c_char,
    resource_attributes_json: *const c_char,
    service_name: *const c_char,
    service_namespace: *const c_char,
    service_version: *const c_char,
    instrumentation_scope: *const c_char,
    timeout_millis: u64,
    out: *mut *mut FfiOpenInferenceSubscriber,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoFlowStatus::NullPointer;
    }

    let transport = if transport.is_null() {
        "http_binary".to_string()
    } else {
        match c_str_to_string(transport) {
            Ok(value) => value,
            Err(status) => return status,
        }
    };

    let mut config = nemo_flow::observability::openinference::OpenInferenceConfig::new();
    config = match transport.as_str() {
        "http_binary" => config
            .with_transport(nemo_flow::observability::openinference::OtlpTransport::HttpBinary),
        "grpc" => {
            config.with_transport(nemo_flow::observability::openinference::OtlpTransport::Grpc)
        }
        other => {
            set_last_error(&format!(
                "transport must be 'http_binary' or 'grpc', got {other:?}"
            ));
            return NemoFlowStatus::InvalidArg;
        }
    };

    if !service_name.is_null() {
        let value = match c_str_to_string(service_name) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_service_name(value);
    }
    if !endpoint.is_null() {
        let endpoint = match c_str_to_string(endpoint) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_endpoint(endpoint);
    }
    if !service_namespace.is_null() {
        let namespace = match c_str_to_string(service_namespace) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_service_namespace(namespace);
    }
    if !service_version.is_null() {
        let version = match c_str_to_string(service_version) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_service_version(version);
    }
    if !instrumentation_scope.is_null() {
        let scope = match c_str_to_string(instrumentation_scope) {
            Ok(value) => value,
            Err(status) => return status,
        };
        config = config.with_instrumentation_scope(scope);
    }
    if timeout_millis != 0 {
        config = config.with_timeout(Duration::from_millis(timeout_millis));
    }

    for (key, value) in match parse_string_map_json(headers_json, "headers") {
        Ok(map) => map,
        Err(status) => return status,
    } {
        config = config.with_header(key, value);
    }
    for (key, value) in match parse_string_map_json(resource_attributes_json, "resource_attributes")
    {
        Ok(map) => map,
        Err(status) => return status,
    } {
        config = config.with_resource_attribute(key, value);
    }

    let _runtime_guard = tokio_runtime().enter();
    let subscriber_result =
        nemo_flow::observability::openinference::OpenInferenceSubscriber::new(config);
    match subscriber_result {
        Ok(subscriber) => {
            unsafe { *out = Box::into_raw(Box::new(FfiOpenInferenceSubscriber(subscriber))) };
            NemoFlowStatus::Ok
        }
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}

/// Registers the OpenInference subscriber as an event subscriber.
///
/// # Safety
/// `subscriber` and `name` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_openinference_subscriber_register(
    subscriber: *const FfiOpenInferenceSubscriber,
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };

    match unsafe { &*subscriber }.0.register(&name) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}

/// Deregisters the OpenInference subscriber by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_openinference_subscriber_deregister(
    name: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };

    match core_subscriber_api::deregister_subscriber(&name) {
        Ok(_) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Forces a flush of finished spans through the exporter.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_openinference_subscriber_force_flush(
    subscriber: *const FfiOpenInferenceSubscriber,
) -> NemoFlowStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoFlowStatus::NullPointer;
    }

    match unsafe { &*subscriber }.0.force_flush() {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}

/// Shuts down the underlying tracer provider.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_openinference_subscriber_shutdown(
    subscriber: *const FfiOpenInferenceSubscriber,
) -> NemoFlowStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoFlowStatus::NullPointer;
    }

    match unsafe { &*subscriber }.0.shutdown() {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoFlowStatus::Internal
        }
    }
}
