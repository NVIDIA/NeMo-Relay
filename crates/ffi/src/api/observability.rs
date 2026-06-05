// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    Duration, FfiAtifExporter, FfiAtofExporter, FfiOpenInferenceSubscriber,
    FfiOpenTelemetrySubscriber, NemoRelayStatus, c_char, c_str_to_json, c_str_to_string,
    clear_last_error, core_subscriber_api, json_to_c_string, set_last_error, status_from_error,
    str_to_c_string, tokio_runtime,
};

type AtofExporter = nemo_relay::observability::atof::AtofExporter;
type AtofExporterConfig = nemo_relay::observability::atof::AtofExporterConfig;
type AtofExporterError = nemo_relay::observability::atof::AtofExporterError;
type AtofExporterMode = nemo_relay::observability::atof::AtofExporterMode;
type OpenTelemetryConfig = nemo_relay::observability::otel::OpenTelemetryConfig;
type OpenTelemetrySubscriber = nemo_relay::observability::otel::OpenTelemetrySubscriber;
type OpenInferenceConfig = nemo_relay::observability::openinference::OpenInferenceConfig;
type OpenInferenceSubscriber = nemo_relay::observability::openinference::OpenInferenceSubscriber;
type ObservabilityComponentSpec = nemo_relay::observability::plugin_component::ComponentSpec;
type ObservabilityConfig = nemo_relay::observability::plugin_component::ObservabilityConfig;

fn status_from_atof_error(error: &AtofExporterError) -> NemoRelayStatus {
    set_last_error(&error.to_string());
    match error {
        AtofExporterError::Runtime(error) => status_from_error(error),
        _ => NemoRelayStatus::Internal,
    }
}

// ---------------------------------------------------------------------------
// Observability plugin component helpers
// ---------------------------------------------------------------------------

/// Return the built-in observability plugin kind.
///
/// The caller owns the returned string and must free it with `nemo_relay_string_free`.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_relay_observability_plugin_kind() -> *mut c_char {
    str_to_c_string(nemo_relay::observability::plugin_component::OBSERVABILITY_PLUGIN_KIND)
}

/// Return the default observability plugin config as JSON.
///
/// # Safety
/// `out_json` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_observability_default_config_json(
    out_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out_json.is_null() {
        set_last_error("out_json pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let config_json = match serde_json::to_value(ObservabilityConfig::default()) {
        Ok(value) => value,
        Err(error) => {
            set_last_error(&error.to_string());
            return NemoRelayStatus::Internal;
        }
    };
    unsafe { *out_json = json_to_c_string(&config_json) };
    NemoRelayStatus::Ok
}

/// Wrap an observability config JSON object as a top-level plugin component.
///
/// Pass null for `config_json` to use the default observability config. The
/// returned JSON can be inserted into `PluginConfig.components`.
///
/// # Safety
/// `config_json`, when non-null, must be a valid C string. `out_json` must be a
/// valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_observability_component_spec_json(
    config_json: *const c_char,
    enabled: bool,
    out_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out_json.is_null() {
        set_last_error("out_json pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let config = if config_json.is_null() {
        ObservabilityConfig::default()
    } else {
        let Some(config_value) = c_str_to_json(config_json) else {
            return NemoRelayStatus::InvalidJson;
        };
        match serde_json::from_value::<ObservabilityConfig>(config_value) {
            Ok(config) => config,
            Err(error) => {
                set_last_error(&error.to_string());
                return NemoRelayStatus::InvalidJson;
            }
        }
    };
    let component: nemo_relay::plugin::PluginComponentSpec =
        ObservabilityComponentSpec { enabled, config }.into();
    let component_json = match serde_json::to_value(component) {
        Ok(value) => value,
        Err(error) => {
            set_last_error(&error.to_string());
            return NemoRelayStatus::Internal;
        }
    };
    unsafe { *out_json = json_to_c_string(&component_json) };
    NemoRelayStatus::Ok
}

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
pub unsafe extern "C" fn nemo_relay_atif_exporter_create(
    session_id: *const c_char,
    agent_name: *const c_char,
    agent_version: *const c_char,
    model_name: *const c_char,
    out: *mut *mut FfiAtifExporter,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoRelayStatus::NullPointer;
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

    let agent_info = nemo_relay::observability::atif::AtifAgentInfo {
        name: agent_name,
        version: agent_version,
        model_name: model_name_opt,
        tool_definitions: None,
        extra: None,
    };

    let exporter = nemo_relay::observability::atif::AtifExporter::new(session_id, agent_info);
    unsafe { *out = Box::into_raw(Box::new(FfiAtifExporter(exporter))) };
    NemoRelayStatus::Ok
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
pub unsafe extern "C" fn nemo_relay_atif_exporter_register(
    exporter: *const FfiAtifExporter,
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let subscriber = unsafe { &*exporter }.0.subscriber();
    match core_subscriber_api::register_subscriber(&name, subscriber) {
        Ok(()) => NemoRelayStatus::Ok,
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
pub unsafe extern "C" fn nemo_relay_atif_exporter_deregister(
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_subscriber_api::deregister_subscriber(&name) {
        Ok(_) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Exports collected events as an ATIF trajectory JSON string.
///
/// # Parameters
/// - `exporter`: The exporter handle.
/// - `out`: On success, receives a JSON string (caller must free with
///   `nemo_relay_string_free`).
///
/// # Safety
/// `exporter` and `out` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_atif_exporter_export(
    exporter: *const FfiAtifExporter,
    out: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let trajectory = match unsafe { &*exporter }.0.try_export() {
        Ok(trajectory) => trajectory,
        Err(e) => return status_from_error(&e),
    };
    match serde_json::to_string(&trajectory) {
        Ok(json_str) => {
            unsafe { *out = str_to_c_string(&json_str) };
            NemoRelayStatus::Ok
        }
        Err(e) => {
            set_last_error(&format!("failed to serialize trajectory: {e}"));
            NemoRelayStatus::Internal
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
pub unsafe extern "C" fn nemo_relay_atif_exporter_clear(
    exporter: *const FfiAtifExporter,
) -> NemoRelayStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    unsafe { &*exporter }.0.clear();
    NemoRelayStatus::Ok
}

// ---------------------------------------------------------------------------
// ATOF JSONL exporter
// ---------------------------------------------------------------------------

/// Creates a new filesystem-backed ATOF JSONL exporter.
///
/// # Parameters
/// - `output_directory`: Output directory path (nullable for current directory).
/// - `mode`: `"append"` or `"overwrite"` (nullable for `"append"`).
/// - `filename`: Output filename (nullable for generated default).
/// - `out`: On success, receives a heap-allocated `FfiAtofExporter`.
///
/// # Safety
/// All non-null string pointers must be valid C strings. `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_atof_exporter_create(
    output_directory: *const c_char,
    mode: *const c_char,
    filename: *const c_char,
    out: *mut *mut FfiAtofExporter,
) -> NemoRelayStatus {
    clear_last_error();
    if let Err(status) = required_out_ptr(out) {
        return status;
    }

    let output_directory = match parse_optional_string(output_directory) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mode = match parse_string_or_default(mode, "append") {
        Ok(value) => value,
        Err(status) => return status,
    };
    let filename = match parse_optional_string(filename) {
        Ok(value) => value,
        Err(status) => return status,
    };

    let Some(mode) = AtofExporterMode::parse(&mode) else {
        set_last_error("ATOF exporter mode must be 'append' or 'overwrite'");
        return NemoRelayStatus::InvalidArg;
    };

    let mut config = AtofExporterConfig::new().with_mode(mode);
    if let Some(output_directory) = output_directory {
        config = config.with_output_directory(output_directory);
    }
    if let Some(filename) = filename {
        config = config.with_filename(filename);
    }

    match AtofExporter::new(config) {
        Ok(exporter) => {
            unsafe { *out = Box::into_raw(Box::new(FfiAtofExporter(exporter))) };
            NemoRelayStatus::Ok
        }
        Err(error) => status_from_atof_error(&error),
    }
}

/// Creates a new ATOF exporter from a JSON config object.
///
/// # Parameters
/// - `config_json`: JSON object matching `AtofExporterConfig`.
/// - `out`: On success, receives a heap-allocated `FfiAtofExporter`.
///
/// # Safety
/// `config_json` must be a valid C string. `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_atof_exporter_create_from_json(
    config_json: *const c_char,
    out: *mut *mut FfiAtofExporter,
) -> NemoRelayStatus {
    clear_last_error();
    if let Err(status) = required_out_ptr(out) {
        return status;
    }
    let config_json = match c_str_to_string(config_json) {
        Ok(config_json) => config_json,
        Err(status) => return status,
    };
    let config_value = match serde_json::from_str(&config_json) {
        Ok(config_value) => config_value,
        Err(error) => {
            set_last_error(&format!("invalid JSON: {error}"));
            return NemoRelayStatus::InvalidJson;
        }
    };
    let config = match serde_json::from_value::<AtofExporterConfig>(config_value) {
        Ok(config) => config,
        Err(error) => {
            set_last_error(&error.to_string());
            return NemoRelayStatus::InvalidJson;
        }
    };
    match AtofExporter::new(config) {
        Ok(exporter) => {
            unsafe { *out = Box::into_raw(Box::new(FfiAtofExporter(exporter))) };
            NemoRelayStatus::Ok
        }
        Err(error) => status_from_atof_error(&error),
    }
}

/// Registers the ATOF exporter as an event subscriber.
///
/// # Safety
/// `exporter` and `name` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_atof_exporter_register(
    exporter: *const FfiAtofExporter,
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match unsafe { &*exporter }.0.register(&name) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => status_from_atof_error(&error),
    }
}

/// Deregisters the ATOF exporter subscriber.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_atof_exporter_deregister(
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core_subscriber_api::deregister_subscriber(&name) {
        Ok(_) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Flushes the ATOF exporter output file.
///
/// # Safety
/// `exporter` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_atof_exporter_force_flush(
    exporter: *const FfiAtofExporter,
) -> NemoRelayStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    match unsafe { &*exporter }.0.force_flush() {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => status_from_atof_error(&error),
    }
}

/// Shuts down the ATOF exporter by flushing output.
///
/// # Safety
/// `exporter` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_atof_exporter_shutdown(
    exporter: *const FfiAtofExporter,
) -> NemoRelayStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    match unsafe { &*exporter }.0.shutdown() {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => status_from_atof_error(&error),
    }
}

/// Returns the ATOF exporter output path as a string.
///
/// # Safety
/// `exporter` and `out` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_atof_exporter_path(
    exporter: *const FfiAtofExporter,
    out: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if exporter.is_null() {
        set_last_error("exporter pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let path = unsafe { &*exporter }.0.path().to_string_lossy();
    unsafe { *out = str_to_c_string(&path) };
    NemoRelayStatus::Ok
}

// ---------------------------------------------------------------------------
// OpenTelemetry subscriber
// ---------------------------------------------------------------------------

fn parse_string_map_json(
    json_ptr: *const c_char,
    field_name: &str,
) -> Result<std::collections::HashMap<String, String>, NemoRelayStatus> {
    if json_ptr.is_null() {
        return Ok(std::collections::HashMap::new());
    }

    let json_string = c_str_to_string(json_ptr)?;
    let value: serde_json::Value = serde_json::from_str(&json_string).map_err(|e| {
        set_last_error(&format!("invalid {field_name} JSON: {e}"));
        NemoRelayStatus::InvalidJson
    })?;

    let serde_json::Value::Object(map) = value else {
        set_last_error(&format!(
            "{field_name} must be a JSON object of string values"
        ));
        return Err(NemoRelayStatus::InvalidArg);
    };

    let mut out = std::collections::HashMap::with_capacity(map.len());
    for (key, value) in map {
        let serde_json::Value::String(value) = value else {
            set_last_error(&format!(
                "{field_name} must be a JSON object of string values"
            ));
            return Err(NemoRelayStatus::InvalidArg);
        };
        out.insert(key, value);
    }
    Ok(out)
}

fn required_out_ptr<T>(out: *mut *mut T) -> Result<(), NemoRelayStatus> {
    if out.is_null() {
        set_last_error("out pointer is null");
        return Err(NemoRelayStatus::NullPointer);
    }
    Ok(())
}

fn parse_optional_string(ptr: *const c_char) -> Result<Option<String>, NemoRelayStatus> {
    if ptr.is_null() {
        Ok(None)
    } else {
        c_str_to_string(ptr).map(Some)
    }
}

fn parse_string_or_default(ptr: *const c_char, default: &str) -> Result<String, NemoRelayStatus> {
    parse_optional_string(ptr).map(|value| value.unwrap_or_else(|| default.to_string()))
}

fn apply_optional_string<T, F>(
    config: T,
    ptr: *const c_char,
    apply: F,
) -> Result<T, NemoRelayStatus>
where
    F: FnOnce(T, String) -> T,
{
    Ok(match parse_optional_string(ptr)? {
        Some(value) => apply(config, value),
        None => config,
    })
}

fn apply_timeout<T, F>(config: T, timeout_millis: u64, apply: F) -> T
where
    F: FnOnce(T, Duration) -> T,
{
    if timeout_millis != 0 {
        apply(config, Duration::from_millis(timeout_millis))
    } else {
        config
    }
}

fn apply_string_map<T, F>(
    mut config: T,
    json_ptr: *const c_char,
    field_name: &str,
    mut apply: F,
) -> Result<T, NemoRelayStatus>
where
    F: FnMut(T, String, String) -> T,
{
    for (key, value) in parse_string_map_json(json_ptr, field_name)? {
        config = apply(config, key, value);
    }
    Ok(config)
}

fn parse_transport(ptr: *const c_char) -> Result<String, NemoRelayStatus> {
    parse_string_or_default(ptr, "http_binary")
}

fn otel_config_for_transport(
    transport: &str,
    service_name: String,
) -> Result<OpenTelemetryConfig, NemoRelayStatus> {
    match transport {
        "http_binary" => Ok(OpenTelemetryConfig::http_binary(service_name)),
        "grpc" => Ok(OpenTelemetryConfig::grpc(service_name)),
        other => {
            set_last_error(&format!(
                "transport must be 'http_binary' or 'grpc', got {other:?}"
            ));
            Err(NemoRelayStatus::InvalidArg)
        }
    }
}

fn openinference_config_for_transport(
    transport: &str,
) -> Result<OpenInferenceConfig, NemoRelayStatus> {
    match transport {
        "http_binary" => Ok(OpenInferenceConfig::new()
            .with_transport(nemo_relay::observability::openinference::OtlpTransport::HttpBinary)),
        "grpc" => Ok(OpenInferenceConfig::new()
            .with_transport(nemo_relay::observability::openinference::OtlpTransport::Grpc)),
        other => {
            set_last_error(&format!(
                "transport must be 'http_binary' or 'grpc', got {other:?}"
            ));
            Err(NemoRelayStatus::InvalidArg)
        }
    }
}

fn create_otel_subscriber(
    config: OpenTelemetryConfig,
) -> Result<OpenTelemetrySubscriber, NemoRelayStatus> {
    let _runtime_guard = tokio_runtime().enter();
    OpenTelemetrySubscriber::new(config).map_err(|error| {
        set_last_error(&error.to_string());
        NemoRelayStatus::Internal
    })
}

fn create_openinference_subscriber(
    config: OpenInferenceConfig,
) -> Result<OpenInferenceSubscriber, NemoRelayStatus> {
    let _runtime_guard = tokio_runtime().enter();
    OpenInferenceSubscriber::new(config).map_err(|error| {
        set_last_error(&error.to_string());
        NemoRelayStatus::Internal
    })
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
pub unsafe extern "C" fn nemo_relay_otel_subscriber_create(
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
) -> NemoRelayStatus {
    clear_last_error();
    if let Err(status) = required_out_ptr(out) {
        return status;
    }

    let transport = match parse_transport(transport) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let service_name = match parse_string_or_default(service_name, "nemo-relay") {
        Ok(value) => value,
        Err(status) => return status,
    };

    let mut config = match otel_config_for_transport(&transport, service_name) {
        Ok(config) => config,
        Err(status) => return status,
    };
    config = match apply_optional_string(config, endpoint, OpenTelemetryConfig::with_endpoint) {
        Ok(config) => config,
        Err(status) => return status,
    };
    config = match apply_optional_string(
        config,
        service_namespace,
        OpenTelemetryConfig::with_service_namespace,
    ) {
        Ok(config) => config,
        Err(status) => return status,
    };
    config = match apply_optional_string(
        config,
        service_version,
        OpenTelemetryConfig::with_service_version,
    ) {
        Ok(config) => config,
        Err(status) => return status,
    };
    config = match apply_optional_string(
        config,
        instrumentation_scope,
        OpenTelemetryConfig::with_instrumentation_scope,
    ) {
        Ok(config) => config,
        Err(status) => return status,
    };
    config = apply_timeout(config, timeout_millis, OpenTelemetryConfig::with_timeout);
    config = match apply_string_map(
        config,
        headers_json,
        "headers",
        OpenTelemetryConfig::with_header,
    ) {
        Ok(config) => config,
        Err(status) => return status,
    };
    config = match apply_string_map(
        config,
        resource_attributes_json,
        "resource_attributes",
        OpenTelemetryConfig::with_resource_attribute,
    ) {
        Ok(config) => config,
        Err(status) => return status,
    };

    let subscriber = match create_otel_subscriber(config) {
        Ok(subscriber) => subscriber,
        Err(status) => return status,
    };
    unsafe { *out = Box::into_raw(Box::new(FfiOpenTelemetrySubscriber(subscriber))) };
    NemoRelayStatus::Ok
}

/// Registers the OpenTelemetry subscriber as an event subscriber.
///
/// # Safety
/// `subscriber` and `name` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_subscriber_register(
    subscriber: *const FfiOpenTelemetrySubscriber,
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };

    match unsafe { &*subscriber }.0.register(&name) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoRelayStatus::Internal
        }
    }
}

/// Deregisters the OpenTelemetry subscriber by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_subscriber_deregister(
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };

    match core_subscriber_api::deregister_subscriber(&name) {
        Ok(_) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Forces a flush of finished spans through the exporter.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_subscriber_force_flush(
    subscriber: *const FfiOpenTelemetrySubscriber,
) -> NemoRelayStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoRelayStatus::NullPointer;
    }

    match unsafe { &*subscriber }.0.force_flush() {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoRelayStatus::Internal
        }
    }
}

/// Shuts down the underlying tracer provider.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_subscriber_shutdown(
    subscriber: *const FfiOpenTelemetrySubscriber,
) -> NemoRelayStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoRelayStatus::NullPointer;
    }

    match unsafe { &*subscriber }.0.shutdown() {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoRelayStatus::Internal
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
pub unsafe extern "C" fn nemo_relay_openinference_subscriber_create(
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
) -> NemoRelayStatus {
    clear_last_error();
    if let Err(status) = required_out_ptr(out) {
        return status;
    }

    let transport = match parse_transport(transport) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut config = match openinference_config_for_transport(&transport) {
        Ok(config) => config,
        Err(status) => return status,
    };
    config =
        match apply_optional_string(config, service_name, OpenInferenceConfig::with_service_name) {
            Ok(config) => config,
            Err(status) => return status,
        };
    config = match apply_optional_string(config, endpoint, OpenInferenceConfig::with_endpoint) {
        Ok(config) => config,
        Err(status) => return status,
    };
    config = match apply_optional_string(
        config,
        service_namespace,
        OpenInferenceConfig::with_service_namespace,
    ) {
        Ok(config) => config,
        Err(status) => return status,
    };
    config = match apply_optional_string(
        config,
        service_version,
        OpenInferenceConfig::with_service_version,
    ) {
        Ok(config) => config,
        Err(status) => return status,
    };
    config = match apply_optional_string(
        config,
        instrumentation_scope,
        OpenInferenceConfig::with_instrumentation_scope,
    ) {
        Ok(config) => config,
        Err(status) => return status,
    };
    config = apply_timeout(config, timeout_millis, OpenInferenceConfig::with_timeout);
    config = match apply_string_map(
        config,
        headers_json,
        "headers",
        OpenInferenceConfig::with_header,
    ) {
        Ok(config) => config,
        Err(status) => return status,
    };
    config = match apply_string_map(
        config,
        resource_attributes_json,
        "resource_attributes",
        OpenInferenceConfig::with_resource_attribute,
    ) {
        Ok(config) => config,
        Err(status) => return status,
    };

    let subscriber = match create_openinference_subscriber(config) {
        Ok(subscriber) => subscriber,
        Err(status) => return status,
    };
    unsafe { *out = Box::into_raw(Box::new(FfiOpenInferenceSubscriber(subscriber))) };
    NemoRelayStatus::Ok
}

/// Registers the OpenInference subscriber as an event subscriber.
///
/// # Safety
/// `subscriber` and `name` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_openinference_subscriber_register(
    subscriber: *const FfiOpenInferenceSubscriber,
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };

    match unsafe { &*subscriber }.0.register(&name) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoRelayStatus::Internal
        }
    }
}

/// Deregisters the OpenInference subscriber by name.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_openinference_subscriber_deregister(
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };

    match core_subscriber_api::deregister_subscriber(&name) {
        Ok(_) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Forces a flush of finished spans through the exporter.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_openinference_subscriber_force_flush(
    subscriber: *const FfiOpenInferenceSubscriber,
) -> NemoRelayStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoRelayStatus::NullPointer;
    }

    match unsafe { &*subscriber }.0.force_flush() {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoRelayStatus::Internal
        }
    }
}

/// Shuts down the underlying tracer provider.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_openinference_subscriber_shutdown(
    subscriber: *const FfiOpenInferenceSubscriber,
) -> NemoRelayStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoRelayStatus::NullPointer;
    }

    match unsafe { &*subscriber }.0.shutdown() {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => {
            set_last_error(&e.to_string());
            NemoRelayStatus::Internal
        }
    }
}
