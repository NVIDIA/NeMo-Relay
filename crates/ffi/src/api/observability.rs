// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    Duration, FfiAtifExporter, FfiAtofExporter, FfiOpenTelemetryLogSubscriber,
    FfiOpenTelemetryMetricSubscriber, FfiOpenTelemetrySubscriber, NemoRelayStatus, c_char,
    c_str_to_json, c_str_to_string, clear_last_error, core_subscriber_api, json_to_c_string,
    set_last_error, status_from_error, str_to_c_string, tokio_runtime,
};

type AtofExporter = nemo_relay::observability::atof::AtofExporter;
type AtofExporterConfig = nemo_relay::observability::atof::AtofExporterConfig;
type AtofExporterError = nemo_relay::observability::atof::AtofExporterError;
type AtofExporterMode = nemo_relay::observability::atof::AtofExporterMode;
type OpenTelemetryConfig = nemo_relay::observability::otel::OpenTelemetryConfig;
type OpenTelemetrySubscriber = nemo_relay::observability::otel::OpenTelemetrySubscriber;
type OpenTelemetryLogConfig = nemo_relay::observability::otel_logs::OpenTelemetryLogConfig;
type OpenTelemetryLogSubscriber = nemo_relay::observability::otel_logs::OpenTelemetryLogSubscriber;
type OpenTelemetryMetricConfig = nemo_relay::observability::otel_metrics::OpenTelemetryMetricConfig;
type OpenTelemetryMetricSubscriber =
    nemo_relay::observability::otel_metrics::OpenTelemetryMetricSubscriber;
type ObservabilityComponentSpec = nemo_relay::observability::plugin_component::ComponentSpec;
type ObservabilityConfig = nemo_relay::observability::plugin_component::ObservabilityConfig;

fn status_from_atof_error(error: &AtofExporterError) -> NemoRelayStatus {
    set_last_error(&error.to_string());
    match error {
        AtofExporterError::Runtime(error) => status_from_error(error),
        AtofExporterError::InvalidEndpoint(_) => NemoRelayStatus::InvalidArg,
        _ => NemoRelayStatus::Internal,
    }
}

/// Write `diagnostics` to a valid, non-null C-string output slot.
///
/// # Safety
/// `out_json` must point to writable storage for one C-string pointer.
unsafe fn write_runtime_diagnostics(
    diagnostics: nemo_relay::observability::OpenTelemetryRuntimeDiagnostics,
    out_json: *mut *mut c_char,
) -> NemoRelayStatus {
    let entries = serde_json::Value::Array(
        diagnostics
            .entries()
            .iter()
            .map(|diagnostic| {
                serde_json::json!({
                    "code": diagnostic.code,
                    "message": diagnostic.message,
                    "count": diagnostic.count,
                })
            })
            .collect(),
    );
    unsafe { *out_json = json_to_c_string(&entries) };
    NemoRelayStatus::Ok
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

/// Outside a native subscriber callback, waits for queued subscriber delivery, then flushes the
/// configured file sink or asks the configured stream sink to drain for up to its timeout.
///
/// A re-entrant call does not establish the delivery barrier. A stream timeout is logged and does
/// not by itself return an error.
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

/// Outside a native subscriber callback, waits for queued subscriber delivery, then flushes the
/// configured file sink or asks the configured stream sink to drain and close up to its timeout.
///
/// A re-entrant call does not establish the delivery barrier. A stream timeout is logged and does
/// not by itself return an error.
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

/// Returns the ATOF exporter output path as a string when its sink is a file.
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
    let Some(path) = unsafe { &*exporter }.0.path() else {
        unsafe { *out = std::ptr::null_mut() };
        return NemoRelayStatus::Ok;
    };
    let path = path.to_string_lossy();
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

fn parse_otlp_transport(
    ptr: *const c_char,
) -> Result<nemo_relay::observability::otel::OtlpTransport, NemoRelayStatus> {
    transport_from_str(&parse_transport(ptr)?)
}

fn transport_from_str(
    value: &str,
) -> Result<nemo_relay::observability::otel::OtlpTransport, NemoRelayStatus> {
    match value {
        "http_binary" => Ok(nemo_relay::observability::otel::OtlpTransport::HttpBinary),
        "grpc" => Ok(nemo_relay::observability::otel::OtlpTransport::Grpc),
        other => {
            set_last_error(&format!(
                "transport must be 'http_binary' or 'grpc', got {other:?}"
            ));
            Err(NemoRelayStatus::InvalidArg)
        }
    }
}

fn parse_mark_projection(
    ptr: *const c_char,
) -> Result<nemo_relay::observability::MarkProjection, NemoRelayStatus> {
    let value = parse_string_or_default(ptr, "inherit")?;
    serde_json::from_value(serde_json::Value::String(value)).map_err(|error| {
        set_last_error(&error.to_string());
        NemoRelayStatus::InvalidArg
    })
}

fn parse_mark_exclude_names(ptr: *const c_char) -> Result<Vec<String>, NemoRelayStatus> {
    if ptr.is_null() {
        return Ok(nemo_relay::observability::default_mark_exclude_names());
    }
    let Some(value) = c_str_to_json(ptr) else {
        return Err(NemoRelayStatus::InvalidJson);
    };
    serde_json::from_value(value).map_err(|error| {
        set_last_error(&format!(
            "mark_exclude_names must be an array of strings: {error}"
        ));
        NemoRelayStatus::InvalidArg
    })
}

fn parse_promote_metadata_prefixes(ptr: *const c_char) -> Result<Vec<String>, NemoRelayStatus> {
    if ptr.is_null() {
        return Ok(Vec::new());
    }
    let Some(value) = c_str_to_json(ptr) else {
        return Err(NemoRelayStatus::InvalidJson);
    };
    let prefixes: Vec<String> = serde_json::from_value(value).map_err(|error| {
        set_last_error(&format!(
            "promote_metadata_prefixes must be an array of strings: {error}"
        ));
        NemoRelayStatus::InvalidArg
    })?;
    nemo_relay::observability::validate_metadata_promotion_prefixes(&prefixes).map_err(
        |error| {
            set_last_error(&error);
            NemoRelayStatus::InvalidArg
        },
    )?;
    Ok(prefixes)
}

fn parse_attribute_mappings(
    ptr: *const c_char,
) -> Result<Vec<nemo_relay::observability::OtlpAttributeMapping>, NemoRelayStatus> {
    if ptr.is_null() {
        return Ok(Vec::new());
    }
    let Some(value) = c_str_to_json(ptr) else {
        return Err(NemoRelayStatus::InvalidJson);
    };
    let mappings: Vec<nemo_relay::observability::OtlpAttributeMapping> =
        serde_json::from_value(value).map_err(|error| {
            set_last_error(&format!(
                "attribute_mappings must be an array of mappings: {error}"
            ));
            NemoRelayStatus::InvalidArg
        })?;
    nemo_relay::observability::validate_attribute_mappings(&mappings).map_err(|error| {
        set_last_error(&error.to_string());
        NemoRelayStatus::InvalidArg
    })?;
    Ok(mappings)
}

fn otel_config_for_transport(
    transport: &str,
    otel_type: nemo_relay::observability::OpenTelemetryType,
    endpoint: String,
    service_name: String,
) -> Result<OpenTelemetryConfig, NemoRelayStatus> {
    let transport = transport_from_str(transport)?;
    Ok(OpenTelemetryConfig::new(otel_type, endpoint)
        .with_transport(transport)
        .with_service_name(service_name))
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

fn parse_otel_type(
    ptr: *const c_char,
) -> Result<nemo_relay::observability::OpenTelemetryType, NemoRelayStatus> {
    match parse_optional_string(ptr)? {
        Some(value) if value == "full" => Ok(nemo_relay::observability::OpenTelemetryType::Full),
        Some(value) if value == "gen_ai" => Ok(nemo_relay::observability::OpenTelemetryType::GenAi),
        Some(value) if value == "openinference" => {
            Ok(nemo_relay::observability::OpenTelemetryType::OpenInference)
        }
        Some(value) => {
            set_last_error(&format!(
                "type must be 'full', 'gen_ai', or 'openinference', got {value:?}"
            ));
            Err(NemoRelayStatus::InvalidArg)
        }
        None => {
            set_last_error("type is required");
            Err(NemoRelayStatus::InvalidArg)
        }
    }
}

fn parse_required_otel_endpoint(ptr: *const c_char) -> Result<String, NemoRelayStatus> {
    match parse_optional_string(ptr)? {
        Some(value) if !value.trim().is_empty() => Ok(value),
        _ => {
            set_last_error("endpoint is required and must be nonblank");
            Err(NemoRelayStatus::InvalidArg)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_ffi_otel_config(
    otel_type: *const c_char,
    transport: *const c_char,
    endpoint: *const c_char,
    headers_json: *const c_char,
    header_env_json: *const c_char,
    resource_attributes_json: *const c_char,
    service_name: *const c_char,
    service_namespace: *const c_char,
    service_version: *const c_char,
    instrumentation_scope: *const c_char,
    timeout_millis: u64,
) -> Result<OpenTelemetryConfig, NemoRelayStatus> {
    let mut config = otel_config_for_transport(
        &parse_transport(transport)?,
        parse_otel_type(otel_type)?,
        parse_required_otel_endpoint(endpoint)?,
        parse_string_or_default(service_name, "unknown_service")?,
    )?;
    config = apply_optional_string(
        config,
        service_namespace,
        OpenTelemetryConfig::with_service_namespace,
    )?;
    config = apply_optional_string(
        config,
        service_version,
        OpenTelemetryConfig::with_service_version,
    )?;
    config = config
        .with_instrumentation_scope(parse_string_or_default(
            instrumentation_scope,
            "opentelemetry",
        )?)
        .with_timeout(Duration::from_millis(if timeout_millis == 0 {
            3_000
        } else {
            timeout_millis
        }));
    config = apply_string_map(
        config,
        headers_json,
        "headers",
        OpenTelemetryConfig::with_header,
    )?;
    config = apply_string_map(
        config,
        header_env_json,
        "header_env",
        OpenTelemetryConfig::with_header_env,
    )?;
    apply_string_map(
        config,
        resource_attributes_json,
        "resource_attributes",
        OpenTelemetryConfig::with_resource_attribute,
    )
}

/// Creates one typed OpenTelemetry exporter subscriber.
///
/// `otel_type` must be `full`, `gen_ai`, or `openinference`. `endpoint` is required.
///
/// # Safety
/// Any non-null C strings must be valid and `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_subscriber_create(
    otel_type: *const c_char,
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
    let config = match build_ffi_otel_config(
        otel_type,
        transport,
        endpoint,
        headers_json,
        std::ptr::null(),
        resource_attributes_json,
        service_name,
        service_namespace,
        service_version,
        instrumentation_scope,
        timeout_millis,
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

/// Creates one typed OpenTelemetry exporter subscriber with projection controls.
///
/// The JSON arrays use `mark_exclude_names: ["llm.chunk"]` and
/// `attribute_mappings: [{"key":"…","alias":"…"}]` shapes. Pass null for either
/// array to use its default. `mark_projection` is `inherit`, `event`, or `tool`.
///
/// # Safety
/// Any non-null C strings must be valid and `out` must be non-null.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_subscriber_create_with_projection_options(
    otel_type: *const c_char,
    transport: *const c_char,
    endpoint: *const c_char,
    headers_json: *const c_char,
    resource_attributes_json: *const c_char,
    service_name: *const c_char,
    service_namespace: *const c_char,
    service_version: *const c_char,
    instrumentation_scope: *const c_char,
    timeout_millis: u64,
    mark_projection: *const c_char,
    mark_exclude_names_json: *const c_char,
    attribute_mappings_json: *const c_char,
    out: *mut *mut FfiOpenTelemetrySubscriber,
) -> NemoRelayStatus {
    unsafe {
        nemo_relay_otel_subscriber_create_with_projection_options_v2(
            otel_type,
            transport,
            endpoint,
            headers_json,
            resource_attributes_json,
            service_name,
            service_namespace,
            service_version,
            instrumentation_scope,
            timeout_millis,
            mark_projection,
            mark_exclude_names_json,
            attribute_mappings_json,
            std::ptr::null(),
            out,
        )
    }
}

/// Creates one typed OpenTelemetry exporter subscriber with projection and metadata controls.
///
/// `promote_metadata_prefixes_json` is a JSON array of literal metadata prefixes,
/// such as `["nv."]`. Pass null to disable metadata promotion.
/// `completed_span_context_ttl_millis` must be greater than zero.
///
/// # Safety
/// Any non-null C strings must be valid and `out` must be non-null.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_subscriber_create_with_projection_options_v4(
    otel_type: *const c_char,
    transport: *const c_char,
    endpoint: *const c_char,
    headers_json: *const c_char,
    header_env_json: *const c_char,
    resource_attributes_json: *const c_char,
    service_name: *const c_char,
    service_namespace: *const c_char,
    service_version: *const c_char,
    instrumentation_scope: *const c_char,
    timeout_millis: u64,
    mark_projection: *const c_char,
    mark_exclude_names_json: *const c_char,
    attribute_mappings_json: *const c_char,
    promote_metadata_prefixes_json: *const c_char,
    completed_span_context_ttl_millis: u64,
    out: *mut *mut FfiOpenTelemetrySubscriber,
) -> NemoRelayStatus {
    clear_last_error();
    if let Err(status) = required_out_ptr(out) {
        return status;
    }
    let config = match build_ffi_otel_config(
        otel_type,
        transport,
        endpoint,
        headers_json,
        header_env_json,
        resource_attributes_json,
        service_name,
        service_namespace,
        service_version,
        instrumentation_scope,
        timeout_millis,
    ) {
        Ok(config) => config,
        Err(status) => return status,
    };
    if completed_span_context_ttl_millis == 0 {
        set_last_error("completed_span_context_ttl_millis must be greater than 0");
        return NemoRelayStatus::InvalidArg;
    }
    let config = config
        .with_mark_projection(match parse_mark_projection(mark_projection) {
            Ok(value) => value,
            Err(status) => return status,
        })
        .with_mark_exclude_names(match parse_mark_exclude_names(mark_exclude_names_json) {
            Ok(value) => value,
            Err(status) => return status,
        })
        .with_attribute_mappings(match parse_attribute_mappings(attribute_mappings_json) {
            Ok(value) => value,
            Err(status) => return status,
        })
        .with_promote_metadata_prefixes(
            match parse_promote_metadata_prefixes(promote_metadata_prefixes_json) {
                Ok(value) => value,
                Err(status) => return status,
            },
        )
        .with_completed_span_context_ttl(Duration::from_millis(completed_span_context_ttl_millis));
    let subscriber = match create_otel_subscriber(config) {
        Ok(subscriber) => subscriber,
        Err(status) => return status,
    };
    unsafe { *out = Box::into_raw(Box::new(FfiOpenTelemetrySubscriber(subscriber))) };
    NemoRelayStatus::Ok
}

/// Creates one typed OpenTelemetry exporter subscriber with projection, metadata, and lineage controls.
///
/// This compatibility entrypoint configures only static headers. Use
/// `nemo_relay_otel_subscriber_create_with_projection_options_v4` for `header_env`.
///
/// # Safety
/// Any non-null C strings must be valid and `out` must be non-null.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_subscriber_create_with_projection_options_v3(
    otel_type: *const c_char,
    transport: *const c_char,
    endpoint: *const c_char,
    headers_json: *const c_char,
    resource_attributes_json: *const c_char,
    service_name: *const c_char,
    service_namespace: *const c_char,
    service_version: *const c_char,
    instrumentation_scope: *const c_char,
    timeout_millis: u64,
    mark_projection: *const c_char,
    mark_exclude_names_json: *const c_char,
    attribute_mappings_json: *const c_char,
    promote_metadata_prefixes_json: *const c_char,
    completed_span_context_ttl_millis: u64,
    out: *mut *mut FfiOpenTelemetrySubscriber,
) -> NemoRelayStatus {
    unsafe {
        nemo_relay_otel_subscriber_create_with_projection_options_v4(
            otel_type,
            transport,
            endpoint,
            headers_json,
            std::ptr::null(),
            resource_attributes_json,
            service_name,
            service_namespace,
            service_version,
            instrumentation_scope,
            timeout_millis,
            mark_projection,
            mark_exclude_names_json,
            attribute_mappings_json,
            promote_metadata_prefixes_json,
            completed_span_context_ttl_millis,
            out,
        )
    }
}

/// Creates one typed OpenTelemetry exporter subscriber with projection, metadata, and lineage controls.
///
/// This compatibility entrypoint preserves the 60-second completed-context retention default.
///
/// # Safety
/// Any non-null C strings must be valid and `out` must be non-null.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_subscriber_create_with_projection_options_v2(
    otel_type: *const c_char,
    transport: *const c_char,
    endpoint: *const c_char,
    headers_json: *const c_char,
    resource_attributes_json: *const c_char,
    service_name: *const c_char,
    service_namespace: *const c_char,
    service_version: *const c_char,
    instrumentation_scope: *const c_char,
    timeout_millis: u64,
    mark_projection: *const c_char,
    mark_exclude_names_json: *const c_char,
    attribute_mappings_json: *const c_char,
    promote_metadata_prefixes_json: *const c_char,
    out: *mut *mut FfiOpenTelemetrySubscriber,
) -> NemoRelayStatus {
    unsafe {
        nemo_relay_otel_subscriber_create_with_projection_options_v3(
            otel_type,
            transport,
            endpoint,
            headers_json,
            resource_attributes_json,
            service_name,
            service_namespace,
            service_version,
            instrumentation_scope,
            timeout_millis,
            mark_projection,
            mark_exclude_names_json,
            attribute_mappings_json,
            promote_metadata_prefixes_json,
            60_000,
            out,
        )
    }
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

/// Return a bounded JSON snapshot of runtime diagnostics for this subscriber.
///
/// The result is a JSON array of `{"code": "…", "message": "…", "count": N}`
/// entries in stable code order. The caller owns `out_json` and must free it
/// with `nemo_relay_string_free`.
///
/// # Safety
/// `subscriber` and `out_json` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_subscriber_runtime_diagnostics_json(
    subscriber: *const FfiOpenTelemetrySubscriber,
    out_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    if let Err(status) = required_out_ptr(out_json) {
        return status;
    }
    unsafe { write_runtime_diagnostics((&*subscriber).0.runtime_diagnostics(), out_json) }
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

fn parse_usize(value: u64, field_name: &str) -> Result<usize, NemoRelayStatus> {
    usize::try_from(value).map_err(|_| {
        set_last_error(&format!("{field_name} exceeds the platform size limit"));
        NemoRelayStatus::InvalidArg
    })
}

#[allow(clippy::too_many_arguments)]
fn build_ffi_otel_log_config(
    transport: *const c_char,
    endpoint: *const c_char,
    headers_json: *const c_char,
    header_env_json: *const c_char,
    resource_attributes_json: *const c_char,
    service_name: *const c_char,
    service_namespace: *const c_char,
    service_version: *const c_char,
    instrumentation_scope: *const c_char,
    timeout_millis: u64,
    minimum_severity: *const c_char,
    max_queue_size: u64,
    max_export_batch_size: u64,
    scheduled_delay_millis: u64,
    completed_span_context_ttl_millis: Option<u64>,
) -> Result<OpenTelemetryLogConfig, NemoRelayStatus> {
    let mut config = OpenTelemetryLogConfig::new(parse_required_otel_endpoint(endpoint)?)
        .with_transport(parse_otlp_transport(transport)?);
    config = apply_optional_string(
        config,
        service_name,
        OpenTelemetryLogConfig::with_service_name,
    )?;
    config = apply_optional_string(
        config,
        instrumentation_scope,
        OpenTelemetryLogConfig::with_instrumentation_scope,
    )?;
    if timeout_millis != 0 {
        config = config.with_timeout(Duration::from_millis(timeout_millis));
    }
    if let Some(value) = parse_optional_string(minimum_severity)? {
        let severity =
            value
                .parse()
                .map_err(|error: nemo_relay::api::event::ParseLogSeverityError| {
                    set_last_error(&error.to_string());
                    NemoRelayStatus::InvalidArg
                })?;
        config = config.with_minimum_severity(severity);
    }
    if max_queue_size != 0 {
        config = config.with_max_queue_size(parse_usize(max_queue_size, "max_queue_size")?);
    }
    if max_export_batch_size != 0 {
        config = config.with_max_export_batch_size(parse_usize(
            max_export_batch_size,
            "max_export_batch_size",
        )?);
    }
    if scheduled_delay_millis != 0 {
        config = config.with_scheduled_delay(Duration::from_millis(scheduled_delay_millis));
    }
    if let Some(completed_span_context_ttl_millis) = completed_span_context_ttl_millis {
        if completed_span_context_ttl_millis == 0 {
            set_last_error("completed_span_context_ttl_millis must be greater than 0");
            return Err(NemoRelayStatus::InvalidArg);
        }
        config = config.with_completed_span_context_ttl(Duration::from_millis(
            completed_span_context_ttl_millis,
        ));
    }
    config = apply_optional_string(
        config,
        service_namespace,
        OpenTelemetryLogConfig::with_service_namespace,
    )?;
    config = apply_optional_string(
        config,
        service_version,
        OpenTelemetryLogConfig::with_service_version,
    )?;
    config = apply_string_map(
        config,
        headers_json,
        "headers",
        OpenTelemetryLogConfig::with_header,
    )?;
    config = apply_string_map(
        config,
        header_env_json,
        "header_env",
        OpenTelemetryLogConfig::with_header_env,
    )?;
    apply_string_map(
        config,
        resource_attributes_json,
        "resource_attributes",
        OpenTelemetryLogConfig::with_resource_attribute,
    )
}

/// Creates an independently managed OpenTelemetry log subscriber.
///
/// Numeric processing settings use their core-config defaults when zero, except
/// `completed_span_context_ttl_millis`, which must be positive.
///
/// # Safety
/// Any non-null C strings must be valid and `out` must be non-null.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nemo_relay_otel_log_subscriber_create(
    transport: *const c_char,
    endpoint: *const c_char,
    headers_json: *const c_char,
    resource_attributes_json: *const c_char,
    service_name: *const c_char,
    service_namespace: *const c_char,
    service_version: *const c_char,
    instrumentation_scope: *const c_char,
    timeout_millis: u64,
    minimum_severity: *const c_char,
    max_queue_size: u64,
    max_export_batch_size: u64,
    scheduled_delay_millis: u64,
    completed_span_context_ttl_millis: u64,
    out: *mut *mut FfiOpenTelemetryLogSubscriber,
) -> NemoRelayStatus {
    unsafe {
        nemo_relay_otel_log_subscriber_create_v2(
            transport,
            endpoint,
            headers_json,
            std::ptr::null(),
            resource_attributes_json,
            service_name,
            service_namespace,
            service_version,
            instrumentation_scope,
            timeout_millis,
            minimum_severity,
            max_queue_size,
            max_export_batch_size,
            scheduled_delay_millis,
            completed_span_context_ttl_millis,
            out,
        )
    }
}

/// Creates an independently managed OpenTelemetry log subscriber with `header_env` support.
///
/// # Safety
/// Any non-null C strings must be valid and `out` must be non-null.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nemo_relay_otel_log_subscriber_create_v2(
    transport: *const c_char,
    endpoint: *const c_char,
    headers_json: *const c_char,
    header_env_json: *const c_char,
    resource_attributes_json: *const c_char,
    service_name: *const c_char,
    service_namespace: *const c_char,
    service_version: *const c_char,
    instrumentation_scope: *const c_char,
    timeout_millis: u64,
    minimum_severity: *const c_char,
    max_queue_size: u64,
    max_export_batch_size: u64,
    scheduled_delay_millis: u64,
    completed_span_context_ttl_millis: u64,
    out: *mut *mut FfiOpenTelemetryLogSubscriber,
) -> NemoRelayStatus {
    clear_last_error();
    if let Err(status) = required_out_ptr(out) {
        return status;
    }
    let config = match build_ffi_otel_log_config(
        transport,
        endpoint,
        headers_json,
        header_env_json,
        resource_attributes_json,
        service_name,
        service_namespace,
        service_version,
        instrumentation_scope,
        timeout_millis,
        minimum_severity,
        max_queue_size,
        max_export_batch_size,
        scheduled_delay_millis,
        Some(completed_span_context_ttl_millis),
    ) {
        Ok(config) => config,
        Err(status) => return status,
    };
    let _runtime_guard = tokio_runtime().enter();
    match OpenTelemetryLogSubscriber::new(config) {
        Ok(subscriber) => {
            unsafe { *out = Box::into_raw(Box::new(FfiOpenTelemetryLogSubscriber(subscriber))) };
            NemoRelayStatus::Ok
        }
        Err(error) => {
            set_last_error(&error.to_string());
            NemoRelayStatus::Internal
        }
    }
}

/// Registers an OpenTelemetry log subscriber globally.
///
/// # Safety
/// `subscriber` and `name` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_log_subscriber_register(
    subscriber: *const FfiOpenTelemetryLogSubscriber,
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    match unsafe { &*subscriber }.0.register(&name) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => {
            set_last_error(&error.to_string());
            NemoRelayStatus::Internal
        }
    }
}

/// Deregisters a subscriber by name from the shared subscriber registry.
///
/// Subscriber names share one global namespace across trace, log, and metric
/// subscribers. This function does not verify the subscriber type.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_log_subscriber_deregister(
    name: *const c_char,
) -> NemoRelayStatus {
    unsafe { nemo_relay_otel_subscriber_deregister(name) }
}

/// Flushes queued Relay events and OTLP log batches.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_log_subscriber_force_flush(
    subscriber: *const FfiOpenTelemetryLogSubscriber,
) -> NemoRelayStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    match unsafe { &*subscriber }.0.force_flush() {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => {
            set_last_error(&error.to_string());
            NemoRelayStatus::Internal
        }
    }
}

/// Return a bounded JSON snapshot of runtime diagnostics for this subscriber.
///
/// The result is a JSON array of `{"code": "…", "message": "…", "count": N}`
/// entries in stable code order. The caller owns `out_json` and must free it
/// with `nemo_relay_string_free`.
///
/// # Safety
/// `subscriber` and `out_json` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_log_subscriber_runtime_diagnostics_json(
    subscriber: *const FfiOpenTelemetryLogSubscriber,
    out_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    if let Err(status) = required_out_ptr(out_json) {
        return status;
    }
    unsafe { write_runtime_diagnostics((&*subscriber).0.runtime_diagnostics(), out_json) }
}

/// Shuts down the OpenTelemetry logger provider.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_log_subscriber_shutdown(
    subscriber: *const FfiOpenTelemetryLogSubscriber,
) -> NemoRelayStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    match unsafe { &*subscriber }.0.shutdown() {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => {
            set_last_error(&error.to_string());
            NemoRelayStatus::Internal
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_ffi_otel_metric_config(
    transport: *const c_char,
    endpoint: *const c_char,
    headers_json: *const c_char,
    header_env_json: *const c_char,
    resource_attributes_json: *const c_char,
    service_name: *const c_char,
    service_namespace: *const c_char,
    service_version: *const c_char,
    instrumentation_scope: *const c_char,
    timeout_millis: u64,
    export_interval_millis: u64,
    temporality: *const c_char,
    max_instruments: u64,
    cardinality_limit: u64,
) -> Result<OpenTelemetryMetricConfig, NemoRelayStatus> {
    let mut config = OpenTelemetryMetricConfig::new(parse_required_otel_endpoint(endpoint)?)
        .with_transport(parse_otlp_transport(transport)?);
    config = apply_optional_string(
        config,
        service_name,
        OpenTelemetryMetricConfig::with_service_name,
    )?;
    config = apply_optional_string(
        config,
        instrumentation_scope,
        OpenTelemetryMetricConfig::with_instrumentation_scope,
    )?;
    if timeout_millis != 0 {
        config = config.with_timeout(Duration::from_millis(timeout_millis));
    }
    if export_interval_millis != 0 {
        config = config.with_export_interval(Duration::from_millis(export_interval_millis));
    }
    if let Some(value) = parse_optional_string(temporality)? {
        let temporality = value.parse().map_err(|error: String| {
            set_last_error(&error);
            NemoRelayStatus::InvalidArg
        })?;
        config = config.with_temporality(temporality);
    }
    if max_instruments != 0 {
        config = config.with_max_instruments(parse_usize(max_instruments, "max_instruments")?);
    }
    if cardinality_limit != 0 {
        config =
            config.with_cardinality_limit(parse_usize(cardinality_limit, "cardinality_limit")?);
    }
    config = apply_optional_string(
        config,
        service_namespace,
        OpenTelemetryMetricConfig::with_service_namespace,
    )?;
    config = apply_optional_string(
        config,
        service_version,
        OpenTelemetryMetricConfig::with_service_version,
    )?;
    config = apply_string_map(
        config,
        headers_json,
        "headers",
        OpenTelemetryMetricConfig::with_header,
    )?;
    config = apply_string_map(
        config,
        header_env_json,
        "header_env",
        OpenTelemetryMetricConfig::with_header_env,
    )?;
    apply_string_map(
        config,
        resource_attributes_json,
        "resource_attributes",
        OpenTelemetryMetricConfig::with_resource_attribute,
    )
}

/// Creates an independently managed OpenTelemetry metric subscriber.
///
/// Numeric processing settings use their core-config defaults when zero.
///
/// # Safety
/// Any non-null C strings must be valid and `out` must be non-null.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nemo_relay_otel_metric_subscriber_create(
    transport: *const c_char,
    endpoint: *const c_char,
    headers_json: *const c_char,
    resource_attributes_json: *const c_char,
    service_name: *const c_char,
    service_namespace: *const c_char,
    service_version: *const c_char,
    instrumentation_scope: *const c_char,
    timeout_millis: u64,
    export_interval_millis: u64,
    temporality: *const c_char,
    max_instruments: u64,
    cardinality_limit: u64,
    out: *mut *mut FfiOpenTelemetryMetricSubscriber,
) -> NemoRelayStatus {
    unsafe {
        nemo_relay_otel_metric_subscriber_create_v2(
            transport,
            endpoint,
            headers_json,
            std::ptr::null(),
            resource_attributes_json,
            service_name,
            service_namespace,
            service_version,
            instrumentation_scope,
            timeout_millis,
            export_interval_millis,
            temporality,
            max_instruments,
            cardinality_limit,
            out,
        )
    }
}

/// Creates an independently managed OpenTelemetry metric subscriber with `header_env` support.
///
/// # Safety
/// Any non-null C strings must be valid and `out` must be non-null.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nemo_relay_otel_metric_subscriber_create_v2(
    transport: *const c_char,
    endpoint: *const c_char,
    headers_json: *const c_char,
    header_env_json: *const c_char,
    resource_attributes_json: *const c_char,
    service_name: *const c_char,
    service_namespace: *const c_char,
    service_version: *const c_char,
    instrumentation_scope: *const c_char,
    timeout_millis: u64,
    export_interval_millis: u64,
    temporality: *const c_char,
    max_instruments: u64,
    cardinality_limit: u64,
    out: *mut *mut FfiOpenTelemetryMetricSubscriber,
) -> NemoRelayStatus {
    clear_last_error();
    if let Err(status) = required_out_ptr(out) {
        return status;
    }
    let config = match build_ffi_otel_metric_config(
        transport,
        endpoint,
        headers_json,
        header_env_json,
        resource_attributes_json,
        service_name,
        service_namespace,
        service_version,
        instrumentation_scope,
        timeout_millis,
        export_interval_millis,
        temporality,
        max_instruments,
        cardinality_limit,
    ) {
        Ok(config) => config,
        Err(status) => return status,
    };
    let _runtime_guard = tokio_runtime().enter();
    match OpenTelemetryMetricSubscriber::new(config) {
        Ok(subscriber) => {
            unsafe { *out = Box::into_raw(Box::new(FfiOpenTelemetryMetricSubscriber(subscriber))) };
            NemoRelayStatus::Ok
        }
        Err(error) => {
            set_last_error(&error.to_string());
            NemoRelayStatus::Internal
        }
    }
}

/// Registers an OpenTelemetry metric subscriber globally.
///
/// # Safety
/// `subscriber` and `name` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_metric_subscriber_register(
    subscriber: *const FfiOpenTelemetryMetricSubscriber,
    name: *const c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    match unsafe { &*subscriber }.0.register(&name) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => {
            set_last_error(&error.to_string());
            NemoRelayStatus::Internal
        }
    }
}

/// Deregisters a subscriber by name from the shared subscriber registry.
///
/// Subscriber names share one global namespace across trace, log, and metric
/// subscribers. This function does not verify the subscriber type.
///
/// # Safety
/// `name` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_metric_subscriber_deregister(
    name: *const c_char,
) -> NemoRelayStatus {
    unsafe { nemo_relay_otel_subscriber_deregister(name) }
}

/// Flushes queued Relay events and collects current metric aggregates.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_metric_subscriber_force_flush(
    subscriber: *const FfiOpenTelemetryMetricSubscriber,
) -> NemoRelayStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    match unsafe { &*subscriber }.0.force_flush() {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => {
            set_last_error(&error.to_string());
            NemoRelayStatus::Internal
        }
    }
}

/// Return a bounded JSON snapshot of runtime diagnostics for this subscriber.
///
/// The result is a JSON array of `{"code": "…", "message": "…", "count": N}`
/// entries in stable code order. The caller owns `out_json` and must free it
/// with `nemo_relay_string_free`.
///
/// # Safety
/// `subscriber` and `out_json` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_metric_subscriber_runtime_diagnostics_json(
    subscriber: *const FfiOpenTelemetryMetricSubscriber,
    out_json: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    if let Err(status) = required_out_ptr(out_json) {
        return status;
    }
    unsafe { write_runtime_diagnostics((&*subscriber).0.runtime_diagnostics(), out_json) }
}

/// Shuts down the OpenTelemetry meter provider and performs final collection.
///
/// # Safety
/// `subscriber` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_metric_subscriber_shutdown(
    subscriber: *const FfiOpenTelemetryMetricSubscriber,
) -> NemoRelayStatus {
    clear_last_error();
    if subscriber.is_null() {
        set_last_error("subscriber pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    match unsafe { &*subscriber }.0.shutdown() {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => {
            set_last_error(&error.to_string());
            NemoRelayStatus::Internal
        }
    }
}
