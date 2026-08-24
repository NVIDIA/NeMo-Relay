// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    FfiScopeHandle, NemoRelayLogSeverity, NemoRelayMetricMeasurement, NemoRelayScopeType,
    NemoRelayStatus, ScopeAttributes, c_char, c_str_to_opt_json, c_str_to_string, clear_last_error,
    core_scope_api, set_last_error, status_from_error, unix_micros_to_opt_timestamp,
};
use crate::types::{NEMO_RELAY_METRIC_KIND_UNSPECIFIED, NEMO_RELAY_METRIC_VALUE_TYPE_UNSPECIFIED};
use crate::types::{log_severity_from_ffi, metric_kind_from_ffi, metric_value_type_from_ffi};

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

fn set_metric_discriminator_error(
    index: usize,
    field: &str,
    value: i32,
    unspecified: i32,
    expected: &str,
) {
    if value == unspecified {
        set_last_error(&format!(
            "measurements[{index}].{field} is unspecified; set one of {expected}"
        ));
    } else {
        set_last_error(&format!(
            "measurements[{index}].{field} has invalid value {value}"
        ));
    }
}

/// Retrieve the current scope handle from the thread-local scope stack.
///
/// # Parameters
/// - `out`: On success, receives a heap-allocated `FfiScopeHandle` that must be
///   freed with `nemo_relay_scope_handle_free`.
///
/// # Safety
/// `out` must be a valid, non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_get_handle(out: *mut *mut FfiScopeHandle) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    match core_scope_api::get_handle() {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiScopeHandle(h))) };
            NemoRelayStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Push a new scope onto the scope stack.
///
/// This creates a scope handle, emits a scope Start event, and makes the new
/// scope the current top of the active stack.
///
/// # Parameters
/// - `name`: Null-terminated scope name.
/// - `scope_type`: The type of scope to create.
/// - `parent`: Optional parent scope handle, or null to use the current top of
///   stack.
/// - `attributes`: Bitfield of scope attributes.
/// - `data_json`: Optional null-terminated JSON string stored on the scope
///   handle, or null.
/// - `metadata_json`: Optional null-terminated JSON metadata string recorded
///   on the start event, or null.
/// - `input_json`: Optional null-terminated JSON string exported as the
///   semantic scope input on the start event, or null.
/// - `timestamp_unix_micros`: Optional Unix microseconds timestamp for the
///   handle start time and start event, or null to use the current UTC time.
/// - `out`: On success, receives a heap-allocated `FfiScopeHandle` that must
///   be freed with `nemo_relay_scope_handle_free`.
///
/// # Errors
/// Returns `InvalidJson` for invalid JSON inputs and `InvalidArg` when
/// `timestamp_unix_micros` is outside the supported timestamp range.
///
/// # Safety
/// `name` must be a valid C string. `out` must be non-null. `parent`,
/// `data_json`, `metadata_json`, `input_json`, and `timestamp_unix_micros` may
/// be null; when non-null, optional pointers must be valid for reads for the
/// duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_push_scope(
    name: *const c_char,
    scope_type: NemoRelayScopeType,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    input_json: *const c_char,
    timestamp_unix_micros: *const i64,
    out: *mut *mut FfiScopeHandle,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = ScopeAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoRelayStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoRelayStatus::InvalidJson,
    };
    let input = match c_str_to_opt_json(input_json) {
        Some(v) => v,
        None => return NemoRelayStatus::InvalidJson,
    };
    let timestamp = match unix_micros_to_opt_timestamp(timestamp_unix_micros) {
        Some(v) => v,
        None => return NemoRelayStatus::InvalidArg,
    };

    match core_scope_api::push_scope(
        core_scope_api::PushScopeParams::builder()
            .name(name.as_str())
            .scope_type(scope_type.into())
            .parent_opt(parent_ref)
            .attributes(attrs)
            .data_opt(data)
            .metadata_opt(metadata)
            .input_opt(input)
            .timestamp_opt(timestamp)
            .build(),
    ) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiScopeHandle(h))) };
            NemoRelayStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Pop a scope from the scope stack by its handle.
///
/// This emits a scope End event and removes scope-local registrations owned by
/// the popped scope.
///
/// # Parameters
/// - `handle`: The current top-of-stack scope handle to pop.
/// - `output_json`: Optional null-terminated JSON string exported as semantic
///   scope output on the end event, or null.
/// - `metadata_json`: Optional null-terminated JSON metadata string recorded
///   on the end event, or null. Incoming metadata is merged over metadata
///   stored on the scope handle.
/// - `timestamp_unix_micros`: Optional Unix microseconds timestamp for the end
///   event, or null to use the runtime default end timestamp.
///
/// # Errors
/// Returns `InvalidJson` for invalid output or metadata JSON, `InvalidArg` when
/// `timestamp_unix_micros` is outside the supported timestamp range, or an
/// error status when `handle` is not the current top scope.
///
/// # Safety
/// `handle` must be a valid, non-null `FfiScopeHandle` pointer. Optional
/// pointer arguments may be null; when non-null, they must be valid for reads
/// for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_pop_scope(
    handle: *const FfiScopeHandle,
    output_json: *const c_char,
    metadata_json: *const c_char,
    timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NemoRelayStatus::NullPointer;
    }
    let output = match c_str_to_opt_json(output_json) {
        Some(v) => v,
        None => return NemoRelayStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(v) => v,
        None => return NemoRelayStatus::InvalidJson,
    };
    let timestamp = match unix_micros_to_opt_timestamp(timestamp_unix_micros) {
        Some(v) => v,
        None => return NemoRelayStatus::InvalidArg,
    };
    match core_scope_api::pop_scope(
        core_scope_api::PopScopeParams::builder()
            .handle_uuid(&unsafe { &*handle }.0.uuid)
            .output_opt(output)
            .metadata_opt(metadata)
            .timestamp_opt(timestamp)
            .build(),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Emit a named lifecycle event.
///
/// This creates a point-in-time Mark event without pushing or popping a scope.
///
/// # Parameters
/// - `name`: Null-terminated event name.
/// - `parent`: Optional parent scope handle, or null to use the current top of
///   stack.
/// - `data_json`: Optional null-terminated JSON data payload recorded on the
///   mark event, or null.
/// - `metadata_json`: Optional null-terminated JSON metadata payload recorded
///   on the mark event, or null.
/// - `timestamp_unix_micros`: Optional Unix microseconds timestamp for the
///   mark event, or null to use the current UTC time.
///
/// # Errors
/// Returns `InvalidJson` for invalid JSON inputs and `InvalidArg` when
/// `timestamp_unix_micros` is outside the supported timestamp range.
///
/// # Safety
/// `name` must be a valid C string. Other pointer args may be null; when
/// non-null, optional pointers must be valid for reads for the duration of the
/// call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event(
    name: *const c_char,
    parent: *const FfiScopeHandle,
    data_json: *const c_char,
    metadata_json: *const c_char,
    timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
    unsafe {
        nemo_relay_event_v2(
            name,
            parent,
            data_json,
            std::ptr::null(),
            metadata_json,
            std::ptr::null(),
            timestamp_unix_micros,
        )
    }
}

/// Emit a named lifecycle event with optional data schema and log severity.
///
/// This is the additive form of [`nemo_relay_event`]. The legacy function
/// remains ABI-compatible and behaves as if both new arguments were null.
///
/// # Parameters
/// - `data_schema_json`: Optional `{"name":"...","version":"..."}` JSON.
/// - `severity`: Optional typed telemetry-log severity.
///
/// # Safety
/// The pointer requirements are the same as [`nemo_relay_event`]. Optional
/// pointers may be null and otherwise must remain valid for the call.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nemo_relay_event_v2(
    name: *const c_char,
    parent: *const FfiScopeHandle,
    data_json: *const c_char,
    data_schema_json: *const c_char,
    metadata_json: *const c_char,
    severity: *const NemoRelayLogSeverity,
    timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoRelayStatus::InvalidJson,
    };
    let data_schema = match c_str_to_opt_json(data_schema_json) {
        Some(Some(value)) => match serde_json::from_value(value) {
            Ok(schema) => Some(schema),
            Err(error) => {
                set_last_error(&format!("invalid data_schema JSON: {error}"));
                return NemoRelayStatus::InvalidJson;
            }
        },
        Some(None) => None,
        None => return NemoRelayStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoRelayStatus::InvalidJson,
    };
    let severity = if severity.is_null() {
        None
    } else {
        let raw_severity = unsafe { *severity };
        let Some(severity) = log_severity_from_ffi(raw_severity) else {
            set_last_error(&format!("severity has invalid value {raw_severity}"));
            return NemoRelayStatus::InvalidArg;
        };
        Some(severity)
    };
    let timestamp = match unix_micros_to_opt_timestamp(timestamp_unix_micros) {
        Some(v) => v,
        None => return NemoRelayStatus::InvalidArg,
    };

    match core_scope_api::event(
        core_scope_api::EmitMarkEventParams::builder()
            .name(&name)
            .parent_opt(parent_ref)
            .data_opt(data)
            .data_schema_opt(data_schema)
            .metadata_opt(metadata)
            .severity_opt(severity)
            .timestamp_opt(timestamp)
            .build(),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Emit an atomic metric measurement mark from canonical JSON.
///
/// `measurements_json` must be a nonempty JSON array using Relay's canonical
/// `MetricMeasurement` shape. Relay validates the complete array before
/// emitting the metric mark.
///
/// # Safety
/// `name` and `measurements_json` must be valid non-null C strings. Optional
/// pointers may be null and otherwise must remain valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_metric_json(
    name: *const c_char,
    parent: *const FfiScopeHandle,
    measurements_json: *const c_char,
    metadata_json: *const c_char,
    timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let Some(measurements_json) = c_str_to_opt_json(measurements_json) else {
        return NemoRelayStatus::InvalidJson;
    };
    let Some(measurements_json) = measurements_json else {
        set_last_error("measurements_json is required");
        return NemoRelayStatus::NullPointer;
    };
    let measurements = match serde_json::from_value(measurements_json) {
        Ok(measurements) => measurements,
        Err(error) => {
            set_last_error(&format!("invalid metric measurements JSON: {error}"));
            return NemoRelayStatus::InvalidJson;
        }
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(metadata) => metadata,
        None => return NemoRelayStatus::InvalidJson,
    };
    let timestamp = match unix_micros_to_opt_timestamp(timestamp_unix_micros) {
        Some(timestamp) => timestamp,
        None => return NemoRelayStatus::InvalidArg,
    };

    match core_scope_api::metric(
        core_scope_api::EmitMetricEventParams::builder()
            .name(&name)
            .measurements(measurements)
            .parent_opt(parent_ref)
            .metadata_opt(metadata)
            .timestamp_opt(timestamp)
            .build(),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => status_from_error(&error),
    }
}

/// Emit an atomic metric measurement mark from typed C measurements.
///
/// `measurements` must reference `measurements_len` initialized entries. Each
/// measurement's `value_type` selects the corresponding numeric value field.
/// Relay validates the complete array before emitting any recording operation.
/// Relay does not retain `measurements`, their strings, or histogram boundaries;
/// caller-owned storage need only remain valid until this function returns.
///
/// # Safety
/// `name` must be a valid non-null C string. `measurements` must be non-null
/// and valid for `measurements_len` reads when the length is nonzero. Every
/// measurement name must be a valid C string; optional strings and JSON must
/// be valid when non-null. A null `boundaries` pointer with zero length leaves
/// boundaries unspecified, while a non-null pointer with zero length requests
/// an explicit empty boundary list. For nonzero lengths, `boundaries` must
/// reference `boundaries_len` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_metric(
    name: *const c_char,
    parent: *const FfiScopeHandle,
    measurements: *const NemoRelayMetricMeasurement,
    measurements_len: usize,
    metadata_json: *const c_char,
    timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(name) => name,
        Err(status) => return status,
    };
    if measurements_len > 0 && measurements.is_null() {
        set_last_error("measurements pointer is null");
        return NemoRelayStatus::NullPointer;
    }
    let raw_measurements = if measurements_len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(measurements, measurements_len) }
    };
    let typed_measurements = raw_measurements
        .iter()
        .enumerate()
        .map(|(index, measurement)| parse_metric_measurement(index, measurement))
        .collect::<Result<Vec<_>, _>>();
    let typed_measurements = match typed_measurements {
        Ok(measurements) => measurements,
        Err(status) => return status,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(metadata) => metadata,
        None => return NemoRelayStatus::InvalidJson,
    };
    let timestamp = match unix_micros_to_opt_timestamp(timestamp_unix_micros) {
        Some(timestamp) => timestamp,
        None => return NemoRelayStatus::InvalidArg,
    };
    match core_scope_api::metric(
        core_scope_api::EmitMetricEventParams::builder()
            .name(&name)
            .measurements(typed_measurements)
            .parent_opt(parent_ref)
            .metadata_opt(metadata)
            .timestamp_opt(timestamp)
            .build(),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => status_from_error(&error),
    }
}

fn parse_metric_measurement(
    index: usize,
    measurement: &NemoRelayMetricMeasurement,
) -> Result<nemo_relay::api::event::MetricMeasurement, NemoRelayStatus> {
    let measurement_name = match c_str_to_string(measurement.name) {
        Ok(name) => name,
        Err(status) => {
            set_last_error(&format!("measurements[{index}].name is invalid"));
            return Err(status);
        }
    };
    let Some(kind) = metric_kind_from_ffi(measurement.kind) else {
        set_metric_discriminator_error(
            index,
            "kind",
            measurement.kind,
            NEMO_RELAY_METRIC_KIND_UNSPECIFIED,
            "NEMO_RELAY_METRIC_KIND_COUNTER, NEMO_RELAY_METRIC_KIND_UP_DOWN_COUNTER, NEMO_RELAY_METRIC_KIND_GAUGE, or NEMO_RELAY_METRIC_KIND_HISTOGRAM",
        );
        return Err(NemoRelayStatus::InvalidArg);
    };
    let Some(value_type) = metric_value_type_from_ffi(measurement.value_type) else {
        set_metric_discriminator_error(
            index,
            "value_type",
            measurement.value_type,
            NEMO_RELAY_METRIC_VALUE_TYPE_UNSPECIFIED,
            "NEMO_RELAY_METRIC_VALUE_TYPE_U64, NEMO_RELAY_METRIC_VALUE_TYPE_I64, or NEMO_RELAY_METRIC_VALUE_TYPE_F64",
        );
        return Err(NemoRelayStatus::InvalidArg);
    };
    let value = match value_type {
        nemo_relay::api::event::MetricValueType::U64 => {
            serde_json::Value::from(measurement.u64_value)
        }
        nemo_relay::api::event::MetricValueType::I64 => {
            serde_json::Value::from(measurement.i64_value)
        }
        nemo_relay::api::event::MetricValueType::F64 => {
            let Some(number) = serde_json::Number::from_f64(measurement.f64_value) else {
                set_last_error(&format!("measurements[{index}].f64_value must be finite"));
                return Err(NemoRelayStatus::InvalidArg);
            };
            serde_json::Value::Number(number)
        }
    };
    let unit = if measurement.unit.is_null() {
        None
    } else {
        match c_str_to_string(measurement.unit) {
            Ok(value) => Some(value),
            Err(status) => return Err(status),
        }
    };
    let description = if measurement.description.is_null() {
        None
    } else {
        match c_str_to_string(measurement.description) {
            Ok(value) => Some(value),
            Err(status) => return Err(status),
        }
    };
    let attributes = match c_str_to_opt_json(measurement.attributes_json) {
        Some(value) => value,
        None => return Err(NemoRelayStatus::InvalidJson),
    };
    if measurement.boundaries_len > 0 && measurement.boundaries.is_null() {
        set_last_error(&format!("measurements[{index}].boundaries pointer is null"));
        return Err(NemoRelayStatus::NullPointer);
    }
    let boundaries = if measurement.boundaries.is_null() {
        None
    } else if measurement.boundaries_len == 0 {
        Some(Vec::new())
    } else {
        Some(
            unsafe {
                std::slice::from_raw_parts(measurement.boundaries, measurement.boundaries_len)
            }
            .to_vec(),
        )
    };
    Ok(nemo_relay::api::event::MetricMeasurement {
        name: measurement_name,
        kind,
        value_type,
        value,
        unit,
        description,
        attributes,
        boundaries,
    })
}
