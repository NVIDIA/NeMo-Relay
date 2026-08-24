// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared fixtures and lifecycle assertions for FFI coverage tests.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use super::{NemoRelayStatus, types};

pub(crate) struct ValidationFixtures {
    pub(crate) scope_name: CString,
    pub(crate) event_name: CString,
    pub(crate) metric_name: CString,
    pub(crate) instrument_name: CString,
    pub(crate) unit: CString,
    pub(crate) description: CString,
    pub(crate) attributes: CString,
    pub(crate) metadata: CString,
    pub(crate) malformed_json: CString,
    pub(crate) invalid_schema: CString,
    pub(crate) measurements: CString,
    pub(crate) invalid_measurements_shape: CString,
    invalid_utf8: [u8; 2],
    pub(crate) invalid_timestamp: i64,
    pub(crate) boundaries: [f64; 3],
}

impl ValidationFixtures {
    pub(crate) fn new(label: &str) -> Self {
        Self {
            scope_name: CString::new(format!("ffi_{label}_validation_scope")).unwrap(),
            event_name: CString::new(format!("ffi_{label}_validation_event")).unwrap(),
            metric_name: CString::new(format!("ffi_{label}_validation_metric")).unwrap(),
            instrument_name: CString::new("example.validation.value").unwrap(),
            unit: CString::new("{item}").unwrap(),
            description: CString::new("validation sweep").unwrap(),
            attributes: CString::new(r#"{"source":"ffi"}"#).unwrap(),
            metadata: CString::new(r#"{"test":true}"#).unwrap(),
            malformed_json: CString::new("{").unwrap(),
            invalid_schema: CString::new(r#"{"name":1,"version":false}"#).unwrap(),
            measurements: CString::new(
                r#"[{"name":"example.validation.value","kind":"counter","value_type":"u64","value":1}]"#,
            )
            .unwrap(),
            invalid_measurements_shape: CString::new(r#"{"name":"not-an-array"}"#).unwrap(),
            invalid_utf8: [0xff, 0],
            invalid_timestamp: i64::MAX,
            boundaries: [0.0, 1.0, 10.0],
        }
    }

    pub(crate) fn invalid_utf8(&self) -> *const c_char {
        self.invalid_utf8.as_ptr().cast()
    }

    pub(crate) fn push_scope_json_cases(
        &self,
    ) -> [(*const c_char, *const c_char, *const c_char); 3] {
        [
            (self.malformed_json.as_ptr(), ptr::null(), ptr::null()),
            (ptr::null(), self.malformed_json.as_ptr(), ptr::null()),
            (ptr::null(), ptr::null(), self.malformed_json.as_ptr()),
        ]
    }

    pub(crate) fn pop_scope_cases(
        &self,
    ) -> [(*const c_char, *const c_char, *const i64, NemoRelayStatus); 3] {
        [
            (
                self.malformed_json.as_ptr(),
                ptr::null(),
                ptr::null(),
                NemoRelayStatus::InvalidJson,
            ),
            (
                ptr::null(),
                self.malformed_json.as_ptr(),
                ptr::null(),
                NemoRelayStatus::InvalidJson,
            ),
            (
                ptr::null(),
                ptr::null(),
                ptr::from_ref(&self.invalid_timestamp),
                NemoRelayStatus::InvalidArg,
            ),
        ]
    }

    pub(crate) fn event_json_cases(&self) -> [(*const c_char, *const c_char, *const c_char); 4] {
        [
            (self.malformed_json.as_ptr(), ptr::null(), ptr::null()),
            (ptr::null(), self.malformed_json.as_ptr(), ptr::null()),
            (ptr::null(), self.invalid_schema.as_ptr(), ptr::null()),
            (ptr::null(), ptr::null(), self.malformed_json.as_ptr()),
        ]
    }

    pub(crate) fn metric_json_cases(
        &self,
    ) -> [(
        *const c_char,
        *const c_char,
        *const c_char,
        *const i64,
        NemoRelayStatus,
    ); 6] {
        [
            (
                self.invalid_utf8(),
                self.measurements.as_ptr(),
                ptr::null(),
                ptr::null(),
                NemoRelayStatus::InvalidUtf8,
            ),
            (
                self.metric_name.as_ptr(),
                self.malformed_json.as_ptr(),
                ptr::null(),
                ptr::null(),
                NemoRelayStatus::InvalidJson,
            ),
            (
                self.metric_name.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                NemoRelayStatus::NullPointer,
            ),
            (
                self.metric_name.as_ptr(),
                self.invalid_measurements_shape.as_ptr(),
                ptr::null(),
                ptr::null(),
                NemoRelayStatus::InvalidJson,
            ),
            (
                self.metric_name.as_ptr(),
                self.measurements.as_ptr(),
                self.malformed_json.as_ptr(),
                ptr::null(),
                NemoRelayStatus::InvalidJson,
            ),
            (
                self.metric_name.as_ptr(),
                self.measurements.as_ptr(),
                ptr::null(),
                ptr::from_ref(&self.invalid_timestamp),
                NemoRelayStatus::InvalidArg,
            ),
        ]
    }

    pub(crate) fn base_measurement(&self) -> types::NemoRelayMetricMeasurement {
        types::NemoRelayMetricMeasurement {
            name: self.instrument_name.as_ptr(),
            kind: types::NEMO_RELAY_METRIC_KIND_COUNTER,
            value_type: types::NEMO_RELAY_METRIC_VALUE_TYPE_I64,
            u64_value: 0,
            i64_value: -7,
            f64_value: 0.0,
            unit: self.unit.as_ptr(),
            description: self.description.as_ptr(),
            attributes_json: self.attributes.as_ptr(),
            boundaries: ptr::null(),
            boundaries_len: 0,
        }
    }

    pub(crate) fn native_metric_status_cases(
        &self,
    ) -> [(*const c_char, *const i64, NemoRelayStatus); 2] {
        [
            (
                self.malformed_json.as_ptr(),
                ptr::null(),
                NemoRelayStatus::InvalidJson,
            ),
            (
                ptr::null(),
                ptr::from_ref(&self.invalid_timestamp),
                NemoRelayStatus::InvalidArg,
            ),
        ]
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assert_otel_signal_lifecycle<H>(
    name: &CStr,
    create: impl FnOnce(*mut *mut H) -> NemoRelayStatus,
    after_create: impl FnOnce(*mut H),
    register: impl Fn(*const H, *const c_char) -> NemoRelayStatus,
    deregister: impl Fn(*const c_char) -> NemoRelayStatus,
    shutdown: impl Fn(*const H) -> NemoRelayStatus,
    force_flush: impl Fn(*const H) -> NemoRelayStatus,
    free: impl FnOnce(*mut H),
) {
    let mut subscriber = ptr::null_mut();
    assert_eq!(create(&mut subscriber), NemoRelayStatus::Ok);
    assert!(!subscriber.is_null());
    after_create(subscriber);
    assert_eq!(register(subscriber, name.as_ptr()), NemoRelayStatus::Ok);
    assert_eq!(
        register(subscriber, name.as_ptr()),
        NemoRelayStatus::Internal
    );
    assert_eq!(deregister(name.as_ptr()), NemoRelayStatus::Ok);
    assert_eq!(shutdown(subscriber), NemoRelayStatus::Ok);
    assert_eq!(force_flush(subscriber), NemoRelayStatus::Internal);
    assert_eq!(shutdown(subscriber), NemoRelayStatus::Ok);
    free(subscriber);
}
