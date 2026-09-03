// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration coverage for the exported scope-stack entrypoints.

use super::*;
use std::ptr;

#[test]
#[allow(clippy::cognitive_complexity)] // Covers the exported scope-stack contracts in one causal flow.
fn scope_stack_propagation_entrypoints_round_trip_through_the_ffi() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|error| error.into_inner());

    unsafe {
        let mut context_json = ptr::null_mut();
        assert_eq!(
            nemo_relay_capture_propagation_context_json(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_capture_propagation_context_json(&mut context_json),
            NemoRelayStatus::Ok
        );
        let context_text = CStr::from_ptr(context_json).to_str().unwrap().to_owned();
        nemo_relay_string_free(context_json);

        let root_uuid = CString::new("018f13f0-7c1a-7a80-8000-000000000701").unwrap();
        assert_eq!(
            nemo_relay_capture_propagation_context_with_root_json(
                root_uuid.as_ptr(),
                ptr::null_mut()
            ),
            NemoRelayStatus::NullPointer
        );
        let invalid_root = CString::new("invalid-uuid").unwrap();
        assert_eq!(
            nemo_relay_capture_propagation_context_with_root_json(
                invalid_root.as_ptr(),
                &mut context_json
            ),
            NemoRelayStatus::InvalidArg
        );
        assert_eq!(
            nemo_relay_capture_propagation_context_with_root_json(
                root_uuid.as_ptr(),
                &mut context_json,
            ),
            NemoRelayStatus::Ok
        );
        let rooted_text = CStr::from_ptr(context_json).to_str().unwrap();
        assert!(rooted_text.contains("018f13f0-7c1a-7a80-8000-000000000701"));
        nemo_relay_string_free(context_json);

        let context = CString::new(context_text).unwrap();
        let mut stack = ptr::null_mut();
        assert_eq!(
            nemo_relay_scope_stack_create_from_propagation_json(context.as_ptr(), ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        let invalid_json = CString::new("{").unwrap();
        assert_eq!(
            nemo_relay_scope_stack_create_from_propagation_json(invalid_json.as_ptr(), &mut stack),
            NemoRelayStatus::InvalidJson
        );
        assert_eq!(
            nemo_relay_scope_stack_create_from_propagation_json(context.as_ptr(), &mut stack),
            NemoRelayStatus::Ok
        );
        assert!(!stack.is_null());
        assert_eq!(
            nemo_relay_scope_stack_set_thread(ptr::null()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_scope_stack_set_thread(stack),
            NemoRelayStatus::Ok
        );
        assert!(nemo_relay_scope_stack_active());

        let mut captured_traceparent = ptr::null_mut();
        assert_eq!(
            nemo_relay_capture_traceparent(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_capture_traceparent(&mut captured_traceparent),
            NemoRelayStatus::Ok
        );
        assert!(!captured_traceparent.is_null());
        nemo_relay_string_free(captured_traceparent);

        let rooted_context = CString::new(
            r#"{"version":1,"root_uuid":"018f13f0-7c1a-7a80-8000-000000000701","parent_uuid":"018f13f0-7c1a-7a80-8000-000000000702"}"#,
        )
        .unwrap();
        let mut traceparent = ptr::null_mut();
        assert_eq!(
            nemo_relay_propagation_context_to_traceparent(rooted_context.as_ptr(), ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_propagation_context_to_traceparent(ptr::null(), &mut traceparent),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_propagation_context_to_traceparent(
                rooted_context.as_ptr(),
                &mut traceparent,
            ),
            NemoRelayStatus::Ok
        );
        assert!(
            CStr::from_ptr(traceparent)
                .to_str()
                .unwrap()
                .starts_with("00-")
        );
        nemo_relay_string_free(traceparent);

        let mut binding = ptr::null_mut();
        assert_eq!(
            nemo_relay_scope_stack_capture_thread(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_scope_stack_restore_thread(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_scope_stack_capture_thread(&mut binding),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            nemo_relay_scope_stack_restore_thread(binding),
            NemoRelayStatus::Ok
        );
        nemo_relay_scope_stack_free(stack);
    }
}
