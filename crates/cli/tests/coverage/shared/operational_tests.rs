// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::{HeaderMap, HeaderValue};
use uuid::Uuid;

use crate::operational::{OPERATION_ID_HEADER, OperationalContext};

#[test]
fn forwarded_hook_context_keeps_only_the_opaque_operation_id() {
    let source = OperationalContext::new();
    let mut headers = HeaderMap::new();
    source.attach_header(&mut headers);

    let received = OperationalContext::take_from_headers(&mut headers);

    assert_eq!(received.operation_id(), source.operation_id());
    assert!(!headers.contains_key(OPERATION_ID_HEADER));
}

#[test]
fn invalid_operation_header_is_not_reflected() {
    let mut headers = HeaderMap::new();
    headers.insert(OPERATION_ID_HEADER, HeaderValue::from_static("not-a-uuid"));

    let context = OperationalContext::take_from_headers(&mut headers);

    assert_ne!(context.operation_id(), "not-a-uuid");
    assert!(Uuid::parse_str(context.operation_id()).is_ok());
    assert!(!headers.contains_key(OPERATION_ID_HEADER));
}

#[test]
fn gateway_context_replaces_a_caller_supplied_operation_id() {
    let mut headers = HeaderMap::new();
    headers.insert(
        OPERATION_ID_HEADER,
        HeaderValue::from_static("018f0f3f-3f7a-7b72-9d0d-e0d8ced91d8b"),
    );

    let context = OperationalContext::new_gateway(&mut headers);

    assert_ne!(
        context.operation_id(),
        "018f0f3f-3f7a-7b72-9d0d-e0d8ced91d8b"
    );
    assert!(!headers.contains_key(OPERATION_ID_HEADER));
}

#[test]
fn session_correlation_is_stable_and_does_not_retain_native_text() {
    let native_session_id = "private-session-sentinel";
    let first = OperationalContext::new().with_session(native_session_id);
    let second = OperationalContext::new().with_session(native_session_id);
    let different = OperationalContext::new().with_session("other-private-session");

    let first_tag = first.test_session_tag().unwrap();
    assert_eq!(first_tag, second.test_session_tag().unwrap());
    assert_ne!(first_tag, different.test_session_tag().unwrap());
    assert_ne!(first_tag, native_session_id);
    assert!(!first_tag.contains(native_session_id));
}
