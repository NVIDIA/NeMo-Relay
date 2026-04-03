// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_rejection_message_prefers_string_then_object_then_unknown() {
    assert_eq!(rejection_message(Ok("boom".into()), None), "boom");
    assert_eq!(
        rejection_message(
            Err(napi::Error::from_reason("x")),
            Some(Ok("from object".into()))
        ),
        "from object"
    );
    assert_eq!(
        rejection_message(
            Err(napi::Error::from_reason("x")),
            Some(Err(napi::Error::from_reason("y")))
        ),
        "unknown error"
    );
    assert_eq!(
        rejection_message(Err(napi::Error::from_reason("x")), None),
        "unknown error"
    );
}

#[test]
fn test_queue_status_and_closed_helpers_cover_error_paths() {
    assert!(queue_status_result(napi::Status::Ok).is_ok());
    assert!(queue_status_result(napi::Status::GenericFailure)
        .unwrap_err()
        .to_string()
        .contains("failed to queue threadsafe function call"));
    assert!(closed_tsfn_error()
        .to_string()
        .contains("PromiseAwareFn threadsafe function closed"));
}
