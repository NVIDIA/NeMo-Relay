// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::json;

#[test]
fn test_to_napi_err_preserves_message() {
    let err = to_napi_err(FlowError::Internal("node binding failure".into()));
    assert!(err.to_string().contains("node binding failure"));
}

#[test]
fn test_opt_json_filters_null_only() {
    assert_eq!(opt_json(None), None);
    assert_eq!(opt_json(Some(Json::Null)), None);
    assert_eq!(
        opt_json(Some(json!({"ok": true}))),
        Some(json!({"ok": true}))
    );
}

#[test]
fn test_callback_error_store_round_trip() {
    clear_last_callback_error();
    assert_eq!(get_last_callback_error(), None);

    record_callback_error("node callback failed");
    assert_eq!(
        get_last_callback_error(),
        Some("node callback failed".to_string())
    );

    clear_last_callback_error();
    assert_eq!(get_last_callback_error(), None);
}
