// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::json;

#[test]
fn opt_json_filters_null_values() {
    assert_eq!(opt_json(None), None);
    assert_eq!(opt_json(Some(serde_json::Value::Null)), None);
    assert_eq!(
        opt_json(Some(json!({"ok": true}))),
        Some(json!({"ok": true}))
    );
}

#[test]
fn callback_error_helpers_round_trip_messages() {
    clear_last_callback_error();
    assert_eq!(get_last_callback_error(), None);

    record_callback_error("unit callback failure");
    assert_eq!(
        get_last_callback_error().as_deref(),
        Some("unit callback failure")
    );

    clear_last_callback_error();
    assert_eq!(get_last_callback_error(), None);
}
