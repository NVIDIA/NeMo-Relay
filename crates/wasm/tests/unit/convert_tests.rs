// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_callback_error_store_round_trip() {
    clear_last_callback_error();
    assert_eq!(get_last_callback_error(), None);

    record_callback_error("wasm callback failed");
    assert_eq!(
        get_last_callback_error(),
        Some("wasm callback failed".to_string())
    );

    clear_last_callback_error();
    assert_eq!(get_last_callback_error(), None);
}
