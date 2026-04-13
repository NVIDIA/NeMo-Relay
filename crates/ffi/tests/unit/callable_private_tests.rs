// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_callable_private_helper_paths() {
    clear_last_error();
    let err = json_result_from_ptr(std::ptr::null_mut(), "fallback helper message").unwrap_err();
    assert!(err.to_string().contains("fallback helper message"));

    assert_eq!(ptr_to_opt_string(std::ptr::null_mut()), None);

    let raw = CString::new("ffi-string").unwrap().into_raw();
    assert_eq!(ptr_to_opt_string(raw), Some("ffi-string".into()));
    unsafe { nemo_flow_string_free_internal(raw) };
}
