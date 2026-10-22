// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::py_diagnostics::native_numbers;
#[test]
fn native_diagnostic_numbers_are_lossless_without_changing_text() {
    let mut value = serde_json::json!({
        "sequence":"9007199254740993", "count":"18446744073709551615",
        "detail":{"message":"18446744073709551615"},
        "classification":{"canonical_code":"future.unknown", "failure_class":"future.cause"},
        "totals":{"failures_total":{"value":"18446744073709551615", "saturated":true}},
        "since_last_failure_millis":null,
    });
    native_numbers(&mut value, "");
    assert_eq!(value["sequence"].as_u64(), Some(9007199254740993));
    assert_eq!(value["count"].as_u64(), Some(u64::MAX));
    assert_eq!(value["detail"]["message"], "18446744073709551615");
    assert_eq!(value["classification"]["canonical_code"], "future.unknown");
    assert!(value["since_last_failure_millis"].is_null());
}
