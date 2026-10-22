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

#[test]
fn canonical_fixture_converts_numbers_and_keeps_restricted_text() {
    let mut value: serde_json::Value = serde_json::from_str(include_str!(
        "../../../types/tests/fixtures/diagnostics/diagnostic-snapshot-schema-1.json"
    ))
    .unwrap();
    native_numbers(&mut value, "");
    assert_eq!(value["sequence"].as_u64(), Some(u64::MAX));
    assert_eq!(
        value["events"][0]["count"].as_u64(),
        Some(9_007_199_254_740_993)
    );
    assert_eq!(value["events"][0]["correlation"]["session_id"], "session-1");
    assert_eq!(
        value["components"][0]["lifetime"]["totals"]["occurrences_total"]["saturated"],
        true
    );
}

#[test]
fn canonical_memory_fixture_converts_all_quantities_without_retyping_text_or_decimals() {
    let mut value: serde_json::Value = serde_json::from_str(include_str!(
        "../../../types/tests/fixtures/diagnostics/diagnostic-memory-schema-1.json"
    ))
    .unwrap();
    let original = value.clone();
    native_numbers(&mut value, "");
    fn compare(original: &serde_json::Value, converted: &serde_json::Value, field: &str) {
        match original {
            serde_json::Value::String(text)
                if nemo_relay::diagnostics::embedded::is_diagnostic_number_field(field) =>
            {
                assert_eq!(converted.as_u64(), text.parse::<u64>().ok(), "{field}");
                assert!(converted.is_u64(), "{field}");
            }
            serde_json::Value::Object(fields) => {
                for (key, child) in fields {
                    compare(child, &converted[key], key);
                }
            }
            serde_json::Value::Array(values) => {
                for (index, child) in values.iter().enumerate() {
                    compare(child, &converted[index], "");
                }
            }
            _ => assert_eq!(original, converted, "{field}"),
        }
    }
    compare(&original, &value, "");
    assert_eq!(value["sequence"].as_u64(), Some(u64::MAX));
    assert_eq!(
        value["components"][0]["component"]["key"],
        "9007199254740993"
    );
    assert_eq!(
        value["resource"]["sample"]["memory"]["private"]["value"]["value"],
        1.25
    );
}
