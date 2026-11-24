// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checked LLM application context envelope and resource limits.

use nemo_relay_types::api::llm::{ApplicationExecutionContext, ApplicationExecutionContextError};
use serde_json::json;

#[test]
fn envelope_distinguishes_absent_and_empty_data_and_redacts_debug() {
    for value in [
        json!({}),
        json!({"data": {}}),
        json!({"data": {"private": "sentinel"}}),
    ] {
        let context = ApplicationExecutionContext::try_from(value.clone()).unwrap();
        assert_eq!(context.to_json(), value);
        assert_eq!(
            serde_json::from_value::<ApplicationExecutionContext>(value).unwrap(),
            context
        );
        assert!(!format!("{context:?}").contains("sentinel"));
    }
    for value in [
        json!(null),
        json!([]),
        json!("text"),
        json!({"extra": 1}),
        json!({"data": null}),
        json!({"data": []}),
    ] {
        assert!(ApplicationExecutionContext::try_from(value).is_err());
    }
}

#[test]
fn exact_serialized_byte_and_container_depth_limits() {
    let overhead = json!({"data": {"value": ""}}).to_string().len();
    let at_limit = json!({"data": {"value": "x".repeat(16_384 - overhead)}});
    ApplicationExecutionContext::try_from(at_limit).unwrap();
    for value in [
        json!({"data": {"value": "x".repeat(16_385 - overhead)}}),
        json!({"data": {"value": "\0".repeat(3_000)}}),
    ] {
        assert_eq!(
            ApplicationExecutionContext::try_from(value).unwrap_err(),
            ApplicationExecutionContextError::TooLarge
        );
    }
    let mut nested = json!({});
    for _ in 0..62 {
        nested = json!({"child": nested});
    }
    ApplicationExecutionContext::try_from(json!({"data": nested.clone()})).unwrap();
    assert_eq!(
        ApplicationExecutionContext::try_from(json!({"data": {"child": nested}})).unwrap_err(),
        ApplicationExecutionContextError::TooDeep
    );
}

#[test]
fn rejected_context_cleanup_survives_extreme_depth() {
    const CHILD: &str = "RELAY_LLM_CONTEXT_DROP_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "rejected_context_cleanup_survives_extreme_depth",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let deep = || {
        let mut value = json!(null);
        for index in 0..262_144 {
            value = if index % 2 == 0 {
                serde_json::Value::Array(vec![value])
            } else {
                serde_json::Value::Object(serde_json::Map::from_iter([("child".into(), value)]))
            };
        }
        value
    };
    assert_eq!(
        ApplicationExecutionContext::try_from(serde_json::Value::Object(
            serde_json::Map::from_iter([(
                "data".into(),
                serde_json::Value::Object(serde_json::Map::from_iter([("value".into(), deep())]))
            )])
        ))
        .unwrap_err(),
        ApplicationExecutionContextError::TooDeep
    );
    let mut context = ApplicationExecutionContext::default();
    context.data = Some(std::collections::BTreeMap::from([("value".into(), deep())]));
    assert_eq!(
        context.validate().unwrap_err(),
        ApplicationExecutionContextError::TooDeep
    );
    drop(context);
    ApplicationExecutionContext::try_from(json!({"data": {"recovery": true}})).unwrap();
}

#[test]
fn wire_preflight_bounds_depth_before_json_decoding() {
    for (input, expected) in [
        (
            format!("{{\"data\":{{\"value\":\"{}\"}}}}", "x".repeat(1_048_576)),
            ApplicationExecutionContextError::TooLarge,
        ),
        (
            format!(
                "{{\"data\":{{\"value\":{}null{}}}}}",
                "[".repeat(1100),
                "]".repeat(1100)
            ),
            ApplicationExecutionContextError::TooDeep,
        ),
    ] {
        assert_eq!(
            ApplicationExecutionContext::from_json_slice(input.as_bytes()).unwrap_err(),
            expected
        );
        ApplicationExecutionContext::from_json_slice(br#"{"data":{"recovery":true}}"#).unwrap();
    }
    let mut boundary = json!({"data": {"number": 1e-7, "pad": ""}});
    let padding = 16_384 - boundary.to_string().len();
    boundary["data"]["pad"] = json!("x".repeat(padding));
    ApplicationExecutionContext::from_json_slice(boundary.to_string().as_bytes()).unwrap();
    // Quoted braces and escaped quotes do not contribute nesting levels.
    ApplicationExecutionContext::from_json_slice(br#"{"data":{"text":"[\"{}]"}}"#).unwrap();
}
