// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for shared OTLP attribute projection.

use super::{
    OtlpAttributeMapping, apply_attribute_mappings, attribute_mapping_inputs,
    promote_event_metadata_attributes, push_top_level_json_attributes,
    validate_metadata_promotion_prefixes,
};
use crate::api::event::{BaseEvent, Event, MarkEvent};
use std::collections::HashSet;

#[test]
fn retains_only_mapping_sources_and_existing_aliases_between_span_events() {
    let attributes = vec![
        opentelemetry::KeyValue::new("source", "value"),
        opentelemetry::KeyValue::new("alias", "existing"),
        opentelemetry::KeyValue::new("large.request", "payload"),
    ];

    let retained =
        attribute_mapping_inputs(&attributes, &[OtlpAttributeMapping::new("source", "alias")]);

    assert_eq!(retained.len(), 2);
    assert!(
        retained
            .iter()
            .any(|attribute| attribute.key.as_str() == "source")
    );
    assert!(
        retained
            .iter()
            .any(|attribute| attribute.key.as_str() == "alias")
    );
}

#[test]
fn projects_typed_json_and_copies_configured_aliases() {
    let mut attributes = Vec::new();
    push_top_level_json_attributes(
        &mut attributes,
        "nemo_relay.start.metadata",
        Some(&serde_json::json!({
            "tenant": "acme",
            "attempt": 2,
            "enabled": true,
            "unset": null,
            "tags": ["a", "b"],
            "context": {"region": "us-east-1"},
            "request": {"id": "nested-id"},
            "request.id": "flat-id",
            "event_id": 18446744073709551615u64
        })),
    );
    apply_attribute_mappings(
        &mut attributes,
        &[OtlpAttributeMapping::new(
            "nemo_relay.start.metadata.tenant",
            "tenant.id",
        )],
    );

    assert_eq!(
        attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == "nemo_relay.start.metadata.enabled")
            .map(|attribute| &attribute.value),
        Some(&opentelemetry::Value::Bool(true))
    );

    let values = attributes
        .iter()
        .map(|attribute| (attribute.key.as_str(), attribute.value.to_string()))
        .collect::<std::collections::HashMap<_, _>>();
    assert_eq!(
        values.get("nemo_relay.start.metadata.tenant"),
        Some(&"acme".to_string())
    );
    assert_eq!(
        values.get("nemo_relay.start.metadata.attempt"),
        Some(&"2".to_string())
    );
    assert!(!values.contains_key("nemo_relay.start.metadata.unset"));
    assert_eq!(
        values.get("nemo_relay.start.metadata.tags"),
        Some(&"[\"a\",\"b\"]".to_string())
    );
    assert_eq!(
        values.get("nemo_relay.start.metadata.context"),
        Some(&"{\"region\":\"us-east-1\"}".to_string())
    );
    assert_eq!(
        values.get("nemo_relay.start.metadata.request"),
        Some(&"{\"id\":\"nested-id\"}".to_string())
    );
    assert_eq!(
        values.get("nemo_relay.start.metadata.request.id"),
        Some(&"flat-id".to_string())
    );
    assert_eq!(
        values.get("nemo_relay.start.metadata.event_id"),
        Some(&"18446744073709551615".to_string())
    );
    assert_eq!(values.get("tenant.id"), Some(&"acme".to_string()));
    assert!(!values.contains_key("nemo_relay.start.metadata_json"));
}

#[test]
fn rejects_invalid_attribute_mappings() {
    assert!(super::validate_attribute_mappings(&[OtlpAttributeMapping::new("", "alias")]).is_err());
    assert!(
        super::validate_attribute_mappings(&[
            OtlpAttributeMapping::new("one", "duplicate"),
            OtlpAttributeMapping::new("two", "duplicate"),
        ])
        .is_err()
    );
    assert!(
        super::validate_attribute_mappings(&[
            OtlpAttributeMapping::new("one", "duplicate"),
            OtlpAttributeMapping::new("two", " duplicate "),
        ])
        .is_err()
    );
    assert!(
        super::validate_attribute_mappings(&[OtlpAttributeMapping::new("key", "   ")]).is_err()
    );
    for invisible in ["\0", "\u{200b}", "\u{feff}", "\u{2060}"] {
        assert!(
            super::validate_attribute_mappings(&[OtlpAttributeMapping::new(invisible, "alias")])
                .is_err()
        );
        assert!(
            super::validate_attribute_mappings(&[OtlpAttributeMapping::new("key", invisible)])
                .is_err()
        );
    }
    assert!(
        super::validate_attribute_mappings(&[OtlpAttributeMapping::new(
            "key",
            "tenant\u{200b}.id"
        )])
        .is_ok()
    );
    assert!(super::validate_attribute_mappings(&[OtlpAttributeMapping::new("key", ".")]).is_ok());
}

#[test]
fn promotes_matching_primitive_metadata_without_overwriting_owned_keys() {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("metadata-promotion")
            .metadata(serde_json::json!({
                "nv.string": "value",
                "nv.bool": true,
                "nv.integer": 2,
                "nv.strings": ["a", "b"],
                "nv.bools": [true, false],
                "nv.integers": [2, 3],
                "nv.floats": [1.5, 2.5],
                "nv.nested": {"unsupported": true},
                "nv.owned": "attempted-overwrite",
                "other.unmatched": "ignored"
            }))
            .build(),
        None,
        None,
    ));
    let mut unpromoted_attributes = Vec::new();
    let unpromoted_issues =
        promote_event_metadata_attributes(&mut unpromoted_attributes, &event, &[], &HashSet::new());
    assert!(unpromoted_attributes.is_empty());
    assert!(unpromoted_issues.issues.is_empty());

    let mut attributes = vec![opentelemetry::KeyValue::new("nv.owned", "projection")];

    let issues = promote_event_metadata_attributes(
        &mut attributes,
        &event,
        &["nv.".to_string()],
        &HashSet::new(),
    );

    let value = |key| {
        attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == key)
            .map(|attribute| &attribute.value)
    };
    assert_eq!(
        value("nv.string"),
        Some(&opentelemetry::Value::String("value".into()))
    );
    assert_eq!(value("nv.bool"), Some(&opentelemetry::Value::Bool(true)));
    assert_eq!(value("nv.integer"), Some(&opentelemetry::Value::I64(2)));
    assert_eq!(
        value("nv.strings"),
        Some(&opentelemetry::Value::Array(opentelemetry::Array::String(
            vec!["a".into(), "b".into()]
        )))
    );
    assert_eq!(
        value("nv.bools"),
        Some(&opentelemetry::Value::Array(opentelemetry::Array::Bool(
            vec![true, false]
        )))
    );
    assert_eq!(
        value("nv.integers"),
        Some(&opentelemetry::Value::Array(opentelemetry::Array::I64(
            vec![2, 3]
        )))
    );
    assert_eq!(
        value("nv.floats"),
        Some(&opentelemetry::Value::Array(opentelemetry::Array::F64(
            vec![1.5, 2.5]
        )))
    );
    assert_eq!(
        value("nv.owned"),
        Some(&opentelemetry::Value::String("projection".into()))
    );
    assert_eq!(value("other.unmatched"), None);
    assert_eq!(
        issues.issues,
        vec![super::MetadataPromotionIssue {
            key: "nv.nested".to_string(),
            reason: "object values are not supported",
        }]
    );
}

#[test]
fn treats_metadata_promotion_prefixes_as_literal_string_prefixes() {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("literal-metadata-prefixes")
            .metadata(serde_json::json!({
                "nv.dot": "dot",
                "nv_underscore": "underscore",
                "unrelated": "ignored",
                "user_api_key": "api-key",
                "username": "name"
            }))
            .build(),
        None,
        None,
    ));
    let mut attributes = Vec::new();

    let issues = promote_event_metadata_attributes(
        &mut attributes,
        &event,
        &["nv.".to_string(), "nv_".to_string(), "user".to_string()],
        &HashSet::new(),
    );

    assert!(issues.issues.is_empty());
    assert_eq!(
        attributes
            .iter()
            .map(|attribute| attribute.key.as_str())
            .collect::<HashSet<_>>(),
        HashSet::from(["nv.dot", "nv_underscore", "user_api_key", "username"])
    );
}

#[test]
fn rejects_metadata_keys_owned_by_relay_and_otel_projections() {
    let reserved_keys = [
        "error.type",
        "exception.type",
        "gen_ai.request.model",
        "input.value",
        "llm.model_name",
        "metadata",
        "nemo_relay.uuid",
        "openinference.span.kind",
        "output.value",
        "server.address",
        "service.name",
        "session.id",
        "tool.name",
        "tool_call.id",
        "user.id",
    ];
    let mut metadata = serde_json::Map::new();
    for key in reserved_keys {
        metadata.insert(key.to_string(), serde_json::json!("blocked"));
    }
    metadata.insert("nv.source".to_string(), serde_json::json!("allowed"));
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("reserved-metadata-promotion")
            .metadata(serde_json::Value::Object(metadata))
            .build(),
        None,
        None,
    ));
    let mut attributes = Vec::new();
    let prefixes = reserved_keys
        .iter()
        .copied()
        .chain(std::iter::once("nv."))
        .map(str::to_string)
        .collect::<Vec<_>>();

    let issues =
        promote_event_metadata_attributes(&mut attributes, &event, &prefixes, &HashSet::new());

    assert_eq!(
        attributes,
        vec![opentelemetry::KeyValue::new("nv.source", "allowed")]
    );
    assert_eq!(
        issues
            .issues
            .iter()
            .map(|issue| issue.key.as_str())
            .collect::<std::collections::HashSet<_>>(),
        std::collections::HashSet::from(reserved_keys)
    );
    assert!(issues.issues.iter().all(|issue| {
        issue.reason == "attribute key is reserved by Relay or an OpenTelemetry projection"
    }));
}

#[test]
fn promotes_empty_metadata_arrays() {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("empty-metadata-array")
            .metadata(serde_json::json!({"nv.empty": []}))
            .build(),
        None,
        None,
    ));
    let mut attributes = Vec::new();

    let issues = promote_event_metadata_attributes(
        &mut attributes,
        &event,
        &["nv.".to_string()],
        &HashSet::new(),
    );

    assert!(issues.issues.is_empty());
    assert_eq!(
        attributes,
        vec![opentelemetry::KeyValue::new(
            "nv.empty",
            opentelemetry::Value::Array(opentelemetry::Array::String(Vec::new())),
        )]
    );
}

#[test]
fn reports_unsupported_metadata_values() {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("metadata-array-rejections")
            .metadata(serde_json::json!({
                "nv.mixed": [1, "two"],
                "nv.nested": [[1]],
                "nv.nulls": [null],
                "nv.oversized_scalar": 18446744073709551615u64,
                "nv.oversized": [18446744073709551615u64]
            }))
            .build(),
        None,
        None,
    ));
    let mut attributes = Vec::new();

    let issues = promote_event_metadata_attributes(
        &mut attributes,
        &event,
        &["nv.".to_string()],
        &HashSet::new(),
    );

    assert!(attributes.is_empty());
    let issues = issues
        .issues
        .into_iter()
        .map(|issue| (issue.key, issue.reason))
        .collect::<std::collections::HashMap<_, _>>();
    assert_eq!(issues.len(), 5);
    assert_eq!(
        issues.get("nv.mixed"),
        Some(&"array values must have one primitive type")
    );
    assert_eq!(
        issues.get("nv.nested"),
        Some(&"nested arrays and objects are not supported")
    );
    assert_eq!(
        issues.get("nv.nulls"),
        Some(&"arrays of null are not OTLP attributes")
    );
    assert_eq!(
        issues.get("nv.oversized_scalar"),
        Some(&"unsigned integer is larger than OTLP i64")
    );
    assert_eq!(
        issues.get("nv.oversized"),
        Some(&"array contains an unsigned integer larger than OTLP i64")
    );
}

#[test]
fn validates_metadata_promotion_prefixes_against_metadata_key_syntax() {
    assert!(validate_metadata_promotion_prefixes(&[]).is_ok());

    for prefix in [
        "nv",
        "nv.",
        "nv_",
        "nv-",
        "nv.client",
        "nv.client.",
        "nv_client",
        "nv-client",
        "nv2.",
        "NV.",
    ] {
        assert!(
            validate_metadata_promotion_prefixes(&[prefix.to_string()]).is_ok(),
            "expected {prefix:?} to be accepted"
        );
    }

    assert!(
        validate_metadata_promotion_prefixes(&[
            "nv.".to_string(),
            "os.".to_string(),
            "host_".to_string(),
        ])
        .is_ok()
    );

    for prefix in [
        "", " ", " nv.", "nv. ", ".nv", "nv..", "nv:", "nv/", "nv value", "nv.*",
    ] {
        assert!(
            validate_metadata_promotion_prefixes(&[prefix.to_string()]).is_err(),
            "expected {prefix:?} to be rejected"
        );
    }

    assert!(validate_metadata_promotion_prefixes(&["nv.".to_string(), "nv.".to_string()]).is_err());
}
