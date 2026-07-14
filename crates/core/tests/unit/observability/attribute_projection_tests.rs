// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for shared OTLP attribute projection.

use super::{
    OtlpAttributeMapping, apply_attribute_mappings, attribute_mapping_inputs,
    push_top_level_json_attributes,
};

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
    assert_eq!(
        values.get("nemo_relay.start.metadata.enabled"),
        Some(&"true".to_string())
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
}
