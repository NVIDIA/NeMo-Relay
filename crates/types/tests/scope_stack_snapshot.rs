// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serialization contract for detached scope stack snapshots.

use nemo_relay_types::api::scope::{ScopeStackEntry, ScopeStackSnapshot, ScopeType};

#[test]
fn scope_stack_snapshot_shared_fixture_round_trips() {
    let encoded: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/scope_stack_snapshot.json")).unwrap();
    let snapshot: ScopeStackSnapshot = serde_json::from_value(encoded.clone()).unwrap();
    assert_eq!(snapshot.scopes.len(), 5);
    assert_eq!(snapshot.scopes[3].scope_type, ScopeType::Tool);
    assert_eq!(
        snapshot.scopes[0].scope_id.to_string(),
        "019a0000-0000-7000-8000-000000000001"
    );
    assert_eq!(serde_json::to_value(snapshot).unwrap(), encoded);
}

#[test]
fn scope_stack_snapshot_types_round_trip_and_future_category_is_unknown() {
    for category in [
        ScopeType::Agent,
        ScopeType::Function,
        ScopeType::Tool,
        ScopeType::Llm,
        ScopeType::Retriever,
        ScopeType::Embedder,
        ScopeType::Reranker,
        ScopeType::Guardrail,
        ScopeType::Evaluator,
        ScopeType::Custom,
        ScopeType::Unknown,
    ] {
        let entry = ScopeStackEntry {
            scope_id: uuid::Uuid::parse_str("019a0000-0000-7000-8000-000000000001").unwrap(),
            scope_type: category,
        };
        let encoded = serde_json::to_value(&entry).unwrap();
        assert_eq!(encoded["scope_type"], category.as_str());
        assert_eq!(
            serde_json::from_value::<ScopeStackEntry>(encoded).unwrap(),
            entry
        );
    }
    let future = serde_json::json!({"scope_id":"019a0000-0000-7000-8000-000000000001", "scope_type":"future-category"});
    assert_eq!(
        serde_json::from_value::<ScopeStackEntry>(future)
            .unwrap()
            .scope_type,
        ScopeType::Unknown
    );
    assert!(
        serde_json::from_str::<ScopeStackEntry>(r#"{"scope_id":"invalid","scope_type":"agent"}"#)
            .is_err()
    );
}
