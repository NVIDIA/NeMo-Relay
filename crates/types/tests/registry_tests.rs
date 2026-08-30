// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serialization and trait coverage for shared runtime-registration DTOs.

use std::collections::HashSet;

use nemo_relay_types::api::registry::{
    RuntimeRegistrationIdentity, RuntimeRegistrationKind, RuntimeRegistrationOwner,
    RuntimeRegistrationOwnerKind,
};
use serde_json::json;

#[test]
fn runtime_registration_kinds_use_stable_snake_case_names() {
    let cases = [
        (RuntimeRegistrationKind::Subscriber, "subscriber"),
        (
            RuntimeRegistrationKind::EventMetadataInjector,
            "event_metadata_injector",
        ),
        (
            RuntimeRegistrationKind::MarkSanitizeGuardrail,
            "mark_sanitize_guardrail",
        ),
        (
            RuntimeRegistrationKind::ScopeSanitizeStartGuardrail,
            "scope_sanitize_start_guardrail",
        ),
        (
            RuntimeRegistrationKind::ScopeSanitizeEndGuardrail,
            "scope_sanitize_end_guardrail",
        ),
        (
            RuntimeRegistrationKind::ToolSanitizeRequestGuardrail,
            "tool_sanitize_request_guardrail",
        ),
        (
            RuntimeRegistrationKind::ToolSanitizeResponseGuardrail,
            "tool_sanitize_response_guardrail",
        ),
        (
            RuntimeRegistrationKind::ToolConditionalExecutionGuardrail,
            "tool_conditional_execution_guardrail",
        ),
        (
            RuntimeRegistrationKind::ToolRequestIntercept,
            "tool_request_intercept",
        ),
        (
            RuntimeRegistrationKind::ToolExecutionIntercept,
            "tool_execution_intercept",
        ),
        (
            RuntimeRegistrationKind::LlmSanitizeRequestGuardrail,
            "llm_sanitize_request_guardrail",
        ),
        (
            RuntimeRegistrationKind::LlmSanitizeResponseGuardrail,
            "llm_sanitize_response_guardrail",
        ),
        (
            RuntimeRegistrationKind::LlmConditionalExecutionGuardrail,
            "llm_conditional_execution_guardrail",
        ),
        (
            RuntimeRegistrationKind::LlmRequestIntercept,
            "llm_request_intercept",
        ),
        (
            RuntimeRegistrationKind::LlmExecutionIntercept,
            "llm_execution_intercept",
        ),
        (
            RuntimeRegistrationKind::LlmStreamExecutionIntercept,
            "llm_stream_execution_intercept",
        ),
    ];

    for (kind, expected) in cases {
        assert_eq!(kind.as_str(), expected);
        assert_eq!(serde_json::to_value(kind).unwrap(), json!(expected));
        assert_eq!(
            serde_json::from_value::<RuntimeRegistrationKind>(json!(expected)).unwrap(),
            kind
        );
    }
}

#[test]
fn runtime_registration_enums_are_copy_and_kind_is_hashable() {
    fn copied<T: Copy>(value: T) -> T {
        value
    }

    let kind = copied(RuntimeRegistrationKind::Subscriber);
    let owner_kind = copied(RuntimeRegistrationOwnerKind::Plugin);
    let kinds = HashSet::from([kind]);

    assert!(kinds.contains(&RuntimeRegistrationKind::Subscriber));
    assert_eq!(owner_kind, RuntimeRegistrationOwnerKind::Plugin);
}

#[test]
fn runtime_registration_identity_round_trips_plugin_owner() {
    let identity = RuntimeRegistrationIdentity {
        kind: RuntimeRegistrationKind::Subscriber,
        local_name: "opentelemetry".into(),
        effective_name: "nemo-relay-plugin.v1.observability:1:opentelemetry".into(),
        owner: RuntimeRegistrationOwner {
            kind: RuntimeRegistrationOwnerKind::Plugin,
            plugin_kind: Some("observability".into()),
            component_ordinal: Some(1),
        },
    };

    let encoded = serde_json::to_value(&identity).unwrap();
    assert_eq!(
        encoded,
        json!({
            "kind": "subscriber",
            "local_name": "opentelemetry",
            "effective_name": "nemo-relay-plugin.v1.observability:1:opentelemetry",
            "owner": {
                "kind": "plugin",
                "plugin_kind": "observability",
                "component_ordinal": 1
            }
        })
    );
    assert_eq!(
        serde_json::from_value::<RuntimeRegistrationIdentity>(encoded).unwrap(),
        identity
    );
}

#[test]
fn runtime_registration_identity_round_trips_absent_owner_details() {
    let identity = RuntimeRegistrationIdentity {
        kind: RuntimeRegistrationKind::ToolRequestIntercept,
        local_name: "global-intercept".into(),
        effective_name: "global-intercept".into(),
        owner: RuntimeRegistrationOwner {
            kind: RuntimeRegistrationOwnerKind::GlobalApi,
            plugin_kind: None,
            component_ordinal: None,
        },
    };

    let encoded = serde_json::to_value(&identity).unwrap();
    assert_eq!(encoded["owner"]["plugin_kind"], json!(null));
    assert_eq!(encoded["owner"]["component_ordinal"], json!(null));
    assert_eq!(
        serde_json::from_value::<RuntimeRegistrationIdentity>(encoded).unwrap(),
        identity
    );
}
