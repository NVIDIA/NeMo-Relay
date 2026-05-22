// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for types in the NeMo Relay adaptive crate.

use super::*;
use chrono::Utc;
use std::collections::HashMap;
use uuid::Uuid;

fn assert_send_sync<T: Send + Sync>() {}

fn sample_agent_identity() -> AgentIdentity {
    AgentIdentity {
        agent_id: "research-agent".to_string(),
        template_version: "1.2.0".to_string(),
        toolset_hash: "abc123def".to_string(),
        model_family: "claude".to_string(),
        tenant_scope: "tenant-42".to_string(),
    }
}

// -----------------------------------------------------------------------
// Send + Sync compile-time assertions
// -----------------------------------------------------------------------

#[test]
fn test_types_are_send_sync() {
    assert_send_sync::<AgentIdentity>();
    assert_send_sync::<OptimizationIntentBundle>();
    assert_send_sync::<OptimizationIntent>();
    assert_send_sync::<CacheStabilityIntent>();
    assert_send_sync::<ContentExtractionIntent>();
    assert_send_sync::<SerializationIntent>();
    assert_send_sync::<PriorityIntent>();
    assert_send_sync::<ModelRoutingIntent>();
    assert_send_sync::<PlacementIntent>();
    assert_send_sync::<RetentionIntent>();
    assert_send_sync::<ToolScopeIntent>();
    assert_send_sync::<CompressionIntent>();
    assert_send_sync::<SharingScope>();
    assert_send_sync::<RetentionTier>();
    assert_send_sync::<PlacementTarget>();
    assert_send_sync::<ModelClass>();
    assert_send_sync::<IntentType>();
}

// -----------------------------------------------------------------------
// Supporting enum serde round-trips
// -----------------------------------------------------------------------

#[test]
fn test_supporting_enums_serde() {
    // SharingScope
    for scope in [
        SharingScope::Request,
        SharingScope::Session,
        SharingScope::Tenant,
        SharingScope::Global,
    ] {
        let json = serde_json::to_string(&scope).unwrap();
        let restored: SharingScope = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, scope);
    }

    // RetentionTier
    for tier in [
        RetentionTier::Ephemeral,
        RetentionTier::ShortLived,
        RetentionTier::SessionDuration,
        RetentionTier::LongLived,
        RetentionTier::Permanent,
    ] {
        let json = serde_json::to_string(&tier).unwrap();
        let restored: RetentionTier = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, tier);
    }

    // PlacementTarget
    for target in [
        PlacementTarget::CacheablePrefix,
        PlacementTarget::DeferredToolBlock,
        PlacementTarget::ArtifactReference,
        PlacementTarget::RetrievalOnDemand,
        PlacementTarget::SessionMemorySummary,
        PlacementTarget::NonCacheableSuffix,
    ] {
        let json = serde_json::to_string(&target).unwrap();
        let restored: PlacementTarget = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, target);
    }

    // ModelClass
    for class in [
        ModelClass::Economy,
        ModelClass::Standard,
        ModelClass::Premium,
        ModelClass::Critical,
    ] {
        let json = serde_json::to_string(&class).unwrap();
        let restored: ModelClass = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, class);
    }

    // IntentType
    for it in [
        IntentType::CacheStability,
        IntentType::ContentExtraction,
        IntentType::Serialization,
        IntentType::Priority,
        IntentType::ModelRouting,
        IntentType::Placement,
        IntentType::Retention,
        IntentType::ToolScope,
        IntentType::Compression,
    ] {
        let json = serde_json::to_string(&it).unwrap();
        let restored: IntentType = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, it);
    }
}

#[test]
fn test_sharing_scope_default() {
    assert_eq!(SharingScope::default(), SharingScope::Session);
}

// -----------------------------------------------------------------------
// Individual intent type serde
// -----------------------------------------------------------------------

#[test]
fn test_each_intent_type_serde() {
    let intents = vec![
        OptimizationIntent::CacheStability(CacheStabilityIntent {
            stability_score: 0.95,
            stable_prefix_end: 1024,
            recommended_retention_tier: Some(RetentionTier::LongLived),
            scope_label: SharingScope::Session,
            confidence: 0.87,
            evidence_count: 42,
        }),
        OptimizationIntent::ContentExtraction(ContentExtractionIntent {
            block_id: "blk-001".to_string(),
            variable_pattern: r"\{\{date\}\}".to_string(),
            extraction_strategy: "regex".to_string(),
            scope_label: SharingScope::Tenant,
        }),
        OptimizationIntent::Serialization(SerializationIntent {
            fanout_width: 3,
            expected_savings_tokens: 500,
            reuse_probability: 0.75,
            added_latency_ms: Some(12.5),
            scope_label: SharingScope::Global,
        }),
        OptimizationIntent::Priority(PriorityIntent {
            latency_sensitivity: 0.9,
            workflow_phase: Some("research".to_string()),
            caller_tier: Some("premium".to_string()),
        }),
        OptimizationIntent::ModelRouting(ModelRoutingIntent {
            model_class: ModelClass::Premium,
            complexity_score: 0.8,
            criticality: 0.95,
            fallback_allowed: true,
        }),
        OptimizationIntent::Placement(PlacementIntent {
            block_id: "blk-002".to_string(),
            target: PlacementTarget::CacheablePrefix,
            stability_score: 0.99,
            scope_label: SharingScope::Session,
        }),
        OptimizationIntent::Retention(RetentionIntent {
            recommended_tier: RetentionTier::SessionDuration,
            expected_session_duration_secs: Some(3600.0),
            inter_call_gap_p50_ms: Some(250.0),
            scope_label: SharingScope::Session,
        }),
        OptimizationIntent::ToolScope(ToolScopeIntent {
            active_tools: vec!["search".to_string(), "calculator".to_string()],
            phase_label: Some("analysis".to_string()),
            deferred_tools: vec!["code_exec".to_string()],
        }),
        OptimizationIntent::Compression(CompressionIntent {
            block_id: "blk-003".to_string(),
            compression_ratio: 0.65,
            reversible: true,
            contribution_score: 0.3,
        }),
    ];

    assert_eq!(intents.len(), 9, "must have exactly 9 intent types");

    for intent in &intents {
        let json = serde_json::to_string(intent).unwrap();
        let restored: OptimizationIntent = serde_json::from_str(&json).unwrap();
        assert_eq!(&restored, intent);
    }
}

// -----------------------------------------------------------------------
// CacheStabilityIntent f64 precision
// -----------------------------------------------------------------------

#[test]
fn test_cache_stability_intent_serde() {
    let intent = CacheStabilityIntent {
        stability_score: 0.123456789,
        stable_prefix_end: 2048,
        recommended_retention_tier: None,
        scope_label: SharingScope::Request,
        confidence: 0.999999,
        evidence_count: 100,
    };

    let json = serde_json::to_string(&intent).unwrap();
    let restored: CacheStabilityIntent = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.stability_score, intent.stability_score);
    assert_eq!(restored.confidence, intent.confidence);
    assert_eq!(restored.stable_prefix_end, 2048);
    assert_eq!(restored.evidence_count, 100);
    assert!(restored.recommended_retention_tier.is_none());
}

// -----------------------------------------------------------------------
// OptimizationIntentBundle serde round-trips
// -----------------------------------------------------------------------

#[test]
fn test_intent_bundle_serde_roundtrip() {
    let bundle = OptimizationIntentBundle {
        request_id: Uuid::new_v4(),
        agent_identity: sample_agent_identity(),
        policy_version: "1.0.0".to_string(),
        intents: vec![
            OptimizationIntent::CacheStability(CacheStabilityIntent {
                stability_score: 0.95,
                stable_prefix_end: 1024,
                recommended_retention_tier: Some(RetentionTier::LongLived),
                scope_label: SharingScope::Session,
                confidence: 0.87,
                evidence_count: 42,
            }),
            OptimizationIntent::Priority(PriorityIntent {
                latency_sensitivity: 0.9,
                workflow_phase: None,
                caller_tier: None,
            }),
            OptimizationIntent::ToolScope(ToolScopeIntent {
                active_tools: vec!["search".to_string()],
                phase_label: None,
                deferred_tools: vec![],
            }),
        ],
        created_at: Utc::now(),
    };

    let serialized = serde_json::to_string(&bundle).unwrap();
    let deserialized: OptimizationIntentBundle = serde_json::from_str(&serialized).unwrap();

    assert_eq!(deserialized.request_id, bundle.request_id);
    assert_eq!(deserialized.agent_identity, bundle.agent_identity);
    assert_eq!(deserialized.policy_version, "1.0.0");
    assert_eq!(deserialized.intents.len(), 3);
}

#[test]
fn test_empty_bundle_serde_roundtrip() {
    let bundle = OptimizationIntentBundle {
        request_id: Uuid::new_v4(),
        agent_identity: sample_agent_identity(),
        policy_version: "0.0.1".to_string(),
        intents: vec![],
        created_at: Utc::now(),
    };

    let serialized = serde_json::to_string(&bundle).unwrap();
    let deserialized: OptimizationIntentBundle = serde_json::from_str(&serialized).unwrap();

    assert_eq!(deserialized.request_id, bundle.request_id);
    assert!(deserialized.intents.is_empty());
}

// -----------------------------------------------------------------------
// AgentIdentity tests
// -----------------------------------------------------------------------

#[test]
fn test_agent_identity_serde_roundtrip() {
    let identity = sample_agent_identity();

    let serialized = serde_json::to_string(&identity).unwrap();
    let deserialized: AgentIdentity = serde_json::from_str(&serialized).unwrap();

    assert_eq!(deserialized, identity);
}

#[test]
fn test_agent_identity_hash_map_key() {
    let id1 = sample_agent_identity();
    let id2 = sample_agent_identity();

    let mut map = HashMap::new();
    map.insert(id1.clone(), "value-1");

    // Lookup with an equal key succeeds
    assert_eq!(map.get(&id2), Some(&"value-1"));

    // Lookup with a different key fails
    let id3 = AgentIdentity {
        agent_id: "different-agent".to_string(),
        ..id1
    };
    assert_eq!(map.get(&id3), None);
}

#[test]
fn test_agent_identity_equality() {
    let id1 = sample_agent_identity();
    let id2 = sample_agent_identity();
    assert_eq!(id1, id2);

    // Changing any field breaks equality
    let id_diff_agent = AgentIdentity {
        agent_id: "other".to_string(),
        ..id1.clone()
    };
    assert_ne!(id1, id_diff_agent);

    let id_diff_version = AgentIdentity {
        template_version: "2.0.0".to_string(),
        ..id1.clone()
    };
    assert_ne!(id1, id_diff_version);

    let id_diff_toolset = AgentIdentity {
        toolset_hash: "xyz".to_string(),
        ..id1.clone()
    };
    assert_ne!(id1, id_diff_toolset);

    let id_diff_model = AgentIdentity {
        model_family: "gpt".to_string(),
        ..id1.clone()
    };
    assert_ne!(id1, id_diff_model);

    let id_diff_tenant = AgentIdentity {
        tenant_scope: "tenant-99".to_string(),
        ..id1.clone()
    };
    assert_ne!(id1, id_diff_tenant);
}

#[test]
fn test_agent_identity_display() {
    let identity = sample_agent_identity();
    assert_eq!(format!("{identity}"), "research-agent@1.2.0");
}

// -----------------------------------------------------------------------
// TranslationReport types tests
// -----------------------------------------------------------------------

#[test]
fn test_translation_status_serde() {
    let cases = [
        (TranslationStatus::Applied, "\"applied\""),
        (TranslationStatus::Degraded, "\"degraded\""),
        (TranslationStatus::Ignored, "\"ignored\""),
        (TranslationStatus::Rejected, "\"rejected\""),
    ];
    for (status, expected_json) in &cases {
        let json = serde_json::to_string(status).unwrap();
        assert_eq!(&json, expected_json, "serialization of {status:?}");
        let restored: TranslationStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(&restored, status, "round-trip of {status:?}");
    }
}

#[test]
fn test_reason_code_serde() {
    let codes = vec![
        ReasonCode::FullySupported,
        ReasonCode::UnsupportedByBackend,
        ReasonCode::UnsupportedByModel,
        ReasonCode::BackendLimitReached,
        ReasonCode::InsufficientEvidence,
        ReasonCode::FeatureDisabled,
        ReasonCode::UnsafeForRequest,
        ReasonCode::PluginIncomplete,
        ReasonCode::NotRelevant,
        ReasonCode::Custom {
            reason: "experiment-x".to_string(),
        },
    ];
    for code in &codes {
        let json = serde_json::to_string(code).unwrap();
        let restored: ReasonCode = serde_json::from_str(&json).unwrap();
        assert_eq!(&restored, code, "round-trip of {code:?}");
    }

    // Verify tagged serialization uses "code" field with snake_case
    let json = serde_json::to_string(&ReasonCode::FullySupported).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["code"], "fully_supported");

    // Verify Custom variant includes reason field
    let json = serde_json::to_string(&ReasonCode::Custom {
        reason: "test-reason".to_string(),
    })
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["code"], "custom");
    assert_eq!(v["reason"], "test-reason");
}

#[test]
fn test_intent_outcome_serde_roundtrip() {
    let statuses = [
        (TranslationStatus::Applied, ReasonCode::FullySupported),
        (TranslationStatus::Degraded, ReasonCode::BackendLimitReached),
        (TranslationStatus::Ignored, ReasonCode::NotRelevant),
        (TranslationStatus::Rejected, ReasonCode::UnsafeForRequest),
    ];

    for (status, reason) in statuses {
        let outcome = IntentOutcome {
            intent_id: Uuid::new_v4(),
            intent_type: IntentType::CacheStability,
            status,
            reason,
            detail: Some(format!("detail for {status:?}")),
        };

        let json = serde_json::to_string(&outcome).unwrap();
        let restored: IntentOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, outcome);
    }
}

#[test]
fn test_translation_report_serde_roundtrip() {
    let report = TranslationReport {
        request_id: Uuid::new_v4(),
        plugin_id: "anthropic-v1".to_string(),
        outcomes: vec![
            IntentOutcome {
                intent_id: Uuid::new_v4(),
                intent_type: IntentType::CacheStability,
                status: TranslationStatus::Applied,
                reason: ReasonCode::FullySupported,
                detail: None,
            },
            IntentOutcome {
                intent_id: Uuid::new_v4(),
                intent_type: IntentType::ModelRouting,
                status: TranslationStatus::Rejected,
                reason: ReasonCode::UnsupportedByBackend,
                detail: Some("model routing not supported".to_string()),
            },
            IntentOutcome {
                intent_id: Uuid::new_v4(),
                intent_type: IntentType::Compression,
                status: TranslationStatus::Degraded,
                reason: ReasonCode::BackendLimitReached,
                detail: None,
            },
        ],
        created_at: Utc::now(),
    };

    let serialized = serde_json::to_string(&report).unwrap();
    let deserialized: TranslationReport = serde_json::from_str(&serialized).unwrap();

    assert_eq!(deserialized.request_id, report.request_id);
    assert_eq!(deserialized.plugin_id, "anthropic-v1");
    assert_eq!(deserialized.outcomes.len(), 3);
    assert_eq!(deserialized.outcomes[0].status, TranslationStatus::Applied);
    assert_eq!(deserialized.outcomes[1].status, TranslationStatus::Rejected);
    assert_eq!(deserialized.outcomes[2].status, TranslationStatus::Degraded);
}

#[test]
fn test_translation_report_all_applied_true() {
    let report = TranslationReport {
        request_id: Uuid::new_v4(),
        plugin_id: "test-plugin".to_string(),
        outcomes: vec![
            IntentOutcome {
                intent_id: Uuid::new_v4(),
                intent_type: IntentType::CacheStability,
                status: TranslationStatus::Applied,
                reason: ReasonCode::FullySupported,
                detail: None,
            },
            IntentOutcome {
                intent_id: Uuid::new_v4(),
                intent_type: IntentType::Priority,
                status: TranslationStatus::Applied,
                reason: ReasonCode::FullySupported,
                detail: None,
            },
        ],
        created_at: Utc::now(),
    };

    assert!(report.all_applied());
}

#[test]
fn test_translation_report_all_applied_false() {
    let report = TranslationReport {
        request_id: Uuid::new_v4(),
        plugin_id: "test-plugin".to_string(),
        outcomes: vec![
            IntentOutcome {
                intent_id: Uuid::new_v4(),
                intent_type: IntentType::CacheStability,
                status: TranslationStatus::Applied,
                reason: ReasonCode::FullySupported,
                detail: None,
            },
            IntentOutcome {
                intent_id: Uuid::new_v4(),
                intent_type: IntentType::ModelRouting,
                status: TranslationStatus::Rejected,
                reason: ReasonCode::UnsupportedByBackend,
                detail: None,
            },
        ],
        created_at: Utc::now(),
    };

    assert!(!report.all_applied());
}

#[test]
fn test_translation_report_outcomes_by_status() {
    let rejected_id = Uuid::new_v4();
    let report = TranslationReport {
        request_id: Uuid::new_v4(),
        plugin_id: "test-plugin".to_string(),
        outcomes: vec![
            IntentOutcome {
                intent_id: Uuid::new_v4(),
                intent_type: IntentType::CacheStability,
                status: TranslationStatus::Applied,
                reason: ReasonCode::FullySupported,
                detail: None,
            },
            IntentOutcome {
                intent_id: rejected_id,
                intent_type: IntentType::ModelRouting,
                status: TranslationStatus::Rejected,
                reason: ReasonCode::UnsupportedByBackend,
                detail: None,
            },
            IntentOutcome {
                intent_id: Uuid::new_v4(),
                intent_type: IntentType::Compression,
                status: TranslationStatus::Degraded,
                reason: ReasonCode::BackendLimitReached,
                detail: None,
            },
        ],
        created_at: Utc::now(),
    };

    let rejected = report.outcomes_by_status(TranslationStatus::Rejected);
    assert_eq!(rejected.len(), 1);
    assert_eq!(rejected[0].intent_id, rejected_id);
    assert_eq!(rejected[0].intent_type, IntentType::ModelRouting);

    let applied = report.outcomes_by_status(TranslationStatus::Applied);
    assert_eq!(applied.len(), 1);

    let ignored = report.outcomes_by_status(TranslationStatus::Ignored);
    assert!(ignored.is_empty());
}

#[test]
fn test_translation_report_empty() {
    let report = TranslationReport {
        request_id: Uuid::new_v4(),
        plugin_id: "empty-plugin".to_string(),
        outcomes: vec![],
        created_at: Utc::now(),
    };

    // Empty report: all_applied() should be true (vacuous truth)
    assert!(report.all_applied());

    // Round-trip
    let serialized = serde_json::to_string(&report).unwrap();
    let deserialized: TranslationReport = serde_json::from_str(&serialized).unwrap();
    assert_eq!(deserialized.request_id, report.request_id);
    assert!(deserialized.outcomes.is_empty());
}

#[test]
fn test_translation_report_count_by_status() {
    let report = TranslationReport {
        request_id: Uuid::new_v4(),
        plugin_id: "test-plugin".to_string(),
        outcomes: vec![
            IntentOutcome {
                intent_id: Uuid::new_v4(),
                intent_type: IntentType::CacheStability,
                status: TranslationStatus::Applied,
                reason: ReasonCode::FullySupported,
                detail: None,
            },
            IntentOutcome {
                intent_id: Uuid::new_v4(),
                intent_type: IntentType::Priority,
                status: TranslationStatus::Applied,
                reason: ReasonCode::FullySupported,
                detail: None,
            },
            IntentOutcome {
                intent_id: Uuid::new_v4(),
                intent_type: IntentType::ModelRouting,
                status: TranslationStatus::Rejected,
                reason: ReasonCode::UnsupportedByBackend,
                detail: None,
            },
            IntentOutcome {
                intent_id: Uuid::new_v4(),
                intent_type: IntentType::Compression,
                status: TranslationStatus::Degraded,
                reason: ReasonCode::BackendLimitReached,
                detail: None,
            },
        ],
        created_at: Utc::now(),
    };

    assert_eq!(report.count_by_status(TranslationStatus::Applied), 2);
    assert_eq!(report.count_by_status(TranslationStatus::Rejected), 1);
    assert_eq!(report.count_by_status(TranslationStatus::Degraded), 1);
    assert_eq!(report.count_by_status(TranslationStatus::Ignored), 0);
}

#[test]
fn test_translation_report_send_sync() {
    assert_send_sync::<TranslationStatus>();
    assert_send_sync::<ReasonCode>();
    assert_send_sync::<IntentOutcome>();
    assert_send_sync::<TranslationReport>();
}

// -----------------------------------------------------------------------
// OptimizationIntent::discriminant() tests (Plan 05-01)
// -----------------------------------------------------------------------

#[test]
fn test_discriminant_returns_correct_intent_type() {
    let cases: Vec<(OptimizationIntent, IntentType)> = vec![
        (
            OptimizationIntent::CacheStability(CacheStabilityIntent {
                stability_score: 0.9,
                stable_prefix_end: 100,
                recommended_retention_tier: None,
                scope_label: SharingScope::Session,
                confidence: 0.8,
                evidence_count: 10,
            }),
            IntentType::CacheStability,
        ),
        (
            OptimizationIntent::ContentExtraction(ContentExtractionIntent {
                block_id: "b1".into(),
                variable_pattern: ".*".into(),
                extraction_strategy: "regex".into(),
                scope_label: SharingScope::Tenant,
            }),
            IntentType::ContentExtraction,
        ),
        (
            OptimizationIntent::Serialization(SerializationIntent {
                fanout_width: 2,
                expected_savings_tokens: 100,
                reuse_probability: 0.5,
                added_latency_ms: None,
                scope_label: SharingScope::Global,
            }),
            IntentType::Serialization,
        ),
        (
            OptimizationIntent::Priority(PriorityIntent {
                latency_sensitivity: 0.5,
                workflow_phase: None,
                caller_tier: None,
            }),
            IntentType::Priority,
        ),
        (
            OptimizationIntent::ModelRouting(ModelRoutingIntent {
                model_class: ModelClass::Standard,
                complexity_score: 0.5,
                criticality: 0.5,
                fallback_allowed: true,
            }),
            IntentType::ModelRouting,
        ),
        (
            OptimizationIntent::Placement(PlacementIntent {
                block_id: "b2".into(),
                target: PlacementTarget::CacheablePrefix,
                stability_score: 0.9,
                scope_label: SharingScope::Session,
            }),
            IntentType::Placement,
        ),
        (
            OptimizationIntent::Retention(RetentionIntent {
                recommended_tier: RetentionTier::SessionDuration,
                expected_session_duration_secs: None,
                inter_call_gap_p50_ms: None,
                scope_label: SharingScope::Session,
            }),
            IntentType::Retention,
        ),
        (
            OptimizationIntent::ToolScope(ToolScopeIntent {
                active_tools: vec!["search".into()],
                phase_label: None,
                deferred_tools: vec![],
            }),
            IntentType::ToolScope,
        ),
        (
            OptimizationIntent::Compression(CompressionIntent {
                block_id: "b3".into(),
                compression_ratio: 0.5,
                reversible: true,
                contribution_score: 0.7,
            }),
            IntentType::Compression,
        ),
    ];

    assert_eq!(cases.len(), 9, "must test all 9 intent variants");
    for (intent, expected_type) in &cases {
        assert_eq!(
            intent.discriminant(),
            *expected_type,
            "discriminant mismatch for {intent:?}"
        );
    }
}

// -----------------------------------------------------------------------
// TranslationReport::all_ignored() tests (Plan 05-01)
// -----------------------------------------------------------------------

#[test]
fn test_all_ignored_produces_correct_outcomes() {
    let bundle = OptimizationIntentBundle {
        request_id: Uuid::new_v4(),
        agent_identity: sample_agent_identity(),
        policy_version: "1.0.0".to_string(),
        intents: vec![
            OptimizationIntent::CacheStability(CacheStabilityIntent {
                stability_score: 0.9,
                stable_prefix_end: 100,
                recommended_retention_tier: None,
                scope_label: SharingScope::Session,
                confidence: 0.8,
                evidence_count: 10,
            }),
            OptimizationIntent::Priority(PriorityIntent {
                latency_sensitivity: 0.5,
                workflow_phase: None,
                caller_tier: None,
            }),
            OptimizationIntent::Compression(CompressionIntent {
                block_id: "b1".into(),
                compression_ratio: 0.5,
                reversible: true,
                contribution_score: 0.7,
            }),
        ],
        created_at: Utc::now(),
    };

    let report = TranslationReport::all_ignored(
        &bundle,
        "passthrough",
        ReasonCode::NotRelevant,
        Some("no-op".to_string()),
    );

    assert_eq!(report.request_id, bundle.request_id);
    assert_eq!(report.plugin_id, "passthrough");
    assert_eq!(report.outcomes.len(), 3);

    // Verify each outcome
    assert_eq!(report.outcomes[0].intent_type, IntentType::CacheStability);
    assert_eq!(report.outcomes[1].intent_type, IntentType::Priority);
    assert_eq!(report.outcomes[2].intent_type, IntentType::Compression);

    for outcome in &report.outcomes {
        assert_eq!(outcome.status, TranslationStatus::Ignored);
        assert_eq!(outcome.reason, ReasonCode::NotRelevant);
        assert_eq!(outcome.detail, Some("no-op".to_string()));
    }
}

#[test]
fn test_all_ignored_with_empty_bundle() {
    let bundle = OptimizationIntentBundle {
        request_id: Uuid::new_v4(),
        agent_identity: sample_agent_identity(),
        policy_version: "1.0.0".to_string(),
        intents: vec![],
        created_at: Utc::now(),
    };

    let report =
        TranslationReport::all_ignored(&bundle, "passthrough", ReasonCode::NotRelevant, None);

    assert_eq!(report.outcomes.len(), 0);
    assert_eq!(report.request_id, bundle.request_id);
}
