// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for telemetry in the NeMo Relay adaptive crate.

use super::*;
use chrono::{TimeZone, Utc};
use nemo_relay::codec::response::Usage;
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

fn sample_telemetry_event() -> CacheTelemetryEvent {
    CacheTelemetryEvent {
        request_id: Uuid::new_v4(),
        agent_identity: sample_agent_identity(),
        cache_read_tokens: 800,
        cache_creation_tokens: 200,
        total_prompt_tokens: 1000,
        hit_rate: 0.8,
        miss_reason: None,
        miss_diagnosis: None,
        provider: "anthropic".to_string(),
        timestamp: Utc::now(),
    }
}

fn sample_timestamp() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 4, 10, 0, 0, 0).single().unwrap()
}

// -------------------------------------------------------------------
// Send + Sync compile-time assertions
// -------------------------------------------------------------------

#[test]
fn test_telemetry_types_are_send_sync() {
    assert_send_sync::<CacheRequestFacts>();
    assert_send_sync::<CacheMissDiagnosis>();
    assert_send_sync::<CacheMissEvidence>();
    assert_send_sync::<CacheTelemetryEvent>();
    assert_send_sync::<CacheMissReason>();
    assert_send_sync::<CacheHitRate>();
}

// -------------------------------------------------------------------
// CacheMissReason serde round-trip
// -------------------------------------------------------------------

#[test]
fn test_cache_miss_reason_serde_all_variants() {
    let variants = vec![
        CacheMissReason::PrefixMismatch,
        CacheMissReason::BelowMinimumThreshold,
        CacheMissReason::RetentionExpired,
        CacheMissReason::RoutingMismatch,
        CacheMissReason::Evicted,
        CacheMissReason::UnsupportedFeature,
        CacheMissReason::ColdStart,
        CacheMissReason::Unknown,
        CacheMissReason::Other {
            description: "novel miss reason".to_string(),
        },
    ];

    assert_eq!(
        variants.len(),
        9,
        "must have exactly 9 CacheMissReason variants"
    );

    for reason in &variants {
        let json = serde_json::to_string(reason).unwrap();
        let restored: CacheMissReason = serde_json::from_str(&json).unwrap();
        assert_eq!(&restored, reason);
    }
}

#[test]
fn test_cache_miss_reason_prefix_mismatch_json_shape() {
    let reason = CacheMissReason::PrefixMismatch;
    let json = serde_json::to_string(&reason).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value["reason"], "prefix_mismatch");
}

#[test]
fn test_cache_miss_reason_other_json_shape() {
    let reason = CacheMissReason::Other {
        description: "provider returned code 42".to_string(),
    };
    let json = serde_json::to_string(&reason).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value["reason"], "other");
    assert_eq!(value["description"], "provider returned code 42");
}

// -------------------------------------------------------------------
// CacheHitRate serde round-trip
// -------------------------------------------------------------------

#[test]
fn test_cache_hit_rate_serde() {
    let rate = CacheHitRate {
        hit_rate: 0.75,
        sample_count: 100,
        window_duration_secs: 300.0,
    };
    let json = serde_json::to_string(&rate).unwrap();
    let restored: CacheHitRate = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, rate);
}

// -------------------------------------------------------------------
// CacheTelemetryEvent serde round-trip
// -------------------------------------------------------------------

#[test]
fn test_cache_telemetry_event_serde_full() {
    let mut event = sample_telemetry_event();
    event.miss_reason = Some(CacheMissReason::PrefixMismatch);
    event.miss_diagnosis = Some(CacheMissDiagnosis {
        summary: "Stable prefix diverged at span stable-span-2 before cache reuse.".to_string(),
        recommendation: "Move or extract the mismatching block after the stable prefix."
            .to_string(),
        evidence: CacheMissEvidence::PrefixMismatch {
            first_mismatch_span_id: "stable-span-2".to_string(),
            sequence_index: 2,
            expected_hash_prefix: "sha256:112233445566".to_string(),
            actual_hash_prefix: "sha256:aabbccddeeff".to_string(),
        },
    });

    let json = serde_json::to_string(&event).unwrap();
    let restored: CacheTelemetryEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, event);
}

#[test]
fn test_cache_telemetry_event_serde_without_miss_reason() {
    let event = sample_telemetry_event();
    assert!(event.miss_reason.is_none());
    assert!(event.miss_diagnosis.is_none());

    let json = serde_json::to_string(&event).unwrap();
    assert!(!json.contains("miss_reason"));
    assert!(!json.contains("miss_diagnosis"));

    let restored: CacheTelemetryEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, event);
}

#[test]
fn test_anthropic_cache_telemetry_event_reconstructs_total_prompt_tokens() {
    let usage = Usage {
        prompt_tokens: Some(300),
        completion_tokens: None,
        total_tokens: None,
        cache_read_tokens: Some(500),
        cache_write_tokens: Some(200),
    };

    let event = CacheTelemetryEvent::from_usage(
        Uuid::nil(),
        sample_agent_identity(),
        CacheTelemetryProvider::Anthropic,
        &usage,
        sample_timestamp(),
        None,
    )
    .expect("anthropic usage should build a canonical cache event");

    assert_eq!(event.cache_read_tokens, 500);
    assert_eq!(event.cache_creation_tokens, 200);
    assert_eq!(event.total_prompt_tokens, 1000);
    assert!((event.hit_rate - 0.5).abs() < f64::EPSILON);
    assert_eq!(event.provider, "anthropic");
    assert_eq!(event.miss_reason, None);
}

#[test]
fn test_anthropic_cache_telemetry_event_maps_write_only_zero_read_to_cold_start() {
    let usage = Usage {
        prompt_tokens: Some(300),
        completion_tokens: None,
        total_tokens: None,
        cache_read_tokens: Some(0),
        cache_write_tokens: Some(700),
    };

    let event = CacheTelemetryEvent::from_usage(
        Uuid::nil(),
        sample_agent_identity(),
        CacheTelemetryProvider::Anthropic,
        &usage,
        sample_timestamp(),
        None,
    )
    .expect("anthropic usage should build a canonical cache event");

    assert_eq!(event.total_prompt_tokens, 1000);
    assert!((event.hit_rate - 0.0).abs() < f64::EPSILON);
    assert_eq!(event.miss_reason, Some(CacheMissReason::ColdStart));
}

#[test]
fn test_anthropic_cache_telemetry_event_returns_none_without_prompt_tokens() {
    let usage = Usage {
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        cache_read_tokens: Some(500),
        cache_write_tokens: Some(200),
    };

    let event = CacheTelemetryEvent::from_usage(
        Uuid::nil(),
        sample_agent_identity(),
        CacheTelemetryProvider::Anthropic,
        &usage,
        sample_timestamp(),
        None,
    );

    assert_eq!(event, None);
}

#[test]
fn test_openai_cache_telemetry_event_normalizes_creation_tokens_to_zero() {
    let usage = Usage {
        prompt_tokens: Some(1000),
        completion_tokens: None,
        total_tokens: None,
        cache_read_tokens: Some(600),
        cache_write_tokens: Some(999),
    };

    let event = CacheTelemetryEvent::from_usage(
        Uuid::nil(),
        sample_agent_identity(),
        CacheTelemetryProvider::OpenAI,
        &usage,
        sample_timestamp(),
        None,
    )
    .expect("openai usage should build a canonical cache event");

    assert_eq!(event.cache_creation_tokens, 0);
    assert_eq!(event.total_prompt_tokens, 1000);
    assert!((event.hit_rate - 0.6).abs() < f64::EPSILON);
    assert_eq!(event.provider, "openai");
    assert_eq!(event.miss_reason, None);
}

#[test]
fn test_openai_cache_telemetry_event_maps_zero_read_to_unknown() {
    let usage = Usage {
        prompt_tokens: Some(1000),
        completion_tokens: None,
        total_tokens: None,
        cache_read_tokens: Some(0),
        cache_write_tokens: Some(999),
    };

    let event = CacheTelemetryEvent::from_usage(
        Uuid::nil(),
        sample_agent_identity(),
        CacheTelemetryProvider::OpenAI,
        &usage,
        sample_timestamp(),
        None,
    )
    .expect("openai usage should build a canonical cache event");

    assert_eq!(event.cache_creation_tokens, 0);
    assert_eq!(event.miss_reason, Some(CacheMissReason::Unknown));
    assert_eq!(
        event
            .miss_diagnosis
            .expect("unknown misses should include diagnosis")
            .evidence,
        CacheMissEvidence::Unknown {
            missing_facts: vec!["request_facts_unavailable".to_string()],
        }
    );
}

#[test]
fn telemetry_observability_keeps_request_facts_optional_for_anthropic_unknown_misses() {
    let usage = Usage {
        prompt_tokens: Some(900),
        completion_tokens: None,
        total_tokens: None,
        cache_read_tokens: Some(0),
        cache_write_tokens: Some(0),
    };

    let event = CacheTelemetryEvent::from_usage(
        Uuid::nil(),
        sample_agent_identity(),
        CacheTelemetryProvider::Anthropic,
        &usage,
        sample_timestamp(),
        None,
    )
    .expect("usage-only observability should still build telemetry without request facts");

    assert_eq!(event.provider, "anthropic");
    assert_eq!(event.miss_reason, Some(CacheMissReason::Unknown));
    let diagnosis = event
        .miss_diagnosis
        .expect("unknown misses should keep a bounded diagnosis");
    assert_eq!(
        diagnosis.summary,
        "Cache miss could not be classified from the available request facts."
    );
    assert_eq!(
        diagnosis.evidence,
        CacheMissEvidence::Unknown {
            missing_facts: vec!["request_facts_unavailable".to_string()],
        }
    );
}

#[test]
fn test_from_usage_uses_prefix_mismatch_diagnosis_when_request_facts_are_available() {
    let usage = Usage {
        prompt_tokens: Some(1000),
        completion_tokens: None,
        total_tokens: None,
        cache_read_tokens: Some(0),
        cache_write_tokens: Some(0),
    };
    let request_facts = CacheRequestFacts {
        provider: "openai".to_string(),
        stable_prefix_length: 3,
        stable_prefix_tokens: Some(1400),
        required_min_tokens: Some(1024),
        first_mismatch_span_id: Some("stable-span-3".to_string()),
        first_mismatch_sequence_index: Some(3),
        expected_hash_prefix: Some("sha256:1234567890abcdef".to_string()),
        actual_hash_prefix: Some("sha256:fedcba0987654321".to_string()),
        retention_window_secs: None,
        observed_gap_secs: None,
        missing_facts: vec![],
    };

    let event = CacheTelemetryEvent::from_usage(
        Uuid::nil(),
        sample_agent_identity(),
        CacheTelemetryProvider::OpenAI,
        &usage,
        sample_timestamp(),
        Some(&request_facts),
    )
    .expect("request facts should produce a canonical event");

    assert_eq!(event.miss_reason, Some(CacheMissReason::PrefixMismatch));
    assert_eq!(
        event.miss_diagnosis,
        Some(CacheMissDiagnosis {
            summary: "Stable prefix diverged at span stable-span-3 before cache reuse.".to_string(),
            recommendation: "Move or extract the mismatching block after the stable prefix."
                .to_string(),
            evidence: CacheMissEvidence::PrefixMismatch {
                first_mismatch_span_id: "stable-span-3".to_string(),
                sequence_index: 3,
                expected_hash_prefix: "sha256:1234567890ab".to_string(),
                actual_hash_prefix: "sha256:fedcba098765".to_string(),
            },
        })
    );
}

#[test]
fn test_cache_miss_diagnosis_prefix_mismatch_is_bounded_and_serialized() {
    let request_facts = CacheRequestFacts {
        provider: "openai".to_string(),
        stable_prefix_length: 3,
        stable_prefix_tokens: Some(1536),
        required_min_tokens: Some(1024),
        first_mismatch_span_id: Some("stable-span-2".to_string()),
        first_mismatch_sequence_index: Some(2),
        expected_hash_prefix: Some("sha256:112233445566".to_string()),
        actual_hash_prefix: Some("sha256:aabbccddeeff".to_string()),
        retention_window_secs: None,
        observed_gap_secs: None,
        missing_facts: vec![],
    };

    let event = CacheTelemetryEvent::from_usage(
        Uuid::nil(),
        sample_agent_identity(),
        CacheTelemetryProvider::OpenAI,
        &Usage {
            prompt_tokens: Some(1536),
            completion_tokens: None,
            total_tokens: None,
            cache_read_tokens: Some(0),
            cache_write_tokens: Some(0),
        },
        sample_timestamp(),
        Some(&request_facts),
    )
    .expect("cache event should be created");

    assert_eq!(event.miss_reason, Some(CacheMissReason::PrefixMismatch));
    let diagnosis = event
        .miss_diagnosis
        .as_ref()
        .expect("prefix mismatch should include diagnosis");
    assert_eq!(
        diagnosis.recommendation,
        "Move or extract the mismatching block after the stable prefix."
    );
    assert!(
        diagnosis.summary.lines().count() == 1,
        "summary must stay single-line"
    );
    assert!(matches!(
        &diagnosis.evidence,
        CacheMissEvidence::PrefixMismatch {
            first_mismatch_span_id,
            sequence_index: 2,
            expected_hash_prefix,
            actual_hash_prefix,
        } if first_mismatch_span_id == "stable-span-2"
            && expected_hash_prefix == "sha256:112233445566"
            && actual_hash_prefix == "sha256:aabbccddeeff"
    ));

    let json = serde_json::to_string(&event).expect("event should serialize");
    let value: serde_json::Value = serde_json::from_str(&json).expect("event JSON should parse");
    let evidence_json = serde_json::to_string(&value["miss_diagnosis"]["evidence"])
        .expect("evidence should serialize");
    assert!(json.contains("\"miss_diagnosis\""));
    assert!(evidence_json.contains("sha256:"));
    assert!(!evidence_json.contains("You are"));
    assert!(!evidence_json.contains("search"));
}

#[test]
fn test_cache_miss_diagnosis_below_minimum_threshold_reports_exact_token_counts() {
    let request_facts = CacheRequestFacts {
        provider: "openai".to_string(),
        stable_prefix_length: 2,
        stable_prefix_tokens: Some(768),
        required_min_tokens: Some(1024),
        first_mismatch_span_id: None,
        first_mismatch_sequence_index: None,
        expected_hash_prefix: None,
        actual_hash_prefix: None,
        retention_window_secs: None,
        observed_gap_secs: None,
        missing_facts: vec![],
    };

    let event = CacheTelemetryEvent::from_usage(
        Uuid::nil(),
        sample_agent_identity(),
        CacheTelemetryProvider::OpenAI,
        &Usage {
            prompt_tokens: Some(768),
            completion_tokens: None,
            total_tokens: None,
            cache_read_tokens: Some(0),
            cache_write_tokens: Some(0),
        },
        sample_timestamp(),
        Some(&request_facts),
    )
    .expect("cache event should be created");

    assert_eq!(
        event.miss_reason,
        Some(CacheMissReason::BelowMinimumThreshold)
    );
    let diagnosis = event
        .miss_diagnosis
        .as_ref()
        .expect("threshold miss should include diagnosis");
    assert_eq!(
        diagnosis.recommendation,
        "Increase the cacheable prefix above the provider minimum or stop expecting a hit."
    );
    assert!(matches!(
        &diagnosis.evidence,
        CacheMissEvidence::BelowMinimumThreshold {
            observed_prefix_tokens: 768,
            required_min_tokens: 1024,
            ..
        }
    ));
}

#[test]
fn test_cache_miss_diagnosis_retention_expired_reports_gap_and_window() {
    let request_facts = CacheRequestFacts {
        provider: "anthropic".to_string(),
        stable_prefix_length: 4,
        stable_prefix_tokens: Some(2048),
        required_min_tokens: Some(1024),
        first_mismatch_span_id: None,
        first_mismatch_sequence_index: None,
        expected_hash_prefix: None,
        actual_hash_prefix: None,
        retention_window_secs: Some(300.0),
        observed_gap_secs: Some(480.0),
        missing_facts: vec![],
    };

    let event = CacheTelemetryEvent::from_usage(
        Uuid::nil(),
        sample_agent_identity(),
        CacheTelemetryProvider::Anthropic,
        &Usage {
            prompt_tokens: Some(2048),
            completion_tokens: None,
            total_tokens: None,
            cache_read_tokens: Some(0),
            cache_write_tokens: Some(0),
        },
        sample_timestamp(),
        Some(&request_facts),
    )
    .expect("cache event should be created");

    assert_eq!(event.miss_reason, Some(CacheMissReason::RetentionExpired));
    let diagnosis = event
        .miss_diagnosis
        .as_ref()
        .expect("retention miss should include diagnosis");
    assert_eq!(
        diagnosis.recommendation,
        "Reuse the stable prefix inside the active retention window or accept a cold rebuild."
    );
    assert!(matches!(
        &diagnosis.evidence,
        CacheMissEvidence::RetentionExpired {
            observed_gap_secs,
            retention_window_secs,
            ..
        } if (observed_gap_secs - 480.0).abs() < f64::EPSILON
            && (retention_window_secs - 300.0).abs() < f64::EPSILON
    ));
}

#[test]
fn test_cache_miss_diagnosis_unknown_preserves_missing_facts() {
    let request_facts = CacheRequestFacts {
        provider: "openai".to_string(),
        stable_prefix_length: 1,
        stable_prefix_tokens: None,
        required_min_tokens: None,
        first_mismatch_span_id: None,
        first_mismatch_sequence_index: None,
        expected_hash_prefix: None,
        actual_hash_prefix: None,
        retention_window_secs: None,
        observed_gap_secs: None,
        missing_facts: vec![
            "stable_prefix_tokens_unavailable".to_string(),
            "expected_hash_prefix_unavailable".to_string(),
        ],
    };

    let event = CacheTelemetryEvent::from_usage(
        Uuid::nil(),
        sample_agent_identity(),
        CacheTelemetryProvider::OpenAI,
        &Usage {
            prompt_tokens: Some(256),
            completion_tokens: None,
            total_tokens: None,
            cache_read_tokens: Some(0),
            cache_write_tokens: Some(0),
        },
        sample_timestamp(),
        Some(&request_facts),
    )
    .expect("cache event should be created");

    assert_eq!(event.miss_reason, Some(CacheMissReason::Unknown));
    let diagnosis = event
        .miss_diagnosis
        .as_ref()
        .expect("unknown miss should include diagnosis");
    assert_eq!(
        diagnosis.recommendation,
        "Capture request facts earlier or keep the miss classified as unknown."
    );
    assert!(matches!(
        &diagnosis.evidence,
        CacheMissEvidence::Unknown { missing_facts }
            if missing_facts
                == &vec![
                    "stable_prefix_tokens_unavailable".to_string(),
                    "expected_hash_prefix_unavailable".to_string(),
                ]
    ));
}

#[test]
fn test_no_write_anthropic_cache_miss_diagnosis_uses_threshold_facts_without_local_math() {
    let request_facts = CacheRequestFacts {
        provider: "anthropic".to_string(),
        stable_prefix_length: 2,
        stable_prefix_tokens: Some(1600),
        required_min_tokens: Some(2048),
        first_mismatch_span_id: None,
        first_mismatch_sequence_index: None,
        expected_hash_prefix: None,
        actual_hash_prefix: None,
        retention_window_secs: Some(300.0),
        observed_gap_secs: None,
        missing_facts: vec![],
    };

    let event = CacheTelemetryEvent::from_usage(
        Uuid::nil(),
        sample_agent_identity(),
        CacheTelemetryProvider::Anthropic,
        &Usage {
            prompt_tokens: Some(1600),
            completion_tokens: None,
            total_tokens: None,
            cache_read_tokens: Some(0),
            cache_write_tokens: Some(0),
        },
        sample_timestamp(),
        Some(&request_facts),
    )
    .expect("no-write anthropic usage should still build a canonical event");

    assert_eq!(
        event.miss_reason,
        Some(CacheMissReason::BelowMinimumThreshold)
    );
    assert_eq!(event.cache_creation_tokens, 0);
    assert_eq!(event.total_prompt_tokens, 1600);
}

#[test]
fn test_anthropic_multi_breakpoint_telemetry_event_uses_normalized_usage_totals() {
    let request_facts = CacheRequestFacts {
        provider: "anthropic".to_string(),
        stable_prefix_length: 3,
        stable_prefix_tokens: Some(2600),
        required_min_tokens: Some(1024),
        first_mismatch_span_id: None,
        first_mismatch_sequence_index: None,
        expected_hash_prefix: None,
        actual_hash_prefix: None,
        retention_window_secs: Some(300.0),
        observed_gap_secs: None,
        missing_facts: vec![],
    };

    let event = CacheTelemetryEvent::from_usage(
        Uuid::nil(),
        sample_agent_identity(),
        CacheTelemetryProvider::Anthropic,
        &Usage {
            prompt_tokens: Some(1800),
            completion_tokens: None,
            total_tokens: None,
            cache_read_tokens: Some(900),
            cache_write_tokens: Some(600),
        },
        sample_timestamp(),
        Some(&request_facts),
    )
    .expect("profitable multi-breakpoint anthropic usage should build a canonical event");

    assert_eq!(event.cache_read_tokens, 900);
    assert_eq!(event.cache_creation_tokens, 600);
    assert_eq!(event.total_prompt_tokens, 3300);
    assert!((event.hit_rate - (900.0 / 3300.0)).abs() < f64::EPSILON);
    assert_eq!(event.miss_reason, None);
    assert_eq!(event.miss_diagnosis, None);

    let json = serde_json::to_string(&event).expect("event should serialize");
    assert!(!json.contains("You are a careful planner"));
    assert!(!json.contains("Summarize the latest findings"));
}

// -------------------------------------------------------------------
// compute_hit_rate
// -------------------------------------------------------------------

#[test]
fn test_compute_hit_rate_normal() {
    let rate = CacheTelemetryEvent::compute_hit_rate(800, 1000);
    assert!((rate - 0.8).abs() < f64::EPSILON);
}

#[test]
fn test_compute_hit_rate_zero_total() {
    let rate = CacheTelemetryEvent::compute_hit_rate(0, 0);
    assert!((rate - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_compute_hit_rate_full_cache() {
    let rate = CacheTelemetryEvent::compute_hit_rate(500, 500);
    assert!((rate - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_compute_hit_rate_no_cache() {
    let rate = CacheTelemetryEvent::compute_hit_rate(0, 1000);
    assert!((rate - 0.0).abs() < f64::EPSILON);
}
