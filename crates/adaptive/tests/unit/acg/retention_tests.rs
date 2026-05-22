// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for retention in the NeMo Relay adaptive crate.

use crate::acg::profile::DistributionSummary;
use crate::acg::retention::{
    RetentionThresholds, generate_retention_intent, generate_retention_intent_default,
};
use crate::acg::types::{RetentionTier, SharingScope};

fn distribution(p50: f64) -> DistributionSummary {
    DistributionSummary {
        p50,
        p90: p50,
        p99: p50,
        sample_count: 8,
    }
}

#[test]
fn retention_threshold_defaults_match_expected_buckets() {
    let thresholds = RetentionThresholds::default();

    assert_eq!(thresholds.ephemeral_max_secs, 5.0);
    assert_eq!(thresholds.short_lived_max_secs, 60.0);
    assert_eq!(thresholds.session_duration_max_secs, 600.0);
    assert_eq!(thresholds.long_lived_max_secs, 3600.0);
}

#[test]
fn generate_retention_intent_maps_each_threshold_bucket() {
    let thresholds = RetentionThresholds::default();

    let cases = [
        (4.0, RetentionTier::Ephemeral),
        (30.0, RetentionTier::ShortLived),
        (120.0, RetentionTier::SessionDuration),
        (1800.0, RetentionTier::LongLived),
        (7200.0, RetentionTier::Permanent),
    ];

    for (session_p50, expected_tier) in cases {
        let intent =
            generate_retention_intent(&distribution(session_p50), &distribution(1.25), &thresholds);
        assert_eq!(intent.recommended_tier, expected_tier);
        assert_eq!(intent.expected_session_duration_secs, Some(session_p50));
        assert_eq!(intent.inter_call_gap_p50_ms, Some(1250.0));
        assert_eq!(intent.scope_label, SharingScope::Session);
    }
}

#[test]
fn generate_retention_intent_default_uses_default_thresholds() {
    let intent = generate_retention_intent_default(&distribution(40.0), &distribution(0.5));

    assert_eq!(intent.recommended_tier, RetentionTier::ShortLived);
    assert_eq!(intent.inter_call_gap_p50_ms, Some(500.0));
}
