// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn network_rate_requires_a_baseline_and_rejects_counter_reset() {
    let now = Instant::now();
    assert!(rate(10, None, |sample| sample.received, now).is_none());
    let prior = Counters {
        received: 100,
        transmitted: 0,
        received_packets: 0,
        transmitted_packets: 0,
        receive_errors: 0,
        transmit_errors: 0,
        sampled_at: now,
    };
    let later = now + std::time::Duration::from_secs(2);
    assert_eq!(
        rate(200, Some(prior), |sample| sample.received, later)
            .unwrap()
            .value,
        50.0
    );
    assert!(rate(50, Some(prior), |sample| sample.received, later).is_none());
}

#[test]
fn network_scope_is_global_in_every_process_mode() {
    let mut sampler = NetworkSampler::default();
    let snapshot = sampler.sample(&[]);
    assert_eq!(snapshot.measurement_scope, ResourceMeasurementScope::Global);
    assert!(
        snapshot
            .interfaces
            .windows(2)
            .all(|pair| pair[0].name <= pair[1].name)
    );
    if let Some(first) = snapshot.interfaces.first() {
        let selected = sampler.sample(std::slice::from_ref(&first.name));
        assert_eq!(selected.interfaces.len(), 1);
        assert_eq!(selected.interfaces[0].name, first.name);
        assert_eq!(
            selected.system.received_data,
            selected.interfaces[0].traffic.received_data
        );
        assert_eq!(
            selected.system.transmitted_packets,
            selected.interfaces[0].traffic.transmitted_packets
        );
    }
    let missing = sampler.sample(&["__nemo_relay_missing_interface__".into()]);
    assert!(missing.interfaces.is_empty());
    assert!(missing.system.received_data.is_none());
}

#[test]
fn network_rates_reject_zero_and_reversed_intervals() {
    let now = Instant::now();
    let previous = Counters {
        received: 1,
        transmitted: 0,
        received_packets: 0,
        transmitted_packets: 0,
        receive_errors: 0,
        transmit_errors: 0,
        sampled_at: now,
    };
    assert!(rate(2, Some(previous), |sample| sample.received, now).is_none());
    assert!(
        rate(
            2,
            Some(previous),
            |sample| sample.received,
            now - std::time::Duration::from_secs(1)
        )
        .is_none()
    );
}
