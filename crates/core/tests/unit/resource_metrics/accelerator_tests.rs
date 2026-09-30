// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use super::{
    DrmClient, DrmEngineCounter, SamplingState, aggregate_drm_memory, drm_client_utilization,
    parse_drm_client, parse_drm_memory,
};
use nemo_relay_types::api::resource_metrics::{AcceleratorVendor, ResourceMeasurementUnit};

#[test]
fn drm_memory_units_normalize_to_kibibytes() {
    assert_eq!(
        parse_drm_memory("2 MiB"),
        Some((2_048, ResourceMeasurementUnit::Kibibytes))
    );
    assert_eq!(
        parse_drm_memory("2 KiB"),
        Some((2, ResourceMeasurementUnit::Kibibytes))
    );
    assert_eq!(
        parse_drm_memory("2048 bytes"),
        Some((2, ResourceMeasurementUnit::Kibibytes))
    );
    assert_eq!(parse_drm_memory("18446744073709551615 MiB"), None);
    assert_eq!(
        parse_drm_memory("2048"),
        Some((2, ResourceMeasurementUnit::Kibibytes))
    );
}

#[test]
fn drm_process_utilization_uses_counter_deltas() {
    let mut sampling_state = SamplingState::default();
    let sampled_at = Instant::now();
    let mut client = DrmClient {
        identity: "resource-metrics-unit-test-client".into(),
        vendor: AcceleratorVendor::Amd,
        device_identifier: "test-device".into(),
        memory_regions: Vec::new(),
        engine_counters: vec![(
            "render".into(),
            DrmEngineCounter::Cycles {
                busy: 1_000,
                total: 2_000,
            },
        )],
    };
    const TEST_PROCESS_ID: u32 = u32::MAX;
    const TEST_START_IDENTITY: u64 = 1;
    assert_eq!(
        drm_client_utilization(
            TEST_PROCESS_ID,
            TEST_START_IDENTITY,
            &client,
            &mut sampling_state,
            sampled_at,
        ),
        None
    );

    client.engine_counters[0].1 = DrmEngineCounter::Cycles {
        busy: 1_050,
        total: 2_200,
    };
    assert_eq!(
        drm_client_utilization(
            TEST_PROCESS_ID,
            TEST_START_IDENTITY,
            &client,
            &mut sampling_state,
            sampled_at + Duration::from_secs(1),
        ),
        Some(25.0)
    );

    client.engine_counters[0].1 = DrmEngineCounter::Cycles {
        busy: 1_040,
        total: 2_300,
    };
    assert_eq!(
        drm_client_utilization(
            TEST_PROCESS_ID,
            TEST_START_IDENTITY,
            &client,
            &mut sampling_state,
            sampled_at + Duration::from_secs(2),
        ),
        None
    );
    client.engine_counters[0].1 = DrmEngineCounter::Cycles {
        busy: 1_100,
        total: 2_400,
    };
    assert_eq!(
        drm_client_utilization(
            TEST_PROCESS_ID,
            TEST_START_IDENTITY,
            &client,
            &mut sampling_state,
            sampled_at + Duration::from_secs(3),
        ),
        Some(60.0)
    );
}

#[test]
fn drm_engine_time_uses_elapsed_time_and_separate_sampling_state() {
    let mut polling = SamplingState::default();
    let mut on_demand = SamplingState::default();
    let sampled_at = Instant::now();
    let mut client = parse_drm_client(
        "drm-driver: i915\ndrm-client-id: 7\ndrm-pdev: 0000:00:02.0\n\
         drm-engine-render: 1000000000 ns\ndrm-engine-capacity-render: 2\n",
    )
    .unwrap();
    assert_eq!(
        drm_client_utilization(42, 1, &client, &mut polling, sampled_at),
        None
    );
    assert_eq!(
        drm_client_utilization(42, 1, &client, &mut on_demand, sampled_at),
        None
    );
    client.engine_counters[0].1 = DrmEngineCounter::BusyNanoseconds {
        busy: 2_000_000_000,
        capacity: 2,
    };
    assert_eq!(
        drm_client_utilization(
            42,
            1,
            &client,
            &mut polling,
            sampled_at + Duration::from_secs(1),
        ),
        Some(50.0)
    );
    assert_eq!(
        drm_client_utilization(
            42,
            1,
            &client,
            &mut on_demand,
            sampled_at + Duration::from_secs(1),
        ),
        Some(50.0)
    );
}

#[test]
fn drm_parser_reads_current_resident_memory() {
    let client = parse_drm_client(
        "drm-driver: xe\ndrm-client-id: 3\ndrm-pdev: 0000:03:00.0\n\
         drm-resident-vram0: 23992 KiB\n\
         drm-resident-gtt: 192 KiB\ndrm-cycles-rcs: 28257900\n\
         drm-total-cycles-rcs: 7655183225\n",
    )
    .unwrap();
    assert_eq!(
        aggregate_drm_memory(&[client]),
        Some((24_184, ResourceMeasurementUnit::Kibibytes))
    );
}

#[test]
fn drm_memory_aggregates_only_consistent_units() {
    let client = DrmClient {
        identity: "test-client".into(),
        vendor: AcceleratorVendor::Amd,
        device_identifier: "test-device".into(),
        memory_regions: vec![(2, ResourceMeasurementUnit::Kibibytes)],
        engine_counters: Vec::new(),
    };
    assert_eq!(
        aggregate_drm_memory(&[client]),
        Some((2, ResourceMeasurementUnit::Kibibytes))
    );
}

#[test]
fn drm_descriptor_filter_accepts_device_numbers_with_any_minor() {
    // Linux DRM has major 226, with both primary and render-node minors.
    for minor in [0_u64, 128, 255, 256, 65_536] {
        let device = (226 << 8) | (minor & 0xff) | ((minor & !0xff) << 12);
        assert!(super::is_drm_device_number(device));
    }
    assert!(!super::is_drm_device_number(1 << 8));
    assert!(!super::is_drm_device_number((226 << 8) | (1 << 44)));
}

#[test]
fn drm_descriptor_filter_skips_regular_files_and_non_drm_devices() {
    let file = tempfile::NamedTempFile::new().unwrap();
    assert!(!super::is_drm_descriptor(file.path()));
    assert!(!super::is_drm_descriptor(std::path::Path::new("/dev/null")));
    assert!(!super::is_drm_descriptor(std::path::Path::new(
        "/proc/self/fd/-1"
    )));
}
