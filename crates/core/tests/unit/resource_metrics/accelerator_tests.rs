// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use super::{
    DrmClient, DrmEngineCounter, SamplingState, aggregate_drm_memory, drm_client_utilization,
    parse_drm_client, parse_drm_memory,
};
use nemo_relay_types::api::resource_metrics::{AcceleratorVendor, CapacityUnit, UtilizationUnit};

#[test]
fn drm_memory_units_normalize_to_kibibytes() {
    assert_eq!(
        parse_drm_memory("2 MiB"),
        Some((2_048, CapacityUnit::Kibibytes))
    );
    assert_eq!(
        parse_drm_memory("2 KiB"),
        Some((2, CapacityUnit::Kibibytes))
    );
    assert_eq!(
        parse_drm_memory("2048 bytes"),
        Some((2, CapacityUnit::Kibibytes))
    );
    assert_eq!(parse_drm_memory("18446744073709551615 MiB"), None);
    assert_eq!(parse_drm_memory("2048"), Some((2, CapacityUnit::Kibibytes)));
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
        Some((24_184, CapacityUnit::Kibibytes))
    );
}

#[test]
fn drm_memory_aggregates_only_consistent_units() {
    let client = DrmClient {
        identity: "test-client".into(),
        vendor: AcceleratorVendor::Amd,
        device_identifier: "test-device".into(),
        memory_regions: vec![(2, CapacityUnit::Kibibytes)],
        engine_counters: Vec::new(),
    };
    assert_eq!(
        aggregate_drm_memory(&[client]),
        Some((2, CapacityUnit::Kibibytes))
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
#[test]
fn drm_device_files_normalize_amd_and_intel_memory_and_apply_selectors() {
    use crate::plugins::resource_metrics::config::ResourceMetricsGpuConfig;
    use std::fs;
    let root = tempfile::tempdir().unwrap();
    for (card, vendor, memory, utilization) in [
        ("card0", "0x1002", "4096", "25"),
        ("card1", "0x8086", "8192", "50"),
        ("card2", "0x1002", "invalid", "101"),
        ("card3", "0x10de", "4096", "25"),
    ] {
        // sysfs card/device entries resolve to distinct PCI device directories.
        let device = root.path().join(format!("0000:0{}:00.0", &card[4..]));
        fs::create_dir_all(&device).unwrap();
        let card_path = root.path().join(card);
        fs::create_dir_all(&card_path).unwrap();
        std::os::unix::fs::symlink(&device, card_path.join("device")).unwrap();
        for (name, value) in [
            ("vendor", vendor),
            ("mem_info_vram_used", memory),
            ("gpu_busy_percent", utilization),
        ] {
            fs::write(device.join(name), value).unwrap();
        }
    }
    let config = ResourceMetricsGpuConfig::default();
    let mut devices = super::collect_linux_drm_devices_from(root.path(), &config);
    devices.sort_by_key(|device| device.vendor == AcceleratorVendor::Intel);
    assert_eq!(devices.len(), 2);
    assert_eq!(devices[0].vendor, AcceleratorVendor::Amd);
    assert_eq!(devices[0].memory_used.as_ref().unwrap().value, 4_u64);
    assert_eq!(devices[0].compute_utilization.as_ref().unwrap().value, 25.0);
    assert_eq!(devices[1].vendor, AcceleratorVendor::Intel);
    assert_eq!(devices[1].memory_used.as_ref().unwrap().value, 8_u64);
    assert!(devices[1].compute_utilization.is_none());
    let config = ResourceMetricsGpuConfig {
        devices: vec!["unselected-device".into()],
        ..Default::default()
    };
    assert!(super::collect_linux_drm_devices_from(root.path(), &config).is_empty());
    assert!(
        super::collect_linux_drm_devices_from(&root.path().join("missing"), &config).is_empty()
    );
}
#[test]
fn drm_process_records_group_clients_by_device_and_keep_the_busiest_engine() {
    use std::collections::{BTreeMap, BTreeSet};
    let clients = |busy: u64| {
        let mut clients = BTreeMap::new();
        for (id, memory, time) in [(1, 2, busy), (2, 4, busy / 2)] {
            let client = parse_drm_client(&format!(
                "drm-driver: amdgpu\ndrm-client-id: {id}\ndrm-pdev: fixture\ndrm-resident-vram: {memory} KiB\ndrm-engine-render: {time} ns\n"
            )).unwrap();
            clients.insert(client.identity.clone(), client);
        }
        clients
    };
    let now = Instant::now();
    let mut state = SamplingState::default();
    let mut live = BTreeSet::new();
    let first = super::drm_process_records(42, 1, clients(100_000_000), now, &mut state, &mut live);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].process_id, 42);
    assert_eq!(first[0].memory_used.as_ref().unwrap().value, 6_u64);
    assert!(first[0].compute_utilization.is_none());
    assert_eq!(live.len(), 2);
    let second = super::drm_process_records(
        42,
        1,
        clients(600_000_000),
        now + Duration::from_secs(1),
        &mut state,
        &mut live,
    );
    let utilization = second[0].compute_utilization.as_ref().unwrap();
    assert_eq!(utilization.value, 50.0);
    assert_eq!(utilization.unit, UtilizationUnit::Percentage);
    assert_eq!(second[0].device_identifier, "fixture");
    assert!(
        super::drm_process_records(42, 1, BTreeMap::new(), now, &mut state, &mut live).is_empty()
    );
}
