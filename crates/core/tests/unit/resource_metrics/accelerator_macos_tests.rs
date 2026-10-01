// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::time::Duration;

#[test]
fn registry_device_statistics_normalize_vendor_memory_and_utilization() {
    let output = r#"
+-o AGXAccelerator <class AGXAccelerator, id 0x1, registered>
  "PerformanceStatistics" = {"In use system memory"=4096,"Device Utilization %"=25}
+-o AMDRadeonAccelerator <class AMDAccelerator, id 0x2, registered>
  "PerformanceStatistics" = {"inUseVidMemoryBytes"=8192,"Device Utilization %"=50}
+-o IntelAccelerator <class IntelAccelerator, id 0x3, registered>
  "PerformanceStatistics" = {"inUseSysMemoryBytes"=2048,"Device Utilization %"=101}
+-o Unknown <class Unknown, id 0x4, registered>
+-o AGXAccelerator <class AGXAccelerator, id 0xinvalid, registered>
"#;
    let devices = macos_devices(output);
    assert_eq!(devices.len(), 3);
    for (index, (vendor, memory)) in [
        (AcceleratorVendor::Apple, 4_u64),
        (AcceleratorVendor::Amd, 8),
        (AcceleratorVendor::Intel, 2),
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(devices[index].vendor, vendor);
        assert_eq!(devices[index].device_index, Some(index as u32));
        let measurement = devices[index].memory_used.as_ref().unwrap();
        assert_eq!(measurement.unit, CapacityUnit::Kibibytes);
        assert_eq!(measurement.value, memory);
    }
    assert_eq!(devices[0].compute_utilization.as_ref().unwrap().value, 25.0);
    assert!(devices[2].compute_utilization.is_none());
    assert!(registry_number(r#""memory"=18446744073709551616"#, "memory").is_none());
    assert!(registry_number(r#""memory"=invalid"#, "memory").is_none());
}

fn client(id: u64, pid: u32, busy: u64) -> String {
    format!(
        "+-o AGXDeviceUserClient <class AGXDeviceUserClient, id 0x{id:x}, registered>\n\"IOUserClientCreator\" = \"pid {pid}, client\"\n\"accumulatedGPUTime\"={busy}\n"
    )
}

#[test]
fn apple_process_rates_use_owned_clients_and_reset_and_remove_baselines() {
    let mut state = SamplingState::default();
    let now = Instant::now();
    let devices = [("macos:0x1".to_owned(), Some(0))];
    let first = client(1, 42, 1_000_000_000) + &client(2, 42, 2_000_000_000) + &client(3, 99, 10);
    let sample = apple_processes_from_registry(&first, &[42], &devices, &mut state, now).unwrap();
    assert_eq!(sample.len(), 1);
    assert!(sample[0].compute_utilization.is_none());
    assert!(sample[0].memory_used.is_none());
    let next = client(1, 42, 1_250_000_000) + &client(2, 42, 2_250_000_000);
    let sample = apple_processes_from_registry(
        &next,
        &[42],
        &devices,
        &mut state,
        now + Duration::from_secs(1),
    )
    .unwrap();
    assert_eq!(sample[0].compute_utilization.as_ref().unwrap().value, 50.0);
    assert_eq!(sample[0].process_id, 42);
    let reset = client(1, 42, 1);
    let sample = apple_processes_from_registry(
        &reset,
        &[42],
        &devices,
        &mut state,
        now + Duration::from_secs(2),
    )
    .unwrap();
    assert!(sample[0].compute_utilization.is_none());
    assert_eq!(state.client_baselines.len(), 1);
    assert!(
        apple_processes_from_registry(
            "",
            &[42],
            &devices,
            &mut state,
            now + Duration::from_secs(3)
        )
        .unwrap()
        .is_empty()
    );
    assert!(state.client_baselines.is_empty());
    assert!(apple_processes_from_registry(&first, &[42], &[], &mut state, now).is_none());
    let invalid = " +-o AGXDeviceUserClient <id 0xbad, registered>\n\"IOUserClientCreator\" = \"pid invalid, test\"\n";
    assert!(
        apple_processes_from_registry(invalid, &[42], &devices, &mut state, now)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn registry_collection_respects_device_selectors_and_disabled_groups() {
    let output = "+-o AGXAccelerator <class AGXAccelerator, id 0x1, registered>\n\"In use system memory\"=4096\n";
    for selector in ["macos:0x1", "0", "missing"] {
        for (device_metrics, process_metrics) in
            [(true, true), (false, true), (true, false), (false, false)]
        {
            let config = ResourceMetricsGpuConfig {
                devices: vec![selector.into()],
                device_metrics,
                process_metrics,
                ..Default::default()
            };
            let mut calls = 0;
            let result = collect_from_registry(
                &[42],
                &config,
                &mut SamplingState::default(),
                output,
                || {
                    calls += 1;
                    Some(client(1, 42, 100))
                },
            );
            assert_eq!(result.devices.is_some(), device_metrics);
            if let Some(devices) = result.devices {
                assert_eq!(devices.len(), usize::from(selector != "missing"));
            }
            assert_eq!(
                result.processes.is_some(),
                process_metrics && selector != "missing"
            );
            assert_eq!(calls, usize::from(process_metrics && selector != "missing"));
        }
    }
    let multiple =
        format!("{output}+-o AMDRadeonAccelerator <class AMDAccelerator, id 0x2, registered>\n");
    let result = collect_from_registry(
        &[42],
        &ResourceMetricsGpuConfig::default(),
        &mut SamplingState::default(),
        &multiple,
        || panic!("ambiguous process attribution must not query user clients"),
    );
    assert!(result.processes.is_none());
    assert_eq!(result.devices.unwrap().len(), 1);
    let result = collect_from_registry(
        &[42],
        &ResourceMetricsGpuConfig::default(),
        &mut SamplingState::default(),
        output,
        || None,
    );
    assert!(result.processes.is_none());
    assert_eq!(result.devices.unwrap().len(), 1);
}

#[test]
fn registry_clients_skip_missing_or_overflowing_times_and_merge_unavailable_rates() {
    let devices = [("macos:0x1".into(), Some(0))];
    let mut state = SamplingState::default();
    let now = Instant::now();
    apple_processes_from_registry(&client(1, 42, 100), &[42], &devices, &mut state, now).unwrap();
    let output = client(2, 42, 10)
        + &client(1, 42, 200)
        + "+-o AGXDeviceUserClient <id 0x3, registered>\n\"IOUserClientCreator\"=\"pid 42, client\"\n"
        + "+-o AGXDeviceUserClient <id 0x4, registered>\n\"IOUserClientCreator\"=\"pid 42, client\"\n\"accumulatedGPUTime\"=18446744073709551615\n\"accumulatedGPUTime\"=1\n";
    let result = apple_processes_from_registry(
        &output,
        &[42],
        &devices,
        &mut state,
        now + Duration::from_nanos(200),
    )
    .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].compute_utilization.as_ref().unwrap().value, 50.0);
    assert_eq!(state.client_baselines.len(), 2);
}
