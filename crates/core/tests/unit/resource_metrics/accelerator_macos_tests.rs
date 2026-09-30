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
        assert_eq!(measurement.unit, ResourceMeasurementUnit::Kibibytes);
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
