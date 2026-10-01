// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::process::Command;
use std::time::Instant;

use nemo_relay_types::api::resource_metrics::{
    AcceleratorDeviceMetrics, AcceleratorProcessMetrics, AcceleratorVendor, CapacityUnit,
    ResourceMeasurement, UtilizationUnit,
};

use crate::plugins::resource_metrics::config::ResourceMetricsGpuConfig;

use super::AcceleratorSample;

#[derive(Default)]
pub(super) struct SamplingState {
    // Registry entry IDs remain stable for the lifetime of an Apple GPU user client.
    client_baselines: BTreeMap<u64, (u64, Instant)>,
}

pub(super) fn collect(
    process_ids: &[u32],
    config: &ResourceMetricsGpuConfig,
    state: &mut SamplingState,
) -> AcceleratorSample {
    let Some(device_output) = ioreg(&["-r", "-c", "IOAccelerator", "-w0", "-l", "-d1"]) else {
        return AcceleratorSample::default();
    };
    collect_from_registry(process_ids, config, state, &device_output, || {
        ioreg(&["-r", "-c", "AGXDeviceUserClient", "-w0", "-l", "-d1"])
    })
}

fn collect_from_registry(
    process_ids: &[u32],
    config: &ResourceMetricsGpuConfig,
    state: &mut SamplingState,
    device_output: &str,
    process_output: impl FnOnce() -> Option<String>,
) -> AcceleratorSample {
    let all_devices = macos_devices(device_output);
    let selected_devices = all_devices
        .iter()
        .filter(|device| {
            config.devices.is_empty()
                || config.devices.iter().any(|selector| {
                    selector == &device.device_identifier
                        || device
                            .device_index
                            .is_some_and(|index| selector == &index.to_string())
                })
        })
        .cloned()
        .collect::<Vec<_>>();
    let processes = if config.process_metrics {
        if all_devices.len() == 1 && all_devices[0].vendor == AcceleratorVendor::Apple {
            let selected = selected_devices
                .iter()
                .map(|device| (device.device_identifier.clone(), device.device_index))
                .collect::<Vec<_>>();
            if selected.len() != 1 {
                None
            } else {
                process_output().and_then(|output| {
                    apple_processes_from_registry(
                        &output,
                        process_ids,
                        &selected,
                        state,
                        Instant::now(),
                    )
                })
            }
        } else {
            None
        }
    } else {
        None
    };
    let devices = selected_devices
        .into_iter()
        .filter(|device| device.memory_used.is_some() || device.compute_utilization.is_some())
        .collect();
    AcceleratorSample {
        devices: config.device_metrics.then_some(devices),
        processes,
    }
}

fn ioreg(arguments: &[&str]) -> Option<String> {
    // IOAccelerator statistics and AGX AppUsage are diagnostic registry properties. A missing
    // utility, driver, property, or permission leaves the corresponding measurement unavailable.
    let output = Command::new("/usr/sbin/ioreg")
        .args(arguments)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8(output.stdout).ok())
        .flatten()
}

fn macos_devices(output: &str) -> Vec<AcceleratorDeviceMetrics> {
    let mut devices = Vec::new();
    let mut next_index = 0_u32;
    for block in registry_blocks(output) {
        let Some(first_line) = block.lines().next() else {
            continue;
        };
        let vendor = if first_line.contains("AGXAccelerator") {
            AcceleratorVendor::Apple
        } else if first_line.contains("AMDRadeon") || first_line.contains("AMDAccelerator") {
            AcceleratorVendor::Amd
        } else if first_line.contains("IntelAccelerator") {
            AcceleratorVendor::Intel
        } else {
            continue;
        };
        let Some(registry_id) = registry_id(first_line) else {
            continue;
        };
        let identifier = format!("macos:{registry_id:#x}");
        let index = next_index;
        next_index = next_index.saturating_add(1);
        let memory_used = match vendor {
            AcceleratorVendor::Apple => registry_number(block, "In use system memory"),
            AcceleratorVendor::Amd | AcceleratorVendor::Intel => {
                registry_number(block, "inUseVidMemoryBytes")
                    .or_else(|| registry_number(block, "inUseSysMemoryBytes"))
            }
            _ => None,
        }
        .map(|bytes| ResourceMeasurement::new(bytes / 1_024, CapacityUnit::Kibibytes));
        let compute_utilization = registry_number(block, "Device Utilization %")
            .filter(|percent| *percent <= 100)
            .map(|percent| ResourceMeasurement::new(percent as f64, UtilizationUnit::Percentage));
        devices.push(AcceleratorDeviceMetrics {
            vendor,
            device_identifier: identifier,
            device_index: Some(index),
            memory_used,
            compute_utilization,
        });
    }
    devices
}

fn apple_processes_from_registry(
    output: &str,
    process_ids: &[u32],
    devices: &[(String, Option<u32>)],
    state: &mut SamplingState,
    sampled_at: Instant,
) -> Option<Vec<AcceleratorProcessMetrics>> {
    // AGX user-client properties do not identify their parent GPU. Avoid attributing a
    // process to the wrong device on a system with more than one selected Apple GPU.
    let [(identifier, index)] = devices else {
        return None;
    };
    let owned = process_ids.iter().copied().collect::<BTreeSet<_>>();
    let mut live_clients = BTreeSet::new();
    let mut process_busy = BTreeMap::<u32, Option<f64>>::new();
    for block in registry_blocks(output) {
        let Some(first_line) = block.lines().next() else {
            continue;
        };
        let (Some(client_id), Some(process_id)) = (registry_id(first_line), registry_pid(block))
        else {
            continue;
        };
        if !owned.contains(&process_id) {
            continue;
        }
        let gpu_times = registry_numbers(block, "accumulatedGPUTime");
        if gpu_times.is_empty() {
            continue;
        }
        let Some(total_gpu_nanoseconds) = gpu_times.into_iter().try_fold(0_u64, u64::checked_add)
        else {
            continue;
        };
        live_clients.insert(client_id);
        let previous = state
            .client_baselines
            .insert(client_id, (total_gpu_nanoseconds, sampled_at));
        let utilization = previous.and_then(|(last_total, last_at)| {
            let busy = total_gpu_nanoseconds.checked_sub(last_total)?;
            let elapsed = sampled_at.duration_since(last_at).as_nanos();
            (elapsed > 0).then(|| (busy as f64 / elapsed as f64 * 100.0).clamp(0.0, 100.0))
        });
        process_busy
            .entry(process_id)
            .and_modify(|current| {
                *current = match (*current, utilization) {
                    (Some(current), Some(next)) => Some((current + next).min(100.0)),
                    (Some(current), None) => Some(current),
                    (None, next) => next,
                }
            })
            .or_insert(utilization);
    }
    state
        .client_baselines
        .retain(|client_id, _| live_clients.contains(client_id));
    Some(
        process_busy
            .into_iter()
            .map(|(process_id, utilization)| AcceleratorProcessMetrics {
                vendor: AcceleratorVendor::Apple,
                device_identifier: identifier.clone(),
                device_index: *index,
                process_id,
                memory_used: None,
                compute_utilization: utilization
                    .map(|percent| ResourceMeasurement::new(percent, UtilizationUnit::Percentage)),
            })
            .collect(),
    )
}

#[cfg(test)]
#[path = "../../../tests/unit/resource_metrics/accelerator_macos_tests.rs"]
mod tests;

fn registry_blocks(output: &str) -> impl Iterator<Item = &str> {
    output.split("+-o ").skip(1)
}

fn registry_id(first_line: &str) -> Option<u64> {
    let value = first_line.split_once("id 0x")?.1;
    let hex = value.split_once(',')?.0;
    u64::from_str_radix(hex, 16).ok()
}

fn registry_pid(block: &str) -> Option<u32> {
    let creator = block.split_once("\"IOUserClientCreator\" = \"pid ")?.1;
    creator.split_once(',')?.0.parse().ok()
}

fn registry_number(block: &str, name: &str) -> Option<u64> {
    let key = format!("\"{name}\"=");
    let value = block.split_once(&key)?.1;
    let digits = value.bytes().take_while(u8::is_ascii_digit);
    let mut present = false;
    let total = digits.into_iter().try_fold(0_u64, |total, digit| {
        present = true;
        total.checked_mul(10)?.checked_add(u64::from(digit - b'0'))
    })?;
    present.then_some(total)
}

fn registry_numbers(block: &str, name: &str) -> Vec<u64> {
    let key = format!("\"{name}\"=");
    block
        .match_indices(&key)
        .filter_map(|(offset, _)| registry_number(&block[offset..], name))
        .collect()
}
