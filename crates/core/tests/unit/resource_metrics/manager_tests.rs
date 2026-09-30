// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::api::resource_metrics::CpuMetrics;

fn snapshot(total_time: u64) -> ResourceMetricsSnapshot {
    ResourceMetricsSnapshot {
        timestamp: Utc::now(),
        operating_system: crate::api::resource_metrics::ResourceOperatingSystem::Unsupported,
        measurement_scope: ResourceMeasurementScope::ProcessTree,
        process_sampling: None,
        cpu: Some(CpuMetrics {
            user_time: None,
            system_time: None,
            total_time: Some(ResourceMeasurement::new(
                total_time,
                ResourceMeasurementUnit::Milliseconds,
            )),
            consumption_rate: None,
            throttled_time: None,
            effective_limit: None,
            some_pressure_stall_time: None,
            full_pressure_stall_time: None,
            limit_events: Vec::new(),
        }),
        memory: None,
        process: None,
        disk: None,
        gpu: None,
        network: None,
    }
}

#[test]
fn process_rate_survives_exited_children() {
    let start = Instant::now();
    let mut baseline = None;
    let first_samples = [
        ProcessCpuSample {
            process_id: 1,
            start_identity: 10,
            total_cpu_time_millis: 100,
        },
        ProcessCpuSample {
            process_id: 2,
            start_identity: 20,
            total_cpu_time_millis: 500,
        },
    ];
    derive_cpu_rate(&mut snapshot(600), start, &first_samples, &mut baseline, 1);
    let mut next = snapshot(160);
    let next_samples = [ProcessCpuSample {
        process_id: 1,
        start_identity: 10,
        total_cpu_time_millis: 160,
    }];
    derive_cpu_rate(
        &mut next,
        start + Duration::from_millis(100),
        &next_samples,
        &mut baseline,
        1,
    );
    assert_eq!(next.cpu.unwrap().consumption_rate.unwrap().value, 0.6);
}

#[test]
fn global_cpu_sampling_isolated_by_series_and_activation() {
    let manager = ResourceMetricsManager::new().unwrap();
    let target = current_process_target(ResourceMeasurementScope::Global).unwrap();
    let mut config = ResourceMetricsConfig::default();
    config.cpu.enabled = true;
    config.memory.enabled = false;
    config.process.enabled = false;
    config.disk.enabled = false;
    config.gpu.enabled = false;
    let generation = manager.activate(target.clone(), config.clone()).unwrap();
    let first_on_demand = manager.collect_blocking(SamplingSeries::OnDemand).unwrap();
    assert!(first_on_demand.cpu.unwrap().consumption_rate.is_none());
    std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL + Duration::from_millis(10));
    let first_poll = manager.collect_blocking(SamplingSeries::Polling).unwrap();
    assert!(first_poll.cpu.unwrap().consumption_rate.is_none());
    manager.deactivate(generation);
    let generation = manager.activate(target, config).unwrap();
    let first_after_reactivation = manager.collect_blocking(SamplingSeries::OnDemand).unwrap();
    assert!(
        first_after_reactivation
            .cpu
            .unwrap()
            .consumption_rate
            .is_none()
    );
    manager.deactivate(generation);
}

#[test]
fn cli_process_tree_waits_for_the_owned_target() {
    let manager = Arc::new(ResourceMetricsManager::new().unwrap());
    let launch = manager.prepare_cli_owned_target().unwrap();
    let target = current_process_target(ResourceMeasurementScope::ProcessTree).unwrap();
    let config = ResourceMetricsConfig {
        measurement_scope: ResourceMetricsMeasurementScope::ProcessTree,
        ..Default::default()
    };
    let runtime_generation = manager.activate(target.clone(), config).unwrap();

    assert!(manager.polling_target_pending());
    assert!(manager.collect_blocking(SamplingSeries::Polling).is_err());

    let target_generation = manager.target_owned_process_tree(target).unwrap();
    assert!(!manager.polling_target_pending());
    assert!(manager.collect_blocking(SamplingSeries::Polling).is_ok());

    manager.restore_application_target(target_generation);
    assert!(manager.polling_target_pending());
    assert!(manager.collect_blocking(SamplingSeries::Polling).is_err());
    manager.deactivate(runtime_generation);
    drop(launch);
}

#[test]
fn disk_throughput_uses_monotonic_deltas_and_resets_on_target_change() {
    let mut first = snapshot(0);
    first.disk = Some(crate::api::resource_metrics::DiskMetrics {
        read_data: Some(ResourceMeasurement::new(
            100_u64,
            ResourceMeasurementUnit::Bytes,
        )),
        write_data: Some(ResourceMeasurement::new(
            200_u64,
            ResourceMeasurementUnit::Bytes,
        )),
        read_throughput: None,
        write_throughput: None,
        read_operations: None,
        write_operations: None,
        filesystems: Vec::new(),
    });
    let now = Instant::now();
    let first_sample = ProcessIoSample {
        process_id: 1,
        start_identity: 10,
        read_bytes: Some(100),
        write_bytes: Some(200),
    };
    let mut baseline = None;
    derive_disk_rates(&mut first, now, &[first_sample], &mut baseline, 1);
    assert!(first.disk.unwrap().read_throughput.is_none());

    let mut second = snapshot(0);
    second.disk = Some(crate::api::resource_metrics::DiskMetrics {
        read_data: Some(ResourceMeasurement::new(
            200_u64,
            ResourceMeasurementUnit::Bytes,
        )),
        write_data: Some(ResourceMeasurement::new(
            250_u64,
            ResourceMeasurementUnit::Bytes,
        )),
        read_throughput: None,
        write_throughput: None,
        read_operations: None,
        write_operations: None,
        filesystems: Vec::new(),
    });
    let second_sample = ProcessIoSample {
        process_id: 1,
        start_identity: 10,
        read_bytes: Some(200),
        write_bytes: Some(250),
    };
    derive_disk_rates(
        &mut second,
        now + Duration::from_secs(2),
        &[second_sample],
        &mut baseline,
        1,
    );
    let disk = second.disk.unwrap();
    assert_eq!(disk.read_throughput.unwrap().value, 50.0);
    assert_eq!(disk.write_throughput.unwrap().value, 25.0);

    let mut changed = snapshot(0);
    changed.disk = Some(crate::api::resource_metrics::DiskMetrics {
        read_data: Some(ResourceMeasurement::new(
            300_u64,
            ResourceMeasurementUnit::Bytes,
        )),
        write_data: None,
        read_throughput: None,
        write_throughput: None,
        read_operations: None,
        write_operations: None,
        filesystems: Vec::new(),
    });
    let sample = ProcessIoSample {
        process_id: 1,
        start_identity: 10,
        read_bytes: Some(300),
        write_bytes: None,
    };
    derive_disk_rates(
        &mut changed,
        now + Duration::from_secs(3),
        &[sample],
        &mut baseline,
        2,
    );
    assert!(changed.disk.unwrap().read_throughput.is_none());
}

#[test]
fn disk_throughput_includes_newly_seen_processes() {
    let now = Instant::now();
    let mut baseline = None;
    let mut first = snapshot(0);
    first.disk = Some(crate::api::resource_metrics::DiskMetrics {
        read_data: Some(ResourceMeasurement::new(
            100_u64,
            ResourceMeasurementUnit::Bytes,
        )),
        write_data: None,
        read_throughput: None,
        write_throughput: None,
        read_operations: None,
        write_operations: None,
        filesystems: Vec::new(),
    });
    let root = ProcessIoSample {
        process_id: 1,
        start_identity: 10,
        read_bytes: Some(100),
        write_bytes: None,
    };
    derive_disk_rates(&mut first, now, &[root], &mut baseline, 1);

    let mut second = snapshot(0);
    second.disk = Some(crate::api::resource_metrics::DiskMetrics {
        read_data: Some(ResourceMeasurement::new(
            400_u64,
            ResourceMeasurementUnit::Bytes,
        )),
        write_data: None,
        read_throughput: None,
        write_throughput: None,
        read_operations: None,
        write_operations: None,
        filesystems: Vec::new(),
    });
    let child = ProcessIoSample {
        process_id: 2,
        start_identity: 20,
        read_bytes: Some(300),
        write_bytes: None,
    };
    derive_disk_rates(
        &mut second,
        now + Duration::from_secs(2),
        &[root, child],
        &mut baseline,
        1,
    );
    assert_eq!(second.disk.unwrap().read_throughput.unwrap().value, 150.0);
}

#[test]
fn unit_config_rejects_incompatible_or_unknown_choices() {
    let valid: ResourceMetricsConfig = serde_json::from_value(serde_json::json!({
        "units": {"cpu": {"user_time": "seconds"}, "disk": {"read_throughput": "megabits_per_second"}},
        "network": {"interfaces": ["lo0"]}
    })).unwrap();
    assert_eq!(
        valid.units.cpu.user_time,
        crate::plugins::resource_metrics::config::TimeUnit::Seconds
    );
    assert!(
        serde_json::from_value::<ResourceMetricsConfig>(serde_json::json!({
            "units": {"cpu": {"user_time": "mebibytes"}}
        }))
        .is_err()
    );
    assert!(
        serde_json::from_value::<ResourceMetricsConfig>(serde_json::json!({
            "units": {"disk": {"read_operations": "megabytes"}}
        }))
        .is_err()
    );
}

#[test]
fn out_of_order_samples_do_not_rewind_rate_baselines() {
    let now = Instant::now();
    let mut cpu_baseline = None;
    let cpu_sample = |total| ProcessCpuSample {
        process_id: 1,
        start_identity: 10,
        total_cpu_time_millis: total,
    };
    derive_cpu_rate(
        &mut snapshot(100),
        now,
        &[cpu_sample(100)],
        &mut cpu_baseline,
        1,
    );
    derive_cpu_rate(
        &mut snapshot(200),
        now + Duration::from_secs(2),
        &[cpu_sample(200)],
        &mut cpu_baseline,
        1,
    );
    derive_cpu_rate(
        &mut snapshot(150),
        now + Duration::from_secs(1),
        &[cpu_sample(150)],
        &mut cpu_baseline,
        1,
    );
    assert_eq!(
        cpu_baseline.unwrap().sampled_at,
        now + Duration::from_secs(2)
    );

    let disk_sample = |read| ProcessIoSample {
        process_id: 1,
        start_identity: 10,
        read_bytes: Some(read),
        write_bytes: None,
    };
    let mut disk_baseline = None;
    for (offset, read) in [(0, 100), (2, 200), (1, 150)] {
        let mut current = snapshot(0);
        current.disk = Some(crate::api::resource_metrics::DiskMetrics {
            read_data: Some(ResourceMeasurement::new(
                read,
                ResourceMeasurementUnit::Bytes,
            )),
            write_data: None,
            read_throughput: None,
            write_throughput: None,
            read_operations: None,
            write_operations: None,
            filesystems: Vec::new(),
        });
        derive_disk_rates(
            &mut current,
            now + Duration::from_secs(offset),
            &[disk_sample(read)],
            &mut disk_baseline,
            1,
        );
    }
    assert_eq!(
        disk_baseline.unwrap().sampled_at,
        now + Duration::from_secs(2)
    );
}

fn disk_rate_snapshot(scope: ResourceMeasurementScope) -> ResourceMetricsSnapshot {
    let mut value = snapshot(0);
    value.measurement_scope = scope;
    value.process_sampling = Some(crate::api::resource_metrics::ProcessSamplingMetadata {
        visible_processes: 3,
        sampled_processes: 3,
        field_sampled_processes: std::collections::BTreeMap::from([
            ("disk.read_data".into(), 2),
            ("disk.write_data".into(), 2),
        ]),
    });
    value.disk = Some(crate::api::resource_metrics::DiskMetrics {
        read_data: Some(ResourceMeasurement::new(
            100_u64,
            ResourceMeasurementUnit::Bytes,
        )),
        write_data: Some(ResourceMeasurement::new(
            100_u64,
            ResourceMeasurementUnit::Bytes,
        )),
        read_throughput: None,
        write_throughput: None,
        read_operations: None,
        write_operations: None,
        filesystems: vec![],
    });
    value
}

fn io_sample(id: u32, read: Option<u64>, write: Option<u64>) -> ProcessIoSample {
    ProcessIoSample {
        process_id: id,
        start_identity: u64::from(id),
        read_bytes: read,
        write_bytes: write,
    }
}

#[test]
fn global_disk_rates_skip_missing_pairs_and_report_independent_coverage() {
    let now = Instant::now();
    for scope in [
        ResourceMeasurementScope::Global,
        ResourceMeasurementScope::ProcessTree,
        ResourceMeasurementScope::ApplicationProcess,
    ] {
        let mut baseline = None;
        let mut first = disk_rate_snapshot(scope);
        derive_disk_rates(
            &mut first,
            now,
            &[
                io_sample(1, Some(100), Some(100)),
                io_sample(2, None, Some(100)),
                io_sample(3, Some(100), None),
            ],
            &mut baseline,
            1,
        );
        assert_eq!(
            first.process_sampling.unwrap().field_sampled_processes["disk.read_throughput"],
            0
        );
        let mut second = disk_rate_snapshot(scope);
        derive_disk_rates(
            &mut second,
            now + Duration::from_secs(2),
            &[
                io_sample(1, Some(200), None),
                io_sample(2, Some(900), Some(200)),
                io_sample(3, None, Some(900)),
            ],
            &mut baseline,
            1,
        );
        let disk = second.disk.unwrap();
        if scope == ResourceMeasurementScope::Global {
            assert_eq!(disk.read_throughput.unwrap().value, 50.0);
            assert_eq!(disk.write_throughput.unwrap().value, 50.0);
        } else {
            assert!(disk.read_throughput.is_none());
            assert!(disk.write_throughput.is_none());
        }
        let coverage = second.process_sampling.unwrap().field_sampled_processes;
        assert_eq!(coverage["disk.read_throughput"], 1);
        assert_eq!(coverage["disk.write_throughput"], 1);
    }
}

#[test]
fn global_disk_rates_require_usable_pairs_and_reject_resets_and_overflow() {
    let now = Instant::now();
    for (previous, current, expected_count) in [
        (
            vec![io_sample(1, None, Some(0))],
            vec![io_sample(1, Some(20), Some(0))],
            0,
        ),
        (
            vec![
                io_sample(1, Some(100), Some(0)),
                io_sample(2, Some(100), Some(0)),
            ],
            vec![
                io_sample(1, Some(90), Some(0)),
                io_sample(2, Some(200), Some(0)),
            ],
            1,
        ),
        (
            vec![
                io_sample(1, Some(0), Some(0)),
                io_sample(2, Some(0), Some(0)),
            ],
            vec![
                io_sample(1, Some(u64::MAX), Some(0)),
                io_sample(2, Some(1), Some(0)),
            ],
            2,
        ),
    ] {
        let mut baseline = None;
        derive_disk_rates(
            &mut disk_rate_snapshot(ResourceMeasurementScope::Global),
            now,
            &previous,
            &mut baseline,
            1,
        );
        let mut next = disk_rate_snapshot(ResourceMeasurementScope::Global);
        derive_disk_rates(
            &mut next,
            now + Duration::from_secs(1),
            &current,
            &mut baseline,
            1,
        );
        assert!(next.disk.unwrap().read_throughput.is_none());
        assert_eq!(
            next.process_sampling.unwrap().field_sampled_processes["disk.read_throughput"],
            expected_count
        );
    }
}

#[test]
fn global_disk_rates_count_new_processes_and_keep_genuine_zero_rates() {
    let now = Instant::now();
    let mut baseline = None;
    derive_disk_rates(
        &mut disk_rate_snapshot(ResourceMeasurementScope::Global),
        now,
        &[io_sample(1, Some(10), Some(0)), io_sample(2, None, Some(0))],
        &mut baseline,
        1,
    );
    let mut next = disk_rate_snapshot(ResourceMeasurementScope::Global);
    derive_disk_rates(
        &mut next,
        now + Duration::from_secs(2),
        &[
            io_sample(1, Some(10), Some(0)),
            io_sample(2, Some(99), Some(0)),
            io_sample(3, Some(20), Some(0)),
        ],
        &mut baseline,
        1,
    );
    assert_eq!(next.disk.unwrap().read_throughput.unwrap().value, 10.0);
    assert_eq!(
        next.process_sampling.unwrap().field_sampled_processes["disk.read_throughput"],
        2
    );
    let mut unchanged = disk_rate_snapshot(ResourceMeasurementScope::Global);
    derive_disk_rates(
        &mut unchanged,
        now + Duration::from_secs(3),
        &[
            io_sample(1, Some(10), Some(0)),
            io_sample(3, Some(20), Some(0)),
        ],
        &mut baseline,
        1,
    );
    assert_eq!(unchanged.disk.unwrap().read_throughput.unwrap().value, 0.0);
    assert_eq!(
        unchanged.process_sampling.unwrap().field_sampled_processes["disk.read_throughput"],
        2
    );
}
