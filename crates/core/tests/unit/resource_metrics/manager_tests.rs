// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::api::resource_metrics::{CpuMetrics, DataUnit, DurationUnit};
use crate::diagnostics::{DiagnosticMemoryPage, DiagnosticQuery, Diagnostics};

#[cfg(any(target_os = "linux", target_os = "macos", windows))]
fn isolated_resource_manager_test() -> bool {
    // Isolate process-global ownership markers and bound regressions that could deadlock.
    const CHILD_ENV: &str = "NEMO_RELAY_TEST_RESOURCE_MANAGER_OWNER_CHILD";
    if std::env::var_os(CHILD_ENV).is_none() {
        let current_thread = std::thread::current();
        let test_name = current_thread.name().expect("named test thread");
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test_name, "--nocapture"])
            .env(CHILD_ENV, "1")
            .spawn()
            .expect("resource manager test process should start");
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success(), "resource manager test failed: {status}");
                return false;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("resource manager test did not finish within 20 seconds");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    true
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
fn manager_installation_reconciles_inherited_inspection_ownership() {
    use nemo_relay_types::diagnostics::ResourceMemoryAvailability;

    if !isolated_resource_manager_test() {
        return;
    }

    let pid = std::process::id();
    let inherited_pid = pid.wrapping_add(1).max(1);
    assert_eq!(
        latest_memory_view(Some(pid)).availability,
        ResourceMemoryAvailability::Disabled
    );
    assert!(GLOBAL_MANAGER.lock().unwrap().is_none());
    assert_eq!(MANAGER_OWNER_PID.load(Ordering::Acquire), pid);

    // Simulate a parent that inspected before activation, then forked while idle.
    MANAGER_OWNER_PID.store(inherited_pid, Ordering::Release);
    assert_eq!(
        latest_memory_view(Some(pid)).availability,
        ResourceMemoryAvailability::Invalidated
    );
    let manager = global_manager().unwrap();
    let target = current_process_target(ResourceMeasurementScope::ApplicationProcess).unwrap();
    let config = ResourceMetricsConfig {
        measurement_scope: ResourceMetricsMeasurementScope::ApplicationProcess,
        ..Default::default()
    };
    let generation = manager.activate(target, config).unwrap();
    assert!(
        manager
            .collect_blocking(SamplingSeries::OnDemand)
            .unwrap()
            .memory
            .is_some()
    );
    assert_eq!(
        latest_memory_view(Some(pid)).availability,
        ResourceMemoryAvailability::Available
    );
    assert_eq!(MANAGER_OWNER_PID.load(Ordering::Acquire), pid);
    assert!(Arc::ptr_eq(&manager, &global_manager().unwrap()));
    manager.deactivate(generation);

    // Neither inspection nor initialization may wait on an inherited locked mutex.
    *GLOBAL_MANAGER.lock().unwrap() = None;
    MANAGER_OWNER_PID.store(inherited_pid, Ordering::Release);
    let _locked = GLOBAL_MANAGER.lock().unwrap();
    ACTIVE_OWNER_PID.store(inherited_pid, Ordering::Release);
    assert!(global_manager().is_err());
    assert_eq!(
        latest_memory_view(Some(pid)).availability,
        ResourceMemoryAvailability::Invalidated
    );
    ACTIVE_OWNER_PID.store(0, Ordering::Release);
    assert!(global_manager().is_err());
    assert_eq!(MANAGER_OWNER_PID.load(Ordering::Acquire), inherited_pid);
    assert_eq!(
        latest_memory_view(Some(pid)).availability,
        ResourceMemoryAvailability::Invalidated
    );
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
fn resource_memory_pages_preserve_budgets_and_samples() {
    use crate::diagnostics::*;
    if !isolated_resource_manager_test() {
        return;
    }
    let diagnostics = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            limits: DiagnosticLimits {
                max_queries: 1,
                ..Default::default()
            },
        },
    )
    .unwrap();
    let code = BuiltinDiagnosticCode::Success(SuccessCode::SinkInitialized);
    let recorder = diagnostics
        .register_component(ComponentRegistration::new("memory.policy", &[code]))
        .unwrap();
    recorder
        .record_without_projection_lazy(|| {
            Observation::new(
                Classification::success(SuccessCode::SinkInitialized),
                "retained",
                std::num::NonZeroU64::MIN,
            )
        })
        .unwrap();
    let query = DiagnosticQuery {
        max_components: Some(0),
        max_events: Some(0),
        ..Default::default()
    };
    let baseline_bytes = diagnostics
        .reader()
        .inspect_memory(&query)
        .unwrap()
        .as_bytes()
        .len();
    assert!(GLOBAL_MANAGER.lock().unwrap().is_none());
    let before = diagnostics
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    let manager = global_manager().unwrap();
    let target = current_process_target(ResourceMeasurementScope::ApplicationProcess).unwrap();
    let config = ResourceMetricsConfig {
        measurement_scope: ResourceMetricsMeasurementScope::ApplicationProcess,
        ..Default::default()
    };
    let generation = manager.activate(target.clone(), config.clone()).unwrap();
    assert_eq!(
        diagnostics
            .reader()
            .inspect_memory(&query)
            .unwrap()
            .resource()
            .availability,
        ResourceMemoryAvailability::NotYetSampled
    );
    manager.collect_blocking(SamplingSeries::OnDemand).unwrap();
    let full = diagnostics.reader().inspect_memory(&query).unwrap();
    assert_eq!(
        full.resource().availability,
        ResourceMemoryAvailability::Available
    );
    let sample = full.resource().sample.unwrap();
    let full_page = full.to_memory_page().unwrap();
    let budget = baseline_bytes + 128;
    assert!(full.as_bytes().len() > budget);
    drop(full);
    assert_bounded_memory_page(&diagnostics, &query, budget, &full_page);
    assert_eq!(diagnostics.memory_usage().query_workspaces, 0);
    assert_eq!(
        latest_memory_view(Some(std::process::id())).sample,
        Some(sample)
    );
    assert_eq!(
        diagnostics
            .reader()
            .inspect_memory(&DiagnosticQuery {
                max_response_bytes: Some(1),
                ..query.clone()
            })
            .unwrap_err(),
        DiagnosticError::SnapshotCapacity
    );
    assert_eq!(diagnostics.memory_usage().query_workspaces, 0);
    assert_eq!(
        diagnostics
            .reader()
            .inspect_memory(&query)
            .unwrap()
            .resource()
            .sample,
        Some(sample)
    );
    let after = diagnostics
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(before.sequence, after.sequence);
    assert_eq!(before.events, after.events);
    assert_eq!(before.lifetime.totals, after.lifetime.totals);
    manager.deactivate(generation);
    assert_eq!(
        latest_memory_view(Some(std::process::id())).availability,
        ResourceMemoryAvailability::Disabled
    );
}

#[cfg(any(target_os = "linux", target_os = "macos", windows))]
fn assert_bounded_memory_page(
    diagnostics: &Diagnostics,
    query: &DiagnosticQuery,
    budget: usize,
    full_page: &DiagnosticMemoryPage,
) {
    use crate::diagnostics::ResourceMemoryAvailability;
    let limited = diagnostics
        .reader()
        .inspect_memory(&DiagnosticQuery {
            max_response_bytes: Some(budget as u32),
            ..query.clone()
        })
        .unwrap();
    assert!(limited.as_bytes().len() <= budget);
    let page = limited.to_memory_page().unwrap();
    assert_eq!(
        page.resource.availability,
        ResourceMemoryAvailability::CapacityUnavailable
    );
    assert!(page.resource.sample.is_none());
    assert_eq!(
        page.aggregate.reserved_bytes,
        full_page.aggregate.reserved_bytes
    );
    assert_eq!(page.aggregate.events, 1);
    assert_eq!(
        page.component_occupied_bytes,
        full_page.component_occupied_bytes
    );
    assert_eq!(
        page.component_occupied_bytes.get() + page.shared_or_unassigned_reserved_bytes.get(),
        page.aggregate.reserved_bytes as u64
    );
    drop(limited);
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
fn resource_memory_pages_respect_inspection_policy_and_pending_targets() {
    use crate::diagnostics::*;
    if !isolated_resource_manager_test() {
        return;
    }
    let diagnostics = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        },
    )
    .unwrap();
    let query = DiagnosticQuery::default();
    let manager = global_manager().unwrap();
    let target = current_process_target(ResourceMeasurementScope::ApplicationProcess).unwrap();
    let mut config = ResourceMetricsConfig {
        measurement_scope: ResourceMetricsMeasurementScope::ApplicationProcess,
        ..Default::default()
    };
    for disable_collection in [false, true] {
        config.memory_inspection.enabled = disable_collection;
        config.memory.enabled = !disable_collection;
        let generation = manager.activate(target.clone(), config.clone()).unwrap();
        assert_eq!(
            diagnostics
                .reader()
                .inspect_memory(&query)
                .unwrap()
                .resource()
                .availability,
            ResourceMemoryAvailability::Disabled
        );
        assert!(
            latest_memory_view(Some(std::process::id()))
                .sample
                .is_none()
        );
        manager.deactivate(generation);
    }
    config.memory.enabled = true;
    config.memory_inspection.enabled = true;
    config.measurement_scope = ResourceMetricsMeasurementScope::ProcessTree;
    let launch = manager.prepare_cli_owned_target().unwrap();
    let generation = manager
        .activate(
            current_process_target(ResourceMeasurementScope::ProcessTree).unwrap(),
            config,
        )
        .unwrap();
    assert_eq!(
        diagnostics
            .reader()
            .inspect_memory(&query)
            .unwrap()
            .resource()
            .availability,
        ResourceMemoryAvailability::AwaitingTarget
    );
    manager.deactivate(generation);
    drop(launch);
}

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
                DurationUnit::Milliseconds,
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
        read_data: Some(ResourceMeasurement::new(100_u64, DataUnit::Bytes)),
        write_data: Some(ResourceMeasurement::new(200_u64, DataUnit::Bytes)),
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
        read_data: Some(ResourceMeasurement::new(200_u64, DataUnit::Bytes)),
        write_data: Some(ResourceMeasurement::new(250_u64, DataUnit::Bytes)),
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
        read_data: Some(ResourceMeasurement::new(300_u64, DataUnit::Bytes)),
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
        read_data: Some(ResourceMeasurement::new(100_u64, DataUnit::Bytes)),
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
        read_data: Some(ResourceMeasurement::new(400_u64, DataUnit::Bytes)),
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
        crate::plugins::resource_metrics::config::DurationUnit::Seconds
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
            read_data: Some(ResourceMeasurement::new(read, DataUnit::Bytes)),
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
        read_data: Some(ResourceMeasurement::new(100_u64, DataUnit::Bytes)),
        write_data: Some(ResourceMeasurement::new(100_u64, DataUnit::Bytes)),
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
use crate::resource_metrics::snapshot_fixture;

#[test]
fn invalidating_a_sample_clears_all_process_measurements_and_keeps_category_shape() {
    let invalidated = super::invalidated_snapshot(snapshot_fixture::full_snapshot());
    let value = serde_json::to_value(&invalidated).unwrap();
    assert!(invalidated.process_sampling.is_none());
    for category in ["cpu", "memory", "process"] {
        for (field, measurement) in value[category].as_object().unwrap() {
            if field == "limit_events" {
                assert_eq!(measurement, &serde_json::json!([]));
            } else {
                assert!(measurement.is_null(), "{category}.{field}: {measurement}");
            }
        }
    }
    for field in [
        "read_data",
        "write_data",
        "read_throughput",
        "write_throughput",
        "read_operations",
        "write_operations",
    ] {
        assert!(value["disk"][field].is_null());
    }
    for field in ["total_capacity", "available_capacity", "free_capacity"] {
        assert!(value["disk"]["filesystems"][0][field].is_null());
    }
    assert!(invalidated.gpu.unwrap().device_metrics.is_none());
    assert!(invalidated.network.unwrap().system.received_data.is_some());
}
#[test]
fn manager_rejects_foreign_process_ownership_and_duplicate_activation() {
    let mut manager = ResourceMetricsManager::new().unwrap();
    manager.owner_pid = std::process::id().wrapping_add(1);
    let target = current_process_target(ResourceMeasurementScope::ProcessTree).unwrap();
    assert!(
        manager
            .activate(target.clone(), ResourceMetricsConfig::default())
            .is_err()
    );
    assert!(manager.collect_blocking(SamplingSeries::OnDemand).is_err());
    manager.deactivate(1);
    let manager = Arc::new(manager);
    drop(CliResourceMetricsLaunchGuard {
        manager: Arc::clone(&manager),
    });
    drop(ResourceMetricsTargetGuard {
        manager,
        generation: 1,
    });
    let manager = ResourceMetricsManager::new().unwrap();
    let generation = manager
        .activate(target.clone(), ResourceMetricsConfig::default())
        .unwrap();
    assert!(
        manager
            .activate(target, ResourceMetricsConfig::default())
            .is_err()
    );
    manager.deactivate(generation);
}

#[test]
fn collection_finishing_after_target_change_keeps_only_fresh_global_network_data() {
    for series in [SamplingSeries::OnDemand, SamplingSeries::Polling] {
        let manager = ResourceMetricsManager::new().unwrap();
        let target = current_process_target(ResourceMeasurementScope::ProcessTree).unwrap();
        let config = ResourceMetricsConfig {
            measurement_scope: ResourceMetricsMeasurementScope::ProcessTree,
            ..Default::default()
        };
        let runtime = manager.activate(target.clone(), config.clone()).unwrap();
        let old_generation = manager.state.lock().unwrap().target_generation;
        manager.target_owned_process_tree(target).unwrap();
        let mut stale = snapshot_fixture::full_snapshot();
        stale.network = None;
        let finished = manager
            .finish_collection_with_ticket(
                collector::CollectedSnapshot {
                    snapshot: stale,
                    process_sampled_instant: Instant::now(),
                    successful: true,
                    root_verified: false,
                    process_cpu_samples: Vec::new(),
                    process_io_samples: Vec::new(),
                },
                &config,
                old_generation,
                series,
                &current_process_target(ResourceMeasurementScope::ProcessTree).unwrap(),
                super::super::memory::Ticket {
                    collector: manager.collector_id,
                    runtime,
                    target: old_generation,
                    ordinal: 1,
                    started_at: Utc::now(),
                    started: Instant::now(),
                    stream: series,
                },
            )
            .unwrap();
        assert!(finished.cpu.unwrap().total_time.is_none());
        assert!(finished.process_sampling.is_none());
        assert_eq!(
            finished.network.unwrap().measurement_scope,
            ResourceMeasurementScope::Global
        );
        let state = manager.state.lock().unwrap();
        assert!(state.on_demand_baseline.is_none());
        assert!(state.polling_baseline.is_none());
        assert!(state.memory.sample.is_none());
        assert_eq!(
            state.on_demand_network_sampler.is_some(),
            matches!(series, SamplingSeries::OnDemand)
        );
        assert_eq!(
            state.polling_network_sampler.is_some(),
            matches!(series, SamplingSeries::Polling)
        );
        drop(state);
        manager.deactivate(runtime);
    }
}

#[test]
fn collection_finishing_after_deactivation_does_not_restart_any_sampler() {
    let manager = ResourceMetricsManager::new().unwrap();
    let config = ResourceMetricsConfig::default();
    let runtime = manager
        .activate(
            current_process_target(ResourceMeasurementScope::ProcessTree).unwrap(),
            config.clone(),
        )
        .unwrap();
    manager.deactivate(runtime);
    let mut stale = snapshot_fixture::full_snapshot();
    stale.network = None;
    let finished = manager
        .finish_collection_with_ticket(
            collector::CollectedSnapshot {
                snapshot: stale,
                process_sampled_instant: Instant::now(),
                successful: true,
                root_verified: false,
                process_cpu_samples: Vec::new(),
                process_io_samples: Vec::new(),
            },
            &config,
            runtime,
            SamplingSeries::OnDemand,
            &current_process_target(ResourceMeasurementScope::ProcessTree).unwrap(),
            super::super::memory::Ticket {
                collector: manager.collector_id,
                runtime,
                target: runtime,
                ordinal: 1,
                started_at: Utc::now(),
                started: Instant::now(),
                stream: SamplingSeries::OnDemand,
            },
        )
        .unwrap();
    assert!(finished.network.is_none());
    assert!(manager.state.lock().unwrap().memory.sample.is_none());
    assert!(finished.cpu.unwrap().total_time.is_none());
    let state = manager.state.lock().unwrap();
    assert!(state.on_demand_network_sampler.is_none());
    assert!(state.on_demand_baseline.is_none());
}

#[test]
fn cpu_rates_clear_missing_samples_and_reject_overflow_without_rewinding_baselines() {
    let now = Instant::now();
    let sample = |id, time| ProcessCpuSample {
        process_id: id,
        start_identity: 1,
        total_cpu_time_millis: time,
    };
    let mut baseline = None;
    derive_cpu_rate(&mut snapshot(1), now, &[sample(1, 1)], &mut baseline, 1);
    let mut disabled = snapshot(0);
    disabled.cpu = None;
    derive_cpu_rate(&mut disabled, now, &[], &mut baseline, 1);
    assert!(baseline.is_none());
    derive_cpu_rate(&mut snapshot(1), now, &[sample(1, 1)], &mut baseline, 1);
    let mut missing = snapshot(0);
    missing.cpu.as_mut().unwrap().total_time = None;
    derive_cpu_rate(&mut missing, now, &[], &mut baseline, 1);
    assert!(baseline.is_none());
    derive_cpu_rate(&mut snapshot(0), now, &[sample(1, 0)], &mut baseline, 1);
    let mut overflow = snapshot(u64::MAX);
    derive_cpu_rate(
        &mut overflow,
        now + Duration::from_secs(1),
        &[sample(1, u64::MAX), sample(2, 1)],
        &mut baseline,
        1,
    );
    assert!(overflow.cpu.unwrap().consumption_rate.is_none());
    let mut reset = snapshot(0);
    derive_cpu_rate(
        &mut reset,
        now + Duration::from_secs(2),
        &[sample(1, 0)],
        &mut baseline,
        1,
    );
    assert!(reset.cpu.unwrap().consumption_rate.is_none());
    let mut direct = snapshot(1);
    direct.cpu.as_mut().unwrap().consumption_rate =
        Some(ResourceMeasurement::new(2.0, CpuUnit::LogicalProcessors));
    derive_cpu_rate(
        &mut direct,
        now + Duration::from_secs(3),
        &[sample(1, 1)],
        &mut baseline,
        1,
    );
    assert_eq!(direct.cpu.unwrap().consumption_rate.unwrap().value, 2.0);
}

#[test]
fn scope_names_cpu_units_and_network_validation_match_the_public_contract() {
    use crate::plugins::resource_metrics::config::{CpuUnit, UtilizationUnit};
    for (scope, name) in [
        (ResourceMetricsMeasurementScope::Global, "global"),
        (
            ResourceMetricsMeasurementScope::ApplicationProcess,
            "application_process",
        ),
        (ResourceMetricsMeasurementScope::ProcessTree, "process_tree"),
    ] {
        assert_eq!(scope.as_str(), name);
    }
    assert_eq!(CpuUnit::Millicores, CpuUnit::Millicores);
    assert_eq!(UtilizationUnit::Percentage, UtilizationUnit::Percentage);
    let mut config = ResourceMetricsConfig::default();
    config.network.interfaces = vec![" \t".into()];
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("blank names")
    );
}
