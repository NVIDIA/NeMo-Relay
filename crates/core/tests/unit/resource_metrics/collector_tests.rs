// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    accelerator_device_is_selected, bytes_to_kibibytes, clock_ticks_to_milliseconds,
    filetime_intervals_to_milliseconds, microseconds_to_milliseconds, nanoseconds_to_milliseconds,
};
use crate::plugins::resource_metrics::config::ResourceMetricsGpuConfig;

#[test]
fn time_conversions_truncate_submillisecond_precision_consistently() {
    assert_eq!(nanoseconds_to_milliseconds(2_000_999), 2);
    assert_eq!(microseconds_to_milliseconds(2_999), 2);
    assert_eq!(filetime_intervals_to_milliseconds(20_999), 2);
    assert_eq!(clock_ticks_to_milliseconds(250, 250), Some(1_000));
    assert_eq!(clock_ticks_to_milliseconds(1, 0), None);
}

#[test]
fn mach_cpu_times_use_the_timebase_and_truncate_only_after_conversion() {
    use super::mach_ticks_to_milliseconds;

    assert_eq!(mach_ticks_to_milliseconds(24_000_000, 125, 3), Some(1_000));
    assert_eq!(mach_ticks_to_milliseconds(24_000, 125, 3), Some(1));
    assert_eq!(mach_ticks_to_milliseconds(23_999, 125, 3), Some(0));
    assert_eq!(mach_ticks_to_milliseconds(1_999_999, 1, 1), Some(1));
    assert_eq!(mach_ticks_to_milliseconds(0, 125, 3), Some(0));
    assert_eq!(mach_ticks_to_milliseconds(1, 0, 3), None);
    assert_eq!(mach_ticks_to_milliseconds(1, 125, 0), None);
    // The intermediate nanosecond count can exceed u64 while milliseconds still fit.
    assert_eq!(
        mach_ticks_to_milliseconds(u64::MAX, 2, 1),
        Some(36_893_488_147_419)
    );
    assert_eq!(mach_ticks_to_milliseconds(u64::MAX, u32::MAX, 1), None);
}

#[test]
fn memory_conversion_normalizes_bytes_to_kibibytes() {
    assert_eq!(bytes_to_kibibytes(2_047), 1);
    assert_eq!(bytes_to_kibibytes(2_048), 2);
}

#[test]
fn gpu_selection_defaults_to_all_and_honors_identifiers_or_indices() {
    assert!(accelerator_device_is_selected(
        &ResourceMetricsGpuConfig::default(),
        "GPU-123",
        0
    ));
    let config = ResourceMetricsGpuConfig {
        devices: vec!["GPU-123".into()],
        ..Default::default()
    };
    assert!(accelerator_device_is_selected(&config, "GPU-123", 4));
    assert!(!accelerator_device_is_selected(&config, "GPU-456", 0));
    let config = ResourceMetricsGpuConfig {
        devices: vec!["4".into()],
        ..Default::default()
    };
    assert!(accelerator_device_is_selected(&config, "GPU-123", 4));
}

fn sample_with_cpu_time(value: Option<u64>) -> super::ProcessSample {
    super::ProcessSample {
        process_id: 1,
        start_identity: 1,
        user_cpu_time: value,
        system_cpu_time: None,
        total_cpu_time: None,
        resident_memory: None,
        private_memory: None,
        physical_footprint: None,
        virtual_memory: None,
        peak_resident_memory: None,
        thread_count: None,
        open_file_descriptor_count: None,
        windows_handle_count: None,
        disk_read_bytes: None,
        disk_write_bytes: None,
        disk_read_operations: None,
        disk_write_operations: None,
    }
}

#[test]
fn global_sums_skip_unreadable_counters_and_report_field_coverage() {
    let samples = [
        sample_with_cpu_time(Some(10)),
        sample_with_cpu_time(None),
        sample_with_cpu_time(Some(20)),
    ];
    let included = samples.iter().collect::<Vec<_>>();
    let mut aggregation = super::ProcessAggregation {
        samples: &included,
        complete: true,
        allow_partial: true,
        field_sampled_processes: Default::default(),
    };
    let sum = aggregation.sum(
        "cpu.user_time",
        super::ResourceMeasurementUnit::Milliseconds,
        |sample| sample.user_cpu_time,
    );
    assert_eq!(sum.unwrap().value, super::ResourceMetricValue::Integer(30));
    assert_eq!(aggregation.field_sampled_processes["cpu.user_time"], 2);
    aggregation.allow_partial = false;
    assert!(
        aggregation
            .sum(
                "cpu.user_time",
                super::ResourceMeasurementUnit::Milliseconds,
                |sample| sample.user_cpu_time
            )
            .is_none()
    );
    assert_eq!(aggregation.field_sampled_processes["cpu.user_time"], 2);
}

#[test]
fn sums_distinguish_zero_unavailable_overflow_and_incomplete_selection() {
    for (values, complete, expected, count) in [
        ([Some(0), None], true, Some(0), 1),
        ([None, None], true, None, 0),
        ([Some(u64::MAX), Some(1)], true, None, 2),
        ([Some(10), Some(20)], false, None, 2),
    ] {
        let samples = values.map(sample_with_cpu_time);
        let included = samples.iter().collect::<Vec<_>>();
        let mut aggregation = super::ProcessAggregation {
            samples: &included,
            complete,
            allow_partial: true,
            field_sampled_processes: Default::default(),
        };
        let sum = aggregation.sum(
            "cpu.user_time",
            super::ResourceMeasurementUnit::Milliseconds,
            |sample| sample.user_cpu_time,
        );
        assert_eq!(
            sum.map(|measurement| measurement.value),
            expected.map(super::ResourceMetricValue::Integer)
        );
        assert_eq!(aggregation.field_sampled_processes["cpu.user_time"], count);
    }
}
#[test]
fn an_invalid_process_identity_returns_unavailable_categories_in_the_stable_shape() {
    let mut target =
        super::current_process_target(super::ResourceMeasurementScope::ApplicationProcess).unwrap();
    target.start_identity = target.start_identity.wrapping_add(1);
    let snapshot = super::collect(
        &target,
        &crate::plugins::resource_metrics::config::ResourceMetricsConfig::default(),
    )
    .snapshot;
    assert_eq!(
        snapshot.measurement_scope,
        super::ResourceMeasurementScope::ApplicationProcess
    );
    let value = serde_json::to_value(&snapshot).unwrap();
    for category in ["cpu", "memory", "process", "disk"] {
        for (field, measurement) in value[category].as_object().unwrap() {
            if field == "limit_events" || field == "filesystems" {
                assert_eq!(measurement, &serde_json::json!([]));
            } else {
                assert!(measurement.is_null(), "{category}.{field}: {measurement}");
            }
        }
    }
    assert!(snapshot.process_sampling.is_none());
    let gpu = snapshot.gpu.unwrap();
    assert!(gpu.device_metrics.is_none());
    assert!(gpu.process_metrics.is_none());
}

#[test]
fn global_cpu_sampler_requires_a_baseline_and_a_minimum_interval() {
    let mut sampler = None;
    assert!(super::GlobalCpuSampler::sample(&mut sampler).is_none());
    assert!(super::GlobalCpuSampler::sample(&mut sampler).is_none());
    // Advance the baseline rather than adding a wall-clock delay to the test.
    sampler.as_mut().unwrap().last_sampled_at =
        std::time::Instant::now() - sysinfo::MINIMUM_CPU_UPDATE_INTERVAL;
    let measurement = super::GlobalCpuSampler::sample(&mut sampler).unwrap();
    assert_eq!(
        measurement.unit,
        super::ResourceMeasurementUnit::LogicalProcessors
    );
    assert!(measurement.value.is_finite());
    assert!(measurement.value >= 0.0);
}

#[test]
fn filesystem_capacity_failure_preserves_the_path_and_other_available_categories() {
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("missing");
    let mut config = crate::plugins::resource_metrics::config::ResourceMetricsConfig::default();
    config.gpu.enabled = false;
    config.disk.filesystem_paths = vec![missing.clone()];
    let target =
        super::current_process_target(super::ResourceMeasurementScope::ApplicationProcess).unwrap();
    let collected = super::collect(&target, &config);
    assert!(collected.successful);
    assert!(collected.snapshot.cpu.unwrap().total_time.is_some());
    let filesystem = collected.snapshot.disk.unwrap().filesystems.pop().unwrap();
    assert_eq!(filesystem.path, missing.to_string_lossy());
    assert!(filesystem.total_capacity.is_none());
    assert!(filesystem.available_capacity.is_none());
    assert!(filesystem.free_capacity.is_none());
}

#[test]
fn collector_maps_source_errors_and_rejects_unowned_or_missing_targets() {
    use std::io::{Error, ErrorKind};
    for (kind, issue) in [
        (
            ErrorKind::NotFound,
            super::CollectionIssue::TargetTerminated,
        ),
        (
            ErrorKind::PermissionDenied,
            super::CollectionIssue::PermissionDenied,
        ),
        (ErrorKind::Unsupported, super::CollectionIssue::Unsupported),
        (ErrorKind::Other, super::CollectionIssue::SourceUnavailable),
    ] {
        assert_eq!(super::reason_for_io_error(&Error::from(kind)), issue);
    }
    assert!(super::owned_process_tree_target(std::process::id()).is_err());
    assert!(super::owned_process_tree_target(u32::MAX).is_err());
    assert!(
        super::count_measurement(1, false, super::ResourceMeasurementUnit::Processes).is_none()
    );
}

fn tree_target() -> super::CollectionTarget {
    super::CollectionTarget {
        process_id: 1,
        start_identity: 1,
        measurement_scope: super::ResourceMeasurementScope::ProcessTree,
        #[cfg(windows)]
        job_handle: None,
    }
}

fn cpu_sample_config() -> super::ProcessSampleConfig {
    super::ProcessSampleConfig {
        cpu: true,
        memory: false,
        process: false,
        disk_io: false,
    }
}

#[test]
fn exited_descendants_are_removed_before_environment_gpu_and_process_aggregation() {
    use std::io::{Error, ErrorKind};
    for exit_during_identity in [true, false] {
        let selected = super::sample_selected_processes(
            &tree_target(),
            vec![1, 2, 3],
            cpu_sample_config(),
            |pid| {
                if pid == 2 && exit_during_identity {
                    Err(Error::from(ErrorKind::NotFound))
                } else {
                    Ok(1)
                }
            },
            |pid| {
                if pid == 2 {
                    assert!(!exit_during_identity, "do not sample a vanished PID");
                    Err(Error::from(ErrorKind::NotFound))
                } else {
                    let mut sample = sample_with_cpu_time(Some(10));
                    sample.process_id = pid;
                    Ok(sample)
                }
            },
        )
        .unwrap();
        assert_eq!(selected.process_ids, vec![1, 3]);
        assert!(selected.collection_issue.is_none());
        assert_eq!(selected.samples.len(), selected.process_ids.len());
        let included = selected.samples.iter().collect::<Vec<_>>();
        let mut aggregation = super::ProcessAggregation {
            samples: &included,
            complete: true,
            allow_partial: false,
            field_sampled_processes: Default::default(),
        };
        assert_eq!(
            aggregation
                .sum(
                    "cpu.user_time",
                    super::ResourceMeasurementUnit::Milliseconds,
                    |sample| sample.user_cpu_time
                )
                .unwrap()
                .value,
            super::ResourceMetricValue::Integer(20),
        );
    }
}

#[test]
fn process_tree_keeps_root_errors_identity_changes_and_other_descendant_errors_strict() {
    use std::io::{Error, ErrorKind};
    for kind in [
        ErrorKind::NotFound,
        ErrorKind::PermissionDenied,
        ErrorKind::Other,
    ] {
        assert!(
            super::sample_selected_processes(
                &tree_target(),
                vec![1, 2],
                cpu_sample_config(),
                |_| Ok(1),
                |_| Err(Error::from(kind)),
            )
            .is_none()
        );
    }
    assert!(
        super::sample_selected_processes(
            &tree_target(),
            vec![1],
            cpu_sample_config(),
            |_| Ok(1),
            |_| {
                let mut sample = sample_with_cpu_time(Some(10));
                sample.start_identity = 2;
                Ok(sample)
            },
        )
        .is_none()
    );
    for identity_failure in [true, false] {
        for kind in [
            ErrorKind::PermissionDenied,
            ErrorKind::Other,
            ErrorKind::InvalidInput,
        ] {
            let selected = super::sample_selected_processes(
                &tree_target(),
                vec![1, 2],
                cpu_sample_config(),
                |pid| {
                    if pid == 2 && identity_failure {
                        Err(Error::from(kind))
                    } else {
                        Ok(1)
                    }
                },
                |pid| {
                    if pid == 2 {
                        Err(Error::from(kind))
                    } else {
                        Ok(sample_with_cpu_time(Some(10)))
                    }
                },
            )
            .unwrap();
            assert_eq!(selected.process_ids, vec![1, 2]);
            assert_eq!(
                selected.collection_issue,
                Some(super::reason_for_io_error(&Error::from(kind)))
            );
            assert_eq!(selected.samples.len(), 1);
        }
    }
    let selected = super::sample_selected_processes(
        &tree_target(),
        vec![1, 2],
        cpu_sample_config(),
        |_| Ok(1),
        |pid| {
            let mut sample = sample_with_cpu_time(Some(10));
            sample.process_id = pid;
            if pid == 2 {
                sample.start_identity = 2;
            }
            Ok(sample)
        },
    )
    .unwrap();
    assert_eq!(
        selected.collection_issue,
        Some(super::CollectionIssue::TargetIdentityChanged)
    );
    assert_eq!(selected.process_ids, vec![1, 2]);
}

#[test]
fn global_process_coverage_still_counts_exited_processes_as_unsampled() {
    let mut target = tree_target();
    target.measurement_scope = super::ResourceMeasurementScope::Global;
    let selected = super::sample_selected_processes(
        &target,
        vec![1, 2],
        cpu_sample_config(),
        |_| panic!("global queries do not require a separate identity"),
        |pid| {
            if pid == 2 {
                Err(std::io::ErrorKind::NotFound.into())
            } else {
                Ok(sample_with_cpu_time(Some(10)))
            }
        },
    )
    .unwrap();
    assert_eq!(selected.process_ids, vec![1, 2]);
    assert_eq!(selected.samples.len(), 1);
    assert_eq!(
        selected.collection_issue,
        Some(super::CollectionIssue::TargetTerminated)
    );
}

#[test]
fn exit_errors_do_not_include_generic_invalid_input_or_permission_failures() {
    use std::io::{Error, ErrorKind};
    assert!(super::process_has_exited(&Error::from(ErrorKind::NotFound)));
    for kind in [
        ErrorKind::PermissionDenied,
        ErrorKind::InvalidInput,
        ErrorKind::Other,
    ] {
        assert!(!super::process_has_exited(&Error::from(kind)));
    }
    #[cfg(target_os = "linux")]
    assert!(super::process_has_exited(&Error::from_raw_os_error(
        rustix::io::Errno::SRCH.raw_os_error()
    )));
    #[cfg(windows)]
    assert!(super::process_has_exited(&Error::from_raw_os_error(
        windows_sys::Win32::Foundation::ERROR_INVALID_PARAMETER as i32
    )));
}
