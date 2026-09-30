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
