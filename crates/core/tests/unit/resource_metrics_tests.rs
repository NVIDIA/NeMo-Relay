// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use nemo_relay_types::api::event::Event;
#[cfg(unix)]
use nemo_relay_types::api::resource_metrics::ResourceMeasurementScope;
use nemo_relay_types::api::resource_metrics::{
    AcceleratorDeviceMetrics, AcceleratorProcessMetrics, AcceleratorVendor,
    ResourceLimitEventCount, ResourceLimitEventKind, ResourceLimitResource,
    ResourceMeasurementUnit, ResourceMetricsSnapshot, ResourceOperatingSystem,
};
use serde_json::Value;

use crate::api::resource_metrics::{
    ResourceMeasurement, ResourceMetricsConfig, ResourceMetricsFileConfig,
    ResourceMetricsPollingConfig, ResourceMetricsRuntime, collect_resource_metrics,
    latest_resource_metrics, resource_metrics_history,
};
use crate::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use crate::error::FlowError;

const SCALAR_MEASUREMENT_FIELDS: &[&str] = &[
    "cpu_user_time",
    "cpu_system_time",
    "cpu_total_time",
    "cpu_consumption_rate",
    "cpu_throttled_time",
    "effective_cpu_limit",
    "cpu_some_pressure_stall_time",
    "cpu_full_pressure_stall_time",
    "resident_memory",
    "private_memory",
    "physical_footprint",
    "virtual_memory",
    "peak_resident_memory",
    "memory_limit",
    "environment_accounted_memory",
    "memory_some_pressure_stall_time",
    "memory_full_pressure_stall_time",
    "out_of_memory_event_count",
    "active_process_count",
    "descendant_process_count",
    "thread_count",
    "lifetime_process_creation_count",
    "open_file_descriptor_count",
    "windows_handle_count",
];

fn lock_global_runtime() -> std::sync::MutexGuard<'static, ()> {
    crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

fn wait_for_history_length(minimum: usize, timeout: Duration) -> Vec<ResourceMetricsSnapshot> {
    let deadline = Instant::now() + timeout;
    while resource_metrics_history().unwrap().len() < minimum && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    let history = resource_metrics_history().unwrap();
    assert!(
        history.len() >= minimum,
        "expected at least {minimum} polling snapshots, got {}",
        history.len()
    );
    history
}

fn read_jsonl_lines(path: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("resource metrics output file exists")
        .lines()
        .map(str::to_string)
        .collect()
}

fn assert_complete_jsonl_record(encoded: &str) {
    let record: Value = serde_json::from_str(encoded).unwrap();
    assert_eq!(
        record["operating_system"],
        expected_operating_system().as_str()
    );
    assert!(record["resource_limit_events"].is_array());
    assert!(record["accelerator_devices"].is_array());
    assert!(record["accelerator_processes"].is_array());
    for field in SCALAR_MEASUREMENT_FIELDS {
        let measurement = record.get(field).expect("serialized scalar field");
        assert_eq!(measurement.as_object().unwrap().len(), 3);
        assert_measurement_schema(measurement);
        let object_start = encoded.find(&format!("\"{field}\":{{")).unwrap();
        let measurement_json = &encoded[object_start..];
        let positions = ["\"value\"", "\"unit\"", "\"timestamp\""].map(|key| {
            measurement_json
                .find(key)
                .expect("ordered measurement field in JSONL record")
        });
        assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
    }
}

fn expected_native_units() -> (
    ResourceMeasurementUnit,
    ResourceMeasurementUnit,
    ResourceMeasurementUnit,
) {
    #[cfg(target_os = "macos")]
    {
        (
            ResourceMeasurementUnit::Nanoseconds,
            ResourceMeasurementUnit::NanosecondsPerSecond,
            ResourceMeasurementUnit::Bytes,
        )
    }
    #[cfg(target_os = "linux")]
    {
        (
            ResourceMeasurementUnit::ClockTicks,
            ResourceMeasurementUnit::ClockTicksPerSecond,
            ResourceMeasurementUnit::Pages,
        )
    }
    #[cfg(windows)]
    {
        (
            ResourceMeasurementUnit::HundredNanosecondIntervals,
            ResourceMeasurementUnit::HundredNanosecondIntervalsPerSecond,
            ResourceMeasurementUnit::Bytes,
        )
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
    {
        (
            ResourceMeasurementUnit::Unknown,
            ResourceMeasurementUnit::Unknown,
            ResourceMeasurementUnit::Unknown,
        )
    }
}

fn expected_operating_system() -> ResourceOperatingSystem {
    #[cfg(target_os = "linux")]
    {
        ResourceOperatingSystem::Linux
    }
    #[cfg(target_os = "macos")]
    {
        ResourceOperatingSystem::Macos
    }
    #[cfg(windows)]
    {
        ResourceOperatingSystem::Windows
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        ResourceOperatingSystem::Unsupported
    }
}

fn assert_measurement_field_order(encoded_measurement: &str) {
    let field_positions = ["\"value\"", "\"unit\"", "\"timestamp\""].map(|field| {
        encoded_measurement
            .find(field)
            .expect("serialized measurement field")
    });
    assert!(field_positions.windows(2).all(|pair| pair[0] < pair[1]));
}

fn assert_measurement_schema(measurement: &Value) {
    let timestamp = measurement["timestamp"].as_str().unwrap();
    assert!(DateTime::parse_from_rfc3339(timestamp).is_ok());
    assert!(measurement.get("status").is_none());
    assert!(measurement.get("reason").is_none());
    if measurement["value"].is_null() {
        assert!(measurement["unit"].is_null());
    } else {
        assert!(measurement["value"].is_number());
        assert!(measurement["unit"].as_str().is_some());
    }
}

#[test]
fn resource_metrics_configuration_defaults_and_rejects_invalid_polling_or_file_output() {
    let defaults = ResourceMetricsConfig::default();
    assert!(!defaults.polling.enabled);
    assert!(!defaults.file.enabled);
    defaults.validate().unwrap();

    let invalid_interval = ResourceMetricsConfig {
        polling: ResourceMetricsPollingConfig {
            enabled: true,
            interval_millis: 0,
            ..defaults.polling.clone()
        },
        ..defaults.clone()
    };
    assert!(matches!(
        invalid_interval.validate(),
        Err(FlowError::InvalidArgument(_))
    ));

    let invalid_history = ResourceMetricsConfig {
        polling: ResourceMetricsPollingConfig {
            enabled: true,
            retained_snapshots: 0,
            ..defaults.polling.clone()
        },
        ..defaults.clone()
    };
    assert!(matches!(
        invalid_history.validate(),
        Err(FlowError::InvalidArgument(_))
    ));

    let file_without_polling = ResourceMetricsConfig {
        file: ResourceMetricsFileConfig {
            enabled: true,
            ..defaults.file.clone()
        },
        ..defaults.clone()
    };
    assert!(matches!(
        file_without_polling.validate(),
        Err(FlowError::InvalidArgument(_))
    ));

    let invalid_file = ResourceMetricsConfig {
        polling: ResourceMetricsPollingConfig {
            enabled: true,
            ..defaults.polling.clone()
        },
        file: ResourceMetricsFileConfig {
            enabled: true,
            path: "".into(),
            ..defaults.file.clone()
        },
    };
    assert!(matches!(
        invalid_file.validate(),
        Err(FlowError::InvalidArgument(_))
    ));

    for file in [
        ResourceMetricsFileConfig {
            enabled: true,
            max_file_size_bytes: 0,
            ..defaults.file.clone()
        },
        ResourceMetricsFileConfig {
            enabled: true,
            retained_files: 0,
            ..defaults.file.clone()
        },
    ] {
        let invalid_file = ResourceMetricsConfig {
            polling: ResourceMetricsPollingConfig {
                enabled: true,
                ..defaults.polling.clone()
            },
            file,
        };
        assert!(matches!(
            invalid_file.validate(),
            Err(FlowError::InvalidArgument(_))
        ));
    }
}

#[test]
#[allow(clippy::cognitive_complexity)] // Keeps one serialized snapshot contract in a single test.
fn snapshot_serialization_covers_native_scalar_limit_and_accelerator_measurements() {
    let _guard = lock_global_runtime();
    let mut snapshot = collect_resource_metrics().unwrap();
    let timestamp = Utc::now();
    let later_timestamp = timestamp + ChronoDuration::milliseconds(7);
    snapshot.thread_count = ResourceMeasurement::unavailable(later_timestamp);
    snapshot.resource_limit_events = vec![ResourceLimitEventCount {
        resource: ResourceLimitResource::Cpu,
        event: ResourceLimitEventKind::Throttled,
        count: ResourceMeasurement::available(timestamp, 3, ResourceMeasurementUnit::Events),
    }];
    snapshot.accelerator_devices = vec![AcceleratorDeviceMetrics {
        vendor: AcceleratorVendor::Nvidia,
        device_identifier: "gpu-test".to_string(),
        device_index: Some(0),
        memory_used: ResourceMeasurement::available(
            timestamp,
            4096,
            ResourceMeasurementUnit::Bytes,
        ),
        compute_utilization: ResourceMeasurement::unavailable(later_timestamp),
    }];
    snapshot.accelerator_processes = vec![AcceleratorProcessMetrics {
        vendor: AcceleratorVendor::Nvidia,
        device_identifier: "gpu-test".to_string(),
        device_index: None,
        process_id: std::process::id(),
        memory_used: ResourceMeasurement::unavailable(later_timestamp),
        compute_utilization: ResourceMeasurement::available(
            timestamp,
            0.25,
            ResourceMeasurementUnit::Percentage,
        ),
    }];

    let encoded = serde_json::to_value(&snapshot).unwrap();
    assert_eq!(
        encoded["operating_system"],
        expected_operating_system().as_str()
    );
    for field in SCALAR_MEASUREMENT_FIELDS {
        assert!(
            encoded.get(field).is_some(),
            "missing snapshot field {field}"
        );
        assert_measurement_schema(&encoded[field]);
    }
    assert_eq!(encoded["thread_count"]["value"], Value::Null);
    assert_eq!(encoded["thread_count"]["unit"], Value::Null);
    let encoded_unavailable_timestamp =
        DateTime::parse_from_rfc3339(encoded["thread_count"]["timestamp"].as_str().unwrap())
            .unwrap();
    assert_eq!(
        encoded_unavailable_timestamp.timestamp_micros(),
        later_timestamp.timestamp_micros()
    );
    assert_measurement_field_order(&serde_json::to_string(&snapshot.thread_count).unwrap());

    assert_eq!(encoded["resource_limit_events"][0]["resource"], "cpu");
    assert_eq!(encoded["resource_limit_events"][0]["event"], "throttled");
    assert_eq!(encoded["resource_limit_events"][0]["count"]["value"], 3);
    assert_eq!(
        encoded["resource_limit_events"][0]["count"]["unit"],
        "events"
    );
    let device = &encoded["accelerator_devices"][0];
    assert_eq!(device["vendor"], "nvidia");
    assert_eq!(device["memory_used"]["value"], 4096);
    assert_eq!(device["compute_utilization"]["value"], Value::Null);
    assert_eq!(device["compute_utilization"]["unit"], Value::Null);
    let process = &encoded["accelerator_processes"][0];
    assert_eq!(process["process_id"], std::process::id());
    assert_eq!(process["memory_used"]["value"], Value::Null);
    assert_eq!(process["memory_used"]["unit"], Value::Null);
    assert_eq!(process["compute_utilization"]["value"], 0.25);

    let metric_measurements = super::metric_measurements(&snapshot);
    nemo_relay_types::api::event::validate_metric_measurements(&metric_measurements)
        .expect("GPU measurements have valid metric attributes");
    let device_attributes = metric_measurements
        .iter()
        .find(|measurement| {
            measurement.name == "nemo.relay.resource.accelerator.device.memory_used"
        })
        .and_then(|measurement| measurement.attributes.as_ref())
        .expect("available device memory metric attributes");
    assert_eq!(
        device_attributes["nemo_relay.resource.accelerator.device_index"],
        0
    );
    assert!(
        device_attributes
            .get("nemo_relay.resource.process_id")
            .is_none()
    );
    let process_attributes = metric_measurements
        .iter()
        .find(|measurement| {
            measurement.name == "nemo.relay.resource.accelerator.process.compute_utilization"
        })
        .and_then(|measurement| measurement.attributes.as_ref())
        .expect("available process utilization metric attributes");
    assert!(
        process_attributes
            .get("nemo_relay.resource.accelerator.device_index")
            .is_none()
    );
    assert_eq!(
        process_attributes["nemo_relay.resource.process_id"],
        std::process::id()
    );
}

#[test]
fn collector_rejects_a_target_with_a_changed_start_identity() {
    let _guard = lock_global_runtime();
    let mut target = crate::resource_metrics::collector::current_process_target().unwrap();
    target.start_identity = target.start_identity.wrapping_add(1);

    let collected = crate::resource_metrics::collector::collect(&target);
    assert!(matches!(
        collected.snapshot.active_process_count,
        ResourceMeasurement {
            value: None,
            unit: None,
            ..
        }
    ));
    assert!(!collected.successful);
}

#[test]
#[allow(clippy::cognitive_complexity)] // Exercises the complete serialized lifecycle under one global-runtime lock.
fn fresh_polling_history_file_agent_marks_and_owned_target_are_consistent() {
    let _guard = lock_global_runtime();

    let latest_before_fresh = latest_resource_metrics().unwrap();
    let history_before_fresh = resource_metrics_history().unwrap();
    let fresh = collect_resource_metrics().expect("fresh application-process snapshot");
    assert_eq!(fresh.operating_system, expected_operating_system());
    assert!(
        fresh
            .active_process_count
            .value()
            .is_some_and(|count| *count > 0),
        "unexpected current-process observation: {:?}",
        fresh.active_process_count
    );
    assert_eq!(latest_resource_metrics().unwrap(), latest_before_fresh);
    assert_eq!(resource_metrics_history().unwrap(), history_before_fresh);
    let encoded_fresh = serde_json::to_value(&fresh).unwrap();
    for field in SCALAR_MEASUREMENT_FIELDS {
        assert_measurement_schema(&encoded_fresh[field]);
    }
    if fresh.cpu_consumption_rate.value.is_some() {
        assert_eq!(
            fresh.cpu_consumption_rate.unit,
            Some(expected_native_units().1)
        );
    } else {
        assert_eq!(fresh.cpu_consumption_rate.unit, None);
    }
    assert_eq!(fresh.cpu_user_time.unit, Some(expected_native_units().0));
    assert_eq!(fresh.cpu_system_time.unit, Some(expected_native_units().0));
    assert_eq!(fresh.cpu_total_time.unit, Some(expected_native_units().0));
    assert_eq!(fresh.resident_memory.unit, Some(expected_native_units().2));
    assert_eq!(
        fresh.active_process_count.unit,
        Some(ResourceMeasurementUnit::Processes)
    );
    assert_eq!(
        fresh.thread_count.unit,
        Some(ResourceMeasurementUnit::Threads)
    );
    #[cfg(target_os = "macos")]
    {
        assert!(fresh.physical_footprint.value.is_some());
        assert!(fresh.virtual_memory.value.is_some());
        assert!(fresh.open_file_descriptor_count.value.is_some());
        for field in [
            "private_memory",
            "peak_resident_memory",
            "cpu_throttled_time",
            "effective_cpu_limit",
            "cpu_some_pressure_stall_time",
            "cpu_full_pressure_stall_time",
            "memory_limit",
            "environment_accounted_memory",
            "memory_some_pressure_stall_time",
            "memory_full_pressure_stall_time",
            "out_of_memory_event_count",
            "lifetime_process_creation_count",
            "windows_handle_count",
        ] {
            assert_eq!(encoded_fresh[field]["value"], Value::Null);
            assert_eq!(encoded_fresh[field]["unit"], Value::Null);
        }
    }
    assert_eq!(
        encoded_fresh["operating_system"],
        expected_operating_system().as_str()
    );
    assert!(encoded_fresh.get("measurement_scope").is_none());
    assert!(encoded_fresh.get("collected_at").is_none());
    assert!(encoded_fresh.get("sample_interval_millis").is_none());
    assert_measurement_field_order(&serde_json::to_string(&fresh.cpu_total_time).unwrap());
    let _: Option<&u64> = fresh.cpu_total_time.value();
    let _: Option<&f64> = fresh.cpu_consumption_rate.value();

    let directory = tempfile::tempdir().unwrap();
    let resource_file = directory.path().join("resource-metrics.jsonl");
    let captured = Arc::new(Mutex::new(Vec::<Event>::new()));
    let subscriber_events = Arc::clone(&captured);
    register_subscriber(
        "resource-metrics-test-observer",
        Arc::new(move |event| {
            subscriber_events
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(event.clone());
        }),
    )
    .unwrap();

    let runtime = ResourceMetricsRuntime::configure(ResourceMetricsConfig {
        polling: ResourceMetricsPollingConfig {
            enabled: true,
            interval_millis: 10,
            retained_snapshots: 2,
        },
        file: ResourceMetricsFileConfig {
            enabled: true,
            path: resource_file.clone(),
            max_file_size_bytes: 1_000_000,
            retained_files: 2,
        },
    })
    .expect("managed polling runtime");

    let deadline = Instant::now() + Duration::from_secs(2);
    while resource_metrics_history().unwrap().len() < 2 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    let retained = resource_metrics_history().unwrap();
    assert_eq!(
        retained.len(),
        2,
        "history must enforce its configured bound"
    );
    assert_eq!(latest_resource_metrics().unwrap().as_ref(), retained.last());

    let before_fresh = resource_metrics_history().unwrap().len();
    let independent_fresh = collect_resource_metrics().unwrap();
    assert!(independent_fresh.active_process_count.value().is_some());
    assert_eq!(resource_metrics_history().unwrap().len(), before_fresh);

    flush_subscribers().unwrap();
    assert!(
        captured
            .lock()
            .unwrap()
            .iter()
            .all(|event| event.name() != "nemo.relay.resource_metrics"),
        "polling must not emit metric marks"
    );

    let agent = push_scope(
        PushScopeParams::builder()
            .name("resource_metrics_test_agent")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();

    let semantic_turn = push_scope(
        PushScopeParams::builder()
            .name("resource_metrics_test_turn")
            .scope_type(ScopeType::Custom)
            .metadata(serde_json::json!({"nemo_relay_scope_role": "turn"}))
            .build(),
    )
    .unwrap();
    pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&semantic_turn.uuid)
            .build(),
    )
    .unwrap();

    let ordinary_custom_scope = push_scope(
        PushScopeParams::builder()
            .name("resource_metrics_test_ordinary_custom_scope")
            .scope_type(ScopeType::Custom)
            .build(),
    )
    .unwrap();
    pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&ordinary_custom_scope.uuid)
            .build(),
    )
    .unwrap();
    flush_subscribers().unwrap();

    let events = captured.lock().unwrap();
    let metric_marks = events
        .iter()
        .filter(|event| event.name() == "nemo.relay.resource_metrics")
        .collect::<Vec<_>>();
    assert_eq!(
        metric_marks.len(),
        4,
        "Agent and semantic turn start/end each produce one mark"
    );
    assert_eq!(
        metric_marks
            .iter()
            .filter(|event| event.parent_uuid() == Some(agent.uuid))
            .count(),
        2,
        "Agent start and end each produce one mark"
    );
    assert_eq!(
        metric_marks
            .iter()
            .filter(|event| event.parent_uuid() == Some(semantic_turn.uuid))
            .count(),
        2,
        "semantic turn start and end each produce one mark"
    );
    assert!(
        metric_marks
            .iter()
            .all(|event| event.parent_uuid() != Some(ordinary_custom_scope.uuid))
    );
    assert!(metric_marks.iter().all(|event| {
        let Some(measurements) = event
            .data()
            .and_then(|data| data.get("measurements"))
            .and_then(Value::as_array)
        else {
            return false;
        };
        let timestamps = measurements
            .iter()
            .filter_map(|measurement| {
                DateTime::parse_from_rfc3339(
                    measurement["attributes"]["nemo_relay.resource.measurement_timestamp"]
                        .as_str()?,
                )
                .ok()
                .map(|timestamp| timestamp.timestamp_micros())
            })
            .collect::<Vec<_>>();
        event.kind() == "mark"
            && !measurements.is_empty()
            && measurements.iter().all(|measurement| {
                measurement.get("value").is_some()
                    && measurement["value"].is_number()
                    && measurement.get("reason").is_none()
                    && measurement["unit"].as_str().is_some()
                    && measurement["name"].as_str().is_some_and(|name| {
                        !name.ends_with("_bytes")
                            && !name.ends_with("_nanoseconds")
                            && !name.ends_with("_clock_ticks")
                    })
            })
            && timestamps.len() == measurements.len()
            && measurements.iter().all(|measurement| {
                measurement["attributes"]["nemo_relay.resource.operating_system"]
                    == expected_operating_system().as_str()
                    && measurement["attributes"]
                        .get("nemo_relay.resource.measurement_scope")
                        .is_none()
            })
            && timestamps
                .iter()
                .max()
                .copied()
                .is_some_and(|timestamp| timestamp <= event.timestamp().timestamp_micros())
    }));
    drop(events);

    let file_content = std::fs::read_to_string(&resource_file)
        .expect("polling writes structured snapshots")
        .lines()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let file_lines = file_content
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert!(file_lines.len() >= 2);
    assert!(file_lines.iter().all(|line| {
        line["operating_system"] == expected_operating_system().as_str()
            && line.get("measurement_scope").is_none()
            && line.get("collected_at").is_none()
            && line.get("sample_interval_millis").is_none()
            && line["active_process_count"]["value"].is_number()
    }));
    assert_measurement_field_order(&file_content[0]);
    for measurement_name in SCALAR_MEASUREMENT_FIELDS {
        for snapshot in &file_lines {
            assert_measurement_schema(&snapshot[measurement_name]);
        }
    }

    drop(runtime);
    assert!(deregister_subscriber("resource-metrics-test-observer").unwrap());

    let self_target_error = match ResourceMetricsRuntime::configure_owned_process_tree(
        ResourceMetricsConfig::default(),
        std::process::id(),
    ) {
        Err(error) => error,
        Ok(runtime) => {
            drop(runtime);
            panic!("the Relay process is not its own direct child")
        }
    };
    assert!(matches!(self_target_error, FlowError::InvalidArgument(_)));

    #[cfg(unix)]
    {
        let mut child = ChildProcessGuard(
            std::process::Command::new("sleep")
                .arg("30")
                .spawn()
                .expect("launch a direct child process for ownership validation"),
        );
        let owned_runtime = ResourceMetricsRuntime::configure_owned_process_tree(
            ResourceMetricsConfig::default(),
            child.0.id(),
        )
        .expect("configure the owned direct child process tree");
        let owned_target =
            crate::resource_metrics::collector::owned_process_tree_target(child.0.id()).unwrap();
        assert_eq!(
            owned_target.measurement_scope,
            ResourceMeasurementScope::OwnedProcessTree
        );
        let owned = collect_resource_metrics().unwrap();
        assert_eq!(owned.operating_system, expected_operating_system());
        assert!(
            owned
                .active_process_count
                .value()
                .is_some_and(|count| *count >= 1)
        );

        child.0.kill().unwrap();
        child.0.wait().unwrap();
        let terminated = collect_resource_metrics().unwrap();
        assert_eq!(terminated.active_process_count.value, None);
        assert_eq!(terminated.active_process_count.unit, None);
        drop(owned_runtime);
    }
}

#[test]
fn polling_rotates_complete_jsonl_records_and_retains_configured_file_count() {
    let _guard = lock_global_runtime();
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("resource-metrics.jsonl");
    let runtime = ResourceMetricsRuntime::configure(ResourceMetricsConfig {
        polling: ResourceMetricsPollingConfig {
            enabled: true,
            interval_millis: 10,
            retained_snapshots: 10,
        },
        file: ResourceMetricsFileConfig {
            enabled: true,
            path: output.clone(),
            max_file_size_bytes: 1,
            retained_files: 2,
        },
    })
    .unwrap();
    wait_for_history_length(5, Duration::from_secs(3));
    drop(runtime);

    let active_and_rotated = [
        output.clone(),
        directory.path().join("resource-metrics.1.jsonl"),
        directory.path().join("resource-metrics.2.jsonl"),
    ];
    assert!(active_and_rotated.iter().all(|path| path.is_file()));
    assert!(!directory.path().join("resource-metrics.3.jsonl").exists());
    let mut total_records = 0;
    for path in &active_and_rotated {
        let content = std::fs::read_to_string(path).unwrap();
        assert!(
            content.ends_with('\n'),
            "each JSONL record ends with a newline"
        );
        let records = content.lines().collect::<Vec<_>>();
        assert_eq!(records.len(), 1, "oversized records rotate one per file");
        assert_complete_jsonl_record(records[0]);
        total_records += records.len();
    }
    assert_eq!(total_records, 3);
    let oldest_to_newest = [
        directory.path().join("resource-metrics.2.jsonl"),
        directory.path().join("resource-metrics.1.jsonl"),
        output,
    ];
    let timestamps = oldest_to_newest
        .iter()
        .map(|path| {
            let record: Value = serde_json::from_str(&read_jsonl_lines(path)[0]).unwrap();
            DateTime::parse_from_rfc3339(record["cpu_total_time"]["timestamp"].as_str().unwrap())
                .unwrap()
                .timestamp_micros()
        })
        .collect::<Vec<_>>();
    assert!(timestamps.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn polling_shutdown_stops_file_records_after_drop() {
    let _guard = lock_global_runtime();
    for iteration in 0..3 {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("resource-metrics.jsonl");
        let runtime = ResourceMetricsRuntime::configure(ResourceMetricsConfig {
            polling: ResourceMetricsPollingConfig {
                enabled: true,
                interval_millis: 10,
                retained_snapshots: 20,
            },
            file: ResourceMetricsFileConfig {
                enabled: true,
                path: output.clone(),
                max_file_size_bytes: 1_000_000,
                retained_files: 1,
            },
        })
        .unwrap();
        wait_for_history_length(2, Duration::from_secs(2));
        drop(runtime);

        let stopped_record_count = read_jsonl_lines(&output).len();
        let stopped_history = resource_metrics_history().unwrap();
        std::thread::sleep(Duration::from_millis(80));
        assert_eq!(
            read_jsonl_lines(&output).len(),
            stopped_record_count,
            "polling iteration {iteration} wrote after runtime shutdown"
        );
        assert_eq!(resource_metrics_history().unwrap(), stopped_history);
    }
}

#[cfg(unix)]
#[test]
fn file_rotation_failure_is_isolated_and_polling_recovers_after_path_is_cleared() {
    let _guard = lock_global_runtime();
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("resource-metrics.jsonl");
    let blocked_rotation_path = directory.path().join("resource-metrics.1.jsonl");
    std::fs::create_dir(&blocked_rotation_path).unwrap();
    let runtime = ResourceMetricsRuntime::configure(ResourceMetricsConfig {
        polling: ResourceMetricsPollingConfig {
            enabled: true,
            interval_millis: 10,
            retained_snapshots: 100,
        },
        file: ResourceMetricsFileConfig {
            enabled: true,
            path: output.clone(),
            max_file_size_bytes: 1,
            retained_files: 1,
        },
    })
    .unwrap();
    let failed_write_history = wait_for_history_length(4, Duration::from_secs(3));
    let history_before_additional_failures = resource_metrics_history().unwrap().len();
    std::thread::sleep(Duration::from_millis(40));
    assert!(
        resource_metrics_history().unwrap().len() > history_before_additional_failures,
        "polling must continue even when rotation repeatedly fails"
    );
    let pre_recovery_records = read_jsonl_lines(&output);
    assert_eq!(
        pre_recovery_records.len(),
        1,
        "failed rotation leaves the last complete active record intact"
    );
    assert_complete_jsonl_record(&pre_recovery_records[0]);

    std::fs::remove_dir(&blocked_rotation_path).unwrap();
    let history_before_recovery = failed_write_history.len();
    let recovery_deadline = Instant::now() + Duration::from_secs(2);
    while (!blocked_rotation_path.is_file()
        || resource_metrics_history().unwrap().len() <= history_before_recovery)
        && Instant::now() < recovery_deadline
    {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        blocked_rotation_path.is_file(),
        "rotation recovers after obstruction removal"
    );
    let recovered_active_records = read_jsonl_lines(&output);
    let recovered_rotated_records = read_jsonl_lines(&blocked_rotation_path);
    assert_eq!(recovered_active_records.len(), 1);
    assert_eq!(recovered_rotated_records.len(), 1);
    assert_complete_jsonl_record(&recovered_active_records[0]);
    assert_complete_jsonl_record(&recovered_rotated_records[0]);
    drop(runtime);
}

#[cfg(unix)]
#[test]
fn failed_collection_preserves_latest_history_and_does_not_break_polling_shutdown() {
    let _guard = lock_global_runtime();
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("resource-metrics.jsonl");
    let mut child = ChildProcessGuard(
        std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("launch an owned child process"),
    );
    let runtime = ResourceMetricsRuntime::configure_owned_process_tree(
        ResourceMetricsConfig {
            polling: ResourceMetricsPollingConfig {
                enabled: true,
                interval_millis: 10,
                retained_snapshots: 100,
            },
            file: ResourceMetricsFileConfig {
                enabled: true,
                path: output.clone(),
                max_file_size_bytes: 1_000_000,
                retained_files: 1,
            },
        },
        child.0.id(),
    )
    .unwrap();
    wait_for_history_length(3, Duration::from_secs(3));
    child.0.kill().unwrap();
    child.0.wait().unwrap();
    std::thread::sleep(Duration::from_millis(60));
    let successful_history = resource_metrics_history().unwrap();
    let successful_latest = latest_resource_metrics().unwrap().unwrap();
    assert!(successful_history.len() >= 3);
    assert_eq!(successful_latest, *successful_history.last().unwrap());
    std::thread::sleep(Duration::from_millis(80));
    assert_eq!(resource_metrics_history().unwrap(), successful_history);
    assert_eq!(
        latest_resource_metrics().unwrap(),
        Some(successful_latest.clone())
    );
    let file_record_count = read_jsonl_lines(&output).len();
    assert_eq!(file_record_count, successful_history.len());

    drop(runtime);
    let fresh_after_runtime_drop =
        collect_resource_metrics().expect("fresh collection returns to the Relay process");
    assert!(
        fresh_after_runtime_drop
            .active_process_count
            .value()
            .is_some_and(|count| *count > 0),
        "unexpected application-process observation after dropping the owned-target runtime: {:?}",
        fresh_after_runtime_drop.active_process_count
    );
    assert_eq!(latest_resource_metrics().unwrap(), Some(successful_latest));
    assert_eq!(resource_metrics_history().unwrap(), successful_history);
    std::thread::sleep(Duration::from_millis(80));
    assert_eq!(read_jsonl_lines(&output).len(), file_record_count);
}

#[cfg(unix)]
struct ChildProcessGuard(std::process::Child);

#[cfg(unix)]
impl Drop for ChildProcessGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
