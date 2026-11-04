// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resource configuration and collector contracts.
use nemo_relay_resource_metrics::collector::{
    CollectedSnapshot, CollectionTarget, SamplingState, collect_with_state, current_process_target,
};
use nemo_relay_resource_metrics::config::*;
use nemo_relay_types::api::resource_metrics::{
    CapacityUnit, DurationUnit, ResourceMeasurementScope,
};
use nemo_relay_types::config_editor::{EditorConfig, EditorFieldKind};
use serde_json::json;
use std::path::PathBuf;
fn collect(target: &CollectionTarget, config: &ResourceMetricsConfig) -> CollectedSnapshot {
    collect_with_state(target, config, &mut SamplingState::default())
}

#[test]
fn resource_metrics_defaults_match_the_plugin_contract() {
    let config = ResourceMetricsConfig::default();
    assert_eq!(
        config.measurement_scope,
        ResourceMetricsMeasurementScope::RuntimeDefault
    );
    assert!(!config.polling.enabled);
    assert_eq!(config.polling.interval_millis, 5_000);
    assert!(config.cpu.enabled);
    assert!(config.memory.enabled);
    assert!(config.process.enabled);
    assert!(config.disk.enabled);
    assert!(config.disk.process_io);
    assert!(config.disk.filesystem_paths.is_empty());
    assert!(config.gpu.enabled);
    assert!(config.gpu.devices.is_empty());
    assert!(config.gpu.device_metrics);
    assert!(config.gpu.process_metrics);
}

#[test]
fn resource_metrics_configuration_editor_uses_nested_sections_and_typed_lists() {
    let schema = ResourceMetricsConfig::editor_schema();
    let measurement_scope = schema.field("measurement_scope").unwrap();
    assert_eq!(measurement_scope.kind, EditorFieldKind::Enum);
    assert_eq!(
        measurement_scope.enum_values,
        [
            "runtime_default",
            "global",
            "application_process",
            "process_tree"
        ]
    );
    assert_eq!(
        schema.field("polling").unwrap().kind,
        EditorFieldKind::Section
    );
    assert_eq!(schema.field("cpu").unwrap().kind, EditorFieldKind::Section);
    assert_eq!(
        schema.field("memory").unwrap().kind,
        EditorFieldKind::Section
    );
    assert_eq!(
        schema.field("process").unwrap().kind,
        EditorFieldKind::Section
    );
    assert_eq!(schema.field("disk").unwrap().kind, EditorFieldKind::Section);
    assert_eq!(schema.field("gpu").unwrap().kind, EditorFieldKind::Section);
    let polling = schema.field("polling").unwrap().schema().unwrap();
    assert_eq!(
        polling.field("enabled").unwrap().kind,
        EditorFieldKind::Boolean
    );
    assert_eq!(
        polling.field("interval_millis").unwrap().kind,
        EditorFieldKind::Integer
    );
    let disk = schema.field("disk").unwrap().schema().unwrap();
    let paths = disk.field("filesystem_paths").unwrap();
    assert_eq!(paths.kind, EditorFieldKind::List);
    assert_eq!(paths.list_item.unwrap().kind, EditorFieldKind::String);
    let gpu = schema.field("gpu").unwrap().schema().unwrap();
    let devices = gpu.field("devices").unwrap();
    assert_eq!(devices.kind, EditorFieldKind::List);
    assert_eq!(devices.list_item.unwrap().kind, EditorFieldKind::String);
}

#[test]
fn plugin_configuration_deserializes_the_shared_snake_case_schema() {
    let config: ResourceMetricsConfig = serde_json::from_value(json!({
        "polling": {"enabled": true, "interval_millis": 1250},
        "cpu": {"enabled": false},
        "memory": {"enabled": true},
        "process": {"enabled": false},
        "disk": {"enabled": true, "process_io": false, "filesystem_paths": ["/var"]},
        "gpu": {"enabled": true, "devices": ["GPU-123"], "device_metrics": false, "process_metrics": true}
    })).unwrap();
    assert!(config.polling.enabled);
    assert_eq!(config.polling.interval_millis, 1250);
    assert!(!config.cpu.enabled);
    assert!(!config.process.enabled);
    assert!(!config.disk.process_io);
    assert_eq!(config.disk.filesystem_paths, vec![PathBuf::from("/var")]);
    assert_eq!(config.gpu.devices, vec!["GPU-123"]);
    assert!(!config.gpu.device_metrics);
    assert!(config.gpu.process_metrics);
    assert_eq!(
        config.measurement_scope,
        ResourceMetricsMeasurementScope::RuntimeDefault
    );
}

#[test]
fn plugin_configuration_rejects_invalid_intervals_and_paths() {
    let mut config = ResourceMetricsConfig::default();
    config.polling.interval_millis = 0;
    assert!(config.validate().is_err());

    config = ResourceMetricsConfig::default();
    config
        .disk
        .filesystem_paths
        .push(PathBuf::from("relative/path"));
    assert!(config.validate().is_err());

    config.disk.filesystem_paths = vec![PathBuf::new()];
    assert!(config.validate().is_err());

    config = ResourceMetricsConfig {
        disk: ResourceMetricsDiskConfig {
            filesystem_paths: vec![PathBuf::from("/")],
            ..Default::default()
        },
        gpu: ResourceMetricsGpuConfig {
            devices: vec!["  ".into()],
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn gpu_selector_and_filesystem_paths_keep_empty_selector_semantics() {
    let config: ResourceMetricsConfig = serde_json::from_value(json!({
        "gpu": {"devices": [], "device_metrics": true, "process_metrics": false},
        "disk": {"filesystem_paths": []}
    }))
    .unwrap();
    assert!(config.gpu.devices.is_empty());
    assert!(config.gpu.device_metrics);
    assert!(!config.gpu.process_metrics);
    assert!(config.disk.filesystem_paths.is_empty());
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
fn global_scope_reports_system_memory_and_all_visible_processes() {
    let target = current_process_target(ResourceMeasurementScope::Global).unwrap();
    let snapshot = collect(&target, &ResourceMetricsConfig::default()).snapshot;
    assert_eq!(snapshot.measurement_scope, ResourceMeasurementScope::Global);
    let sampling = snapshot.process_sampling.as_ref().unwrap();
    assert!(sampling.visible_processes >= sampling.sampled_processes);
    assert!(sampling.sampled_processes > 0);
    assert!(snapshot.process.unwrap().active_count.unwrap().value > 0);
    let memory = snapshot.memory.unwrap();
    let total = memory.system_total.unwrap();
    assert!(total.value > 0);
    assert_eq!(total.unit, CapacityUnit::Kibibytes);
    assert!(memory.system_used.is_some());
    assert!(memory.system_available.is_some());
    assert!(memory.resident.is_none());
}

#[test]
fn collector_uses_canonical_units_and_serialization() {
    let target = current_process_target(ResourceMeasurementScope::ApplicationProcess).unwrap();
    let config = ResourceMetricsConfig::default();
    let snapshot = collect(&target, &config).snapshot;
    assert_eq!(
        snapshot.measurement_scope,
        ResourceMeasurementScope::ApplicationProcess
    );
    assert_eq!(
        snapshot
            .process_sampling
            .as_ref()
            .map(|sample| sample.visible_processes),
        Some(1)
    );
    assert_eq!(
        snapshot
            .process_sampling
            .as_ref()
            .map(|sample| sample.sampled_processes),
        Some(1)
    );
    assert_eq!(
        snapshot
            .cpu
            .as_ref()
            .unwrap()
            .user_time
            .as_ref()
            .unwrap()
            .unit,
        DurationUnit::Milliseconds
    );
    assert_eq!(
        snapshot
            .cpu
            .as_ref()
            .unwrap()
            .total_time
            .as_ref()
            .unwrap()
            .unit,
        DurationUnit::Milliseconds
    );
    assert_eq!(
        snapshot
            .memory
            .as_ref()
            .unwrap()
            .resident
            .as_ref()
            .unwrap()
            .unit,
        CapacityUnit::Kibibytes
    );
    let serialized = serde_json::to_value(&snapshot).unwrap();
    let timestamp = serialized["timestamp"].as_str().unwrap();
    assert!(chrono::DateTime::parse_from_rfc3339(timestamp).is_ok());
    assert_eq!(
        serialized["cpu"]["user_time"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        std::collections::BTreeSet::from(["unit".to_string(), "value".to_string()])
    );
    assert_eq!(serialized["measurement_scope"], "application_process");
}

#[test]
fn configured_filesystem_paths_produce_byte_capacity_measurements() {
    let target = current_process_target(ResourceMeasurementScope::ApplicationProcess).unwrap();
    let config = ResourceMetricsConfig {
        disk: ResourceMetricsDiskConfig {
            filesystem_paths: vec![std::env::temp_dir()],
            ..Default::default()
        },
        ..Default::default()
    };
    let snapshot = collect(&target, &config).snapshot;
    let filesystems = &snapshot.disk.as_ref().unwrap().filesystems;
    assert_eq!(filesystems.len(), 1);
    for measurement in [
        filesystems[0].total_capacity.as_ref().unwrap(),
        filesystems[0].available_capacity.as_ref().unwrap(),
        filesystems[0].free_capacity.as_ref().unwrap(),
    ] {
        assert_eq!(measurement.unit, CapacityUnit::Bytes);
        assert!(measurement.value > 0);
    }
}
