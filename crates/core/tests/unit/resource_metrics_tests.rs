// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use nemo_relay_types::api::resource_metrics::ResourceMeasurementUnit;
use serde_json::json;

use crate::api::event::Event;
use crate::api::resource_metrics::{
    ResourceMetricsConfig, ResourceMetricsDiskConfig, ResourceMetricsGpuConfig,
    ResourceMetricsMeasurementScope,
};
use crate::api::runtime::{
    NemoRelayContextState, create_scope_stack, flush_subscribers, global_context,
    set_thread_scope_stack,
};
use crate::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use crate::api::subscriber::{
    deregister_subscriber, register_subscriber, scope_register_subscriber,
};
use crate::config_editor::{EditorConfig, EditorFieldKind};
use crate::plugin::{
    PluginComponentSpec, PluginConfig, test_close_plugin_host, test_initialize_plugin_host_exact,
};

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
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
fn global_scope_reports_system_memory_and_all_visible_processes() {
    let target = crate::resource_metrics::collector::current_process_target(
        crate::api::resource_metrics::ResourceMeasurementScope::Global,
    )
    .unwrap();
    let snapshot =
        crate::resource_metrics::collector::collect(&target, &ResourceMetricsConfig::default())
            .snapshot;
    assert_eq!(
        snapshot.measurement_scope,
        crate::api::resource_metrics::ResourceMeasurementScope::Global
    );
    let sampling = snapshot.process_sampling.as_ref().unwrap();
    assert!(sampling.visible_processes >= sampling.sampled_processes);
    assert!(sampling.sampled_processes > 0);
    assert!(snapshot.process.unwrap().active_count.unwrap().value > 0);
    let memory = snapshot.memory.unwrap();
    let total = memory.system_total.unwrap();
    assert!(total.value > 0);
    assert_eq!(total.unit, ResourceMeasurementUnit::Kibibytes);
    assert!(memory.system_used.is_some());
    assert!(memory.system_available.is_some());
    assert!(memory.resident.is_none());
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
fn collector_uses_canonical_units_and_serialization() {
    let target = crate::resource_metrics::collector::current_process_target(
        crate::api::resource_metrics::ResourceMeasurementScope::ApplicationProcess,
    )
    .unwrap();
    let config = ResourceMetricsConfig::default();
    let snapshot = crate::resource_metrics::collector::collect(&target, &config).snapshot;
    assert_eq!(
        snapshot.measurement_scope,
        crate::api::resource_metrics::ResourceMeasurementScope::ApplicationProcess
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
        ResourceMeasurementUnit::Milliseconds
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
        ResourceMeasurementUnit::Milliseconds
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
        ResourceMeasurementUnit::Kibibytes
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
fn collector_projects_coverage_and_preserves_disabled_category_shape() {
    let target = crate::resource_metrics::collector::current_process_target(
        crate::api::resource_metrics::ResourceMeasurementScope::ApplicationProcess,
    )
    .unwrap();
    let config = ResourceMetricsConfig::default();
    let snapshot = crate::resource_metrics::collector::collect(&target, &config).snapshot;
    let measurement_names = super::metric_measurements(&snapshot)
        .into_iter()
        .map(|measurement| measurement.name)
        .collect::<std::collections::HashSet<_>>();
    assert!(measurement_names.contains("nemo.relay.resource.process_sampling.visible_count"));
    assert!(measurement_names.contains("nemo.relay.resource.process_sampling.sampled_count"));
    let coverage_marks = super::metric_measurements(&snapshot)
        .into_iter()
        .filter(|measurement| {
            measurement.name == "nemo.relay.resource.process_sampling.field_sampled_count"
        })
        .collect::<Vec<_>>();
    let coverage = &snapshot
        .process_sampling
        .as_ref()
        .unwrap()
        .field_sampled_processes;
    assert_eq!(coverage_marks.len(), coverage.len());
    for mark in coverage_marks {
        let field = mark.attributes.as_ref().unwrap()["nemo_relay.resource.field"]
            .as_str()
            .unwrap();
        assert_eq!(mark.value, json!(coverage[field]));
        assert_eq!(mark.unit.as_deref(), Some("processes"));
    }

    let disabled = ResourceMetricsConfig {
        cpu: crate::api::resource_metrics::ResourceMetricsCpuConfig { enabled: false },
        memory: crate::api::resource_metrics::ResourceMetricsMemoryConfig { enabled: false },
        process: crate::api::resource_metrics::ResourceMetricsProcessConfig { enabled: false },
        disk: ResourceMetricsDiskConfig {
            enabled: false,
            ..Default::default()
        },
        gpu: ResourceMetricsGpuConfig {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let snapshot = crate::resource_metrics::collector::collect(&target, &disabled).snapshot;
    assert!(snapshot.cpu.is_none());
    assert!(snapshot.memory.is_none());
    assert!(snapshot.process.is_none());
    assert!(snapshot.disk.is_none());
    assert!(snapshot.gpu.is_none());
}

#[test]
fn metric_marks_use_the_converted_snapshot_units() {
    let target = crate::resource_metrics::collector::current_process_target(
        crate::api::resource_metrics::ResourceMeasurementScope::ApplicationProcess,
    )
    .unwrap();
    let config = ResourceMetricsConfig::default();
    let mut snapshot = crate::resource_metrics::collector::collect(&target, &config).snapshot;
    snapshot.cpu.as_mut().unwrap().user_time =
        Some(crate::api::resource_metrics::ResourceMeasurement::new(
            1_500_u64,
            ResourceMeasurementUnit::Milliseconds,
        ));
    snapshot.network = Some(crate::api::resource_metrics::NetworkMetrics {
        measurement_scope: crate::api::resource_metrics::ResourceMeasurementScope::Global,
        system: crate::api::resource_metrics::NetworkTrafficMetrics {
            received_data: Some(crate::api::resource_metrics::ResourceMeasurement::new(
                1_000_000_u64,
                ResourceMeasurementUnit::Bytes,
            )),
            transmitted_data: None,
            receive_throughput: None,
            transmit_throughput: None,
            received_packets: None,
            transmitted_packets: None,
            receive_errors: None,
            transmit_errors: None,
        },
        interfaces: Vec::new(),
    });
    let mut units = config.units;
    units.cpu.user_time = crate::api::resource_metrics::TimeUnit::Seconds;
    units.network.system.received_data = crate::api::resource_metrics::DataUnit::Megabytes;
    crate::resource_metrics::units::convert_snapshot(&mut snapshot, &units);
    let measurements = super::metric_measurements(&snapshot);
    let measurement = measurements
        .iter()
        .find(|measurement| measurement.name == "nemo.relay.resource.cpu.user_time")
        .unwrap();
    assert_eq!(measurement.unit.as_deref(), Some("seconds"));
    assert_eq!(measurement.value, json!(1.5));
    let network = measurements
        .iter()
        .find(|measurement| measurement.name == "nemo.relay.resource.network.system.received_data")
        .unwrap();
    assert_eq!(network.unit.as_deref(), Some("megabytes"));
    assert_eq!(network.value, json!(1.0));
    assert_eq!(
        network.attributes.as_ref().unwrap()["nemo_relay.resource.measurement_scope"],
        "global"
    );
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
fn configured_filesystem_paths_produce_byte_capacity_measurements() {
    let target = crate::resource_metrics::collector::current_process_target(
        crate::api::resource_metrics::ResourceMeasurementScope::ApplicationProcess,
    )
    .unwrap();
    let config = ResourceMetricsConfig {
        disk: ResourceMetricsDiskConfig {
            filesystem_paths: vec![std::env::temp_dir()],
            ..Default::default()
        },
        ..Default::default()
    };
    let snapshot = crate::resource_metrics::collector::collect(&target, &config).snapshot;
    let filesystems = &snapshot.disk.as_ref().unwrap().filesystems;
    assert_eq!(filesystems.len(), 1);
    for measurement in [
        filesystems[0].total_capacity.as_ref().unwrap(),
        filesystems[0].available_capacity.as_ref().unwrap(),
        filesystems[0].free_capacity.as_ref().unwrap(),
    ] {
        assert_eq!(measurement.unit, ResourceMeasurementUnit::Bytes);
        assert!(measurement.value > 0);
    }
}

#[test]
fn resource_metrics_plugin_is_discovered_as_a_builtin() {
    assert!(
        crate::plugin::list_plugin_kinds()
            .iter()
            .any(|kind| kind == "resource_metrics")
    );
}

#[test]
fn registering_agent_scope_prunes_abandoned_stacks_without_polling() {
    let _runtime_lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let abandoned_stack = create_scope_stack();
    let abandoned = Arc::downgrade(&abandoned_stack);
    drop(abandoned_stack);

    let active_stack = create_scope_stack();
    let active_scope = active_stack.read().unwrap().scopes()[0].clone();
    let mut scopes = super::AGENT_SCOPES
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    scopes.clear();
    scopes.insert(uuid::Uuid::nil(), abandoned);
    drop(scopes);

    super::register_agent_scope(active_scope.clone(), active_stack.clone());
    let mut scopes = super::AGENT_SCOPES
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    assert_eq!(scopes.len(), 1);
    assert!(scopes.contains_key(&active_scope.uuid));
    scopes.clear();
}

#[test]
fn resource_metric_marks_fan_out_to_agent_scopes_and_fall_back_to_the_runtime_root() {
    let _runtime_lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    flush_subscribers().unwrap();
    crate::shared_runtime::reset_runtime_owner_for_tests();
    *global_context().write().unwrap() = NemoRelayContextState::new();
    super::AGENT_SCOPES
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clear();

    let stack = create_scope_stack();
    set_thread_scope_stack(stack.clone());
    let root_uuid = stack.read().unwrap().scopes()[0].uuid;
    let observed = Arc::new(Mutex::new(Vec::<Event>::new()));
    let observed_events = Arc::clone(&observed);
    register_subscriber(
        "resource-metrics-fanout-test",
        Arc::new(move |event| observed_events.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let first_agent = push_scope(
        PushScopeParams::builder()
            .name("first-agent")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    let first_local = Arc::new(Mutex::new(Vec::<uuid::Uuid>::new()));
    let first_local_events = Arc::clone(&first_local);
    scope_register_subscriber(
        &first_agent.uuid,
        "resource-metrics-first-agent-test",
        Arc::new(move |event| {
            if event.name() == super::RESOURCE_METRICS_EVENT_NAME {
                first_local_events
                    .lock()
                    .unwrap()
                    .push(event.parent_uuid().unwrap());
            }
        }),
    )
    .unwrap();
    let custom_scope = push_scope(
        PushScopeParams::builder()
            .name("custom-scope")
            .scope_type(ScopeType::Custom)
            .build(),
    )
    .unwrap();
    let second_agent = push_scope(
        PushScopeParams::builder()
            .name("second-agent")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    let second_local = Arc::new(Mutex::new(Vec::<uuid::Uuid>::new()));
    let second_local_events = Arc::clone(&second_local);
    scope_register_subscriber(
        &second_agent.uuid,
        "resource-metrics-second-agent-test",
        Arc::new(move |event| {
            if event.name() == super::RESOURCE_METRICS_EVENT_NAME {
                second_local_events
                    .lock()
                    .unwrap()
                    .push(event.parent_uuid().unwrap());
            }
        }),
    )
    .unwrap();

    let target = crate::resource_metrics::collector::current_process_target(
        crate::api::resource_metrics::ResourceMeasurementScope::ApplicationProcess,
    )
    .unwrap();
    let snapshot =
        crate::resource_metrics::collector::collect(&target, &ResourceMetricsConfig::default())
            .snapshot;
    super::emit_resource_metrics_snapshot(&snapshot, &stack);
    flush_subscribers().unwrap();

    let marks = observed
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.name() == super::RESOURCE_METRICS_EVENT_NAME)
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(marks.len(), 2);
    assert_eq!(
        marks
            .iter()
            .filter_map(Event::parent_uuid)
            .collect::<std::collections::HashSet<_>>(),
        std::collections::HashSet::from([first_agent.uuid, second_agent.uuid])
    );
    assert_eq!(marks[0].data(), marks[1].data());
    assert_eq!(first_local.lock().unwrap().len(), 2);
    assert_eq!(*second_local.lock().unwrap(), vec![second_agent.uuid]);

    pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&second_agent.uuid)
            .build(),
    )
    .unwrap();
    pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&custom_scope.uuid)
            .build(),
    )
    .unwrap();
    pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&first_agent.uuid)
            .build(),
    )
    .unwrap();
    let custom_only_scope = push_scope(
        PushScopeParams::builder()
            .name("custom-only")
            .scope_type(ScopeType::Custom)
            .build(),
    )
    .unwrap();
    super::emit_resource_metrics_snapshot(&snapshot, &stack);
    flush_subscribers().unwrap();

    let marks = observed
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.name() == super::RESOURCE_METRICS_EVENT_NAME)
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(marks.len(), 3);
    assert_eq!(marks[2].parent_uuid(), Some(root_uuid));
    assert_eq!(marks[0].data(), marks[2].data());

    pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&custom_only_scope.uuid)
            .build(),
    )
    .unwrap();
    let abandoned_stack = create_scope_stack();
    let abandoned_stack_weak = Arc::downgrade(&abandoned_stack);
    set_thread_scope_stack(abandoned_stack.clone());
    let abandoned_agent = push_scope(
        PushScopeParams::builder()
            .name("abandoned-agent")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    set_thread_scope_stack(stack.clone());
    drop(abandoned_stack);
    assert!(abandoned_stack_weak.upgrade().is_none());
    super::emit_resource_metrics_snapshot(&snapshot, &stack);
    flush_subscribers().unwrap();
    assert!(
        !super::AGENT_SCOPES
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .contains_key(&abandoned_agent.uuid)
    );
    let marks = observed.lock().unwrap();
    assert_eq!(marks.last().and_then(Event::parent_uuid), Some(root_uuid));
    drop(marks);
    flush_subscribers().unwrap();
    deregister_subscriber("resource-metrics-fanout-test").unwrap();
}

#[test]
fn plugin_activation_controls_on_demand_collection_polling_cadence_and_teardown() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let _runtime_lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    test_close_plugin_host().unwrap();
    crate::shared_runtime::reset_runtime_owner_for_tests();
    *global_context().write().unwrap() = NemoRelayContextState::new();
    set_thread_scope_stack(create_scope_stack());

    let (poll_tx, poll_rx) = mpsc::channel::<Instant>();
    let plugin_scope_events = Arc::new(Mutex::new(0_usize));
    let plugin_scope_events_for_subscriber = Arc::clone(&plugin_scope_events);
    register_subscriber(
        "resource-metrics-polling-test",
        Arc::new(move |event| {
            if event.name() == super::RESOURCE_METRICS_EVENT_NAME {
                let mut count = plugin_scope_events_for_subscriber.lock().unwrap();
                *count = count.saturating_add(1);
                let _ = poll_tx.send(Instant::now());
            }
        }),
    )
    .unwrap();

    futures::executor::block_on(test_initialize_plugin_host_exact(
        resource_metrics_host_config(false, false, 250),
    ))
    .unwrap();
    assert!(runtime.block_on(super::collect()).is_err());
    test_close_plugin_host().unwrap();

    futures::executor::block_on(test_initialize_plugin_host_exact(
        resource_metrics_host_config(true, false, 250),
    ))
    .unwrap();
    let snapshot = runtime.block_on(super::collect()).unwrap();
    assert_ne!(
        snapshot.operating_system,
        crate::api::resource_metrics::ResourceOperatingSystem::Unsupported
    );
    assert_eq!(
        snapshot.measurement_scope,
        crate::api::resource_metrics::ResourceMeasurementScope::ProcessTree
    );
    assert!(poll_rx.recv_timeout(Duration::from_millis(100)).is_err());
    test_close_plugin_host().unwrap();
    assert!(runtime.block_on(super::collect()).is_err());

    futures::executor::block_on(test_initialize_plugin_host_exact(
        resource_metrics_host_config(true, true, 250),
    ))
    .unwrap();
    let first = poll_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let second = poll_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(second.duration_since(first) >= Duration::from_millis(200));
    test_close_plugin_host().unwrap();
    flush_subscribers().unwrap();
    while poll_rx.try_recv().is_ok() {}
    assert!(poll_rx.recv_timeout(Duration::from_millis(100)).is_err());
    assert!(runtime.block_on(super::collect()).is_err());
    assert!(*plugin_scope_events.lock().unwrap() >= 2);

    deregister_subscriber("resource-metrics-polling-test").unwrap();
    flush_subscribers().unwrap();
}

#[test]
fn polling_releases_an_abandoned_agent_stack_and_uses_its_root() {
    let _runtime_lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    test_close_plugin_host().unwrap();
    crate::shared_runtime::reset_runtime_owner_for_tests();
    *global_context().write().unwrap() = NemoRelayContextState::new();
    super::AGENT_SCOPES
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clear();

    let stack = create_scope_stack();
    let weak_stack = Arc::downgrade(&stack);
    let root_uuid = stack.read().unwrap().scopes()[0].uuid;
    set_thread_scope_stack(stack.clone());
    let agent = push_scope(
        PushScopeParams::builder()
            .name("temporary-agent")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    let (tx, rx) = mpsc::channel();
    register_subscriber(
        "resource-metrics-abandoned-stack-test",
        Arc::new(move |event| {
            if event.name() == super::RESOURCE_METRICS_EVENT_NAME {
                let _ = tx.send(event.parent_uuid());
            }
        }),
    )
    .unwrap();

    let mut config = ResourceMetricsConfig::default();
    config.polling.enabled = true;
    config.polling.interval_millis = 200;
    let runtime =
        crate::resource_metrics::manager::ResourceMetricsRuntime::configure_current_process(config)
            .unwrap();
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(2)).unwrap(),
        Some(agent.uuid)
    );

    set_thread_scope_stack(create_scope_stack());
    drop(stack);
    let deadline = Instant::now() + Duration::from_secs(2);
    while weak_stack.upgrade().is_some() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(weak_stack.upgrade().is_none());
    while rx.try_recv().is_ok() {}
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(2)).unwrap(),
        Some(root_uuid)
    );
    assert!(
        !super::AGENT_SCOPES
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .contains_key(&agent.uuid)
    );

    drop(runtime);
    deregister_subscriber("resource-metrics-abandoned-stack-test").unwrap();
    flush_subscribers().unwrap();
}

fn resource_metrics_host_config(
    component_enabled: bool,
    polling_enabled: bool,
    interval_millis: u64,
) -> PluginConfig {
    PluginConfig {
        version: 1,
        components: vec![PluginComponentSpec {
            kind: "resource_metrics".into(),
            enabled: component_enabled,
            config: json!({
                "polling": {
                    "enabled": polling_enabled,
                    "interval_millis": interval_millis
                }
            })
            .as_object()
            .unwrap()
            .clone(),
        }],
        policy: Default::default(),
    }
}
