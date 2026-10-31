// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use nemo_relay_types::api::resource_metrics::{CapacityUnit, CpuUnit, DataUnit, DurationUnit};
use serde_json::json;

use crate::api::event::Event;
use crate::api::resource_metrics::{
    ResourceMetricsConfig, ResourceMetricsDiskConfig, ResourceMetricsGpuConfig,
};
use crate::api::runtime::{
    NemoRelayContextState, create_scope_stack, flush_subscribers, global_context,
    set_thread_scope_stack,
};
use crate::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use crate::api::subscriber::{
    deregister_subscriber, register_subscriber, scope_register_subscriber,
};

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
            DurationUnit::Milliseconds,
        ));
    snapshot.network = Some(crate::api::resource_metrics::NetworkMetrics {
        measurement_scope: crate::api::resource_metrics::ResourceMeasurementScope::Global,
        system: crate::api::resource_metrics::NetworkTrafficMetrics {
            received_data: Some(crate::api::resource_metrics::ResourceMeasurement::new(
                1_000_000_u64,
                DataUnit::Bytes,
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
    units.cpu.user_time = crate::api::resource_metrics::DurationUnit::Seconds;
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

    for _ in 0..64 {
        super::register_agent_scope(active_scope.clone(), active_stack.clone());
    }
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
fn polling_releases_an_abandoned_agent_stack_and_uses_its_root() {
    let _runtime_lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
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
        crate::resource_metrics::manager::ResourceMetricsRuntime::configure_current_process(
            config,
            super::polling_snapshot_sink().unwrap(),
        )
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

use crate::resource_metrics::snapshot_fixture;

#[test]
fn complete_snapshot_projects_filesystems_accelerators_and_limit_events_as_gauges() {
    let snapshot = snapshot_fixture::full_snapshot();
    let measurements = super::metric_measurements(&snapshot);
    assert!(
        measurements
            .iter()
            .all(|value| value.kind == super::MetricKind::Gauge)
    );
    let limits: Vec<_> = measurements
        .iter()
        .filter(|value| value.name == "nemo.relay.resource.limit.event_count")
        .collect();
    assert_eq!(limits.len(), 3);
    for (resource, event) in [
        ("cpu", "throttled"),
        ("memory", "out_of_memory"),
        ("processes", "maximum"),
    ] {
        assert!(limits.iter().any(|value| {
            let attributes = value.attributes.as_ref().unwrap();
            attributes["nemo_relay.resource.limit.resource"] == resource
                && attributes["nemo_relay.resource.limit.event"] == event
                && value.unit.as_deref() == Some("events")
        }));
    }
    let filesystem = measurements
        .iter()
        .find(|value| value.name == "nemo.relay.resource.disk.filesystem.total_capacity")
        .unwrap();
    assert_eq!(
        filesystem.attributes.as_ref().unwrap()["nemo_relay.resource.disk.filesystem_path"],
        "/fixture"
    );
    let gpu = measurements
        .iter()
        .find(|value| value.name == "nemo.relay.resource.accelerator.process.compute_utilization")
        .unwrap();
    assert_eq!(gpu.value, json!(50.0));
    assert_eq!(
        gpu.attributes.as_ref().unwrap()["nemo_relay.resource.process_id"],
        42
    );
    assert_eq!(
        gpu.attributes.as_ref().unwrap()["nemo_relay.resource.accelerator.device_identifier"],
        "GPU-fixture"
    );
    let network = measurements
        .iter()
        .filter(|value| value.name.starts_with("nemo.relay.resource.network."))
        .collect::<Vec<_>>();
    assert_eq!(network.len(), 16);
    assert!(network.iter().all(
        |value| value.attributes.as_ref().unwrap()["nemo_relay.resource.measurement_scope"]
            == "global"
    ));
}

#[test]
fn metric_projection_omits_nonfinite_values_and_integers_outside_the_exporter_range() {
    use nemo_relay_types::api::resource_metrics::{ResourceMeasurement, ResourceMetricValue};
    let mut snapshot = crate::resource_metrics::snapshot_fixture::full_snapshot();
    snapshot.cpu.as_mut().unwrap().consumption_rate = Some(ResourceMeasurement::new(
        f64::NAN,
        CpuUnit::LogicalProcessors,
    ));
    snapshot.cpu.as_mut().unwrap().total_time = Some(ResourceMeasurement::new(
        u64::MAX,
        DurationUnit::Milliseconds,
    ));
    snapshot.memory.as_mut().unwrap().resident = Some(ResourceMeasurement::new(
        ResourceMetricValue::Decimal(f64::INFINITY),
        CapacityUnit::Kibibytes,
    ));
    let metrics = super::metric_measurements(&snapshot);
    for excluded in [
        "nemo.relay.resource.cpu.consumption_rate",
        "nemo.relay.resource.cpu.total_time",
        "nemo.relay.resource.memory.resident",
    ] {
        assert!(!metrics.iter().any(|metric| metric.name == excluded));
    }
    assert!(!metrics.is_empty());
}

#[test]
fn failed_agent_delivery_does_not_leak_the_sample_to_the_runtime_root() {
    let _runtime_lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    flush_subscribers().unwrap();
    crate::shared_runtime::reset_runtime_owner_for_tests();
    *global_context().write().unwrap() = NemoRelayContextState::new();
    super::AGENT_SCOPES.lock().unwrap().clear();
    let stack = create_scope_stack();
    set_thread_scope_stack(stack.clone());
    push_scope(
        PushScopeParams::builder()
            .name("unavailable-agent-context")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    let observed = Arc::new(Mutex::new(Vec::<Event>::new()));
    let events = Arc::clone(&observed);
    register_subscriber(
        "resource-metrics-failed-agent-test",
        Arc::new(move |event| events.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    flush_subscribers().unwrap();
    observed.lock().unwrap().clear();
    let poisoned = Arc::clone(&stack);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _guard = poisoned.write().unwrap();
        panic!("simulate an unavailable Agent context");
    }));
    assert!(result.is_err());
    super::emit_resource_metrics_snapshot(
        &crate::resource_metrics::snapshot_fixture::full_snapshot(),
        &create_scope_stack(),
    );
    flush_subscribers().unwrap();
    assert!(observed.lock().unwrap().is_empty());
    super::AGENT_SCOPES.lock().unwrap().clear();
    deregister_subscriber("resource-metrics-failed-agent-test").unwrap();
    set_thread_scope_stack(create_scope_stack());
}

#[test]
fn resource_metric_delivery_cleans_closed_registry_entries_and_handles_unavailable_root_context() {
    let _runtime_lock = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    flush_subscribers().unwrap();
    crate::shared_runtime::reset_runtime_owner_for_tests();
    *global_context().write().unwrap() = NemoRelayContextState::new();
    let stack = create_scope_stack();
    super::AGENT_SCOPES
        .lock()
        .unwrap()
        .insert(uuid::Uuid::nil(), Arc::downgrade(&stack));
    let poisoned = Arc::clone(&stack);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _guard = poisoned.write().unwrap();
            panic!("simulate an unavailable root context");
        }))
        .is_err()
    );
    super::emit_resource_metrics_snapshot(
        &crate::resource_metrics::snapshot_fixture::full_snapshot(),
        &stack,
    );
    assert!(super::AGENT_SCOPES.lock().unwrap().is_empty());
    set_thread_scope_stack(create_scope_stack());
}
