// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use serde_json::json;

use crate::test_support::plugin_host::{test_close_plugin_host, test_initialize_plugin_host_exact};
use nemo_relay_plugin_host::{PluginComponentSpec, PluginConfig};
use nemo_relay_runtime::api::runtime::{
    NemoRelayContextState, create_scope_stack, flush_subscribers, global_context,
    set_thread_scope_stack,
};
use nemo_relay_runtime::api::subscriber::{deregister_subscriber, register_subscriber};

#[test]
fn resource_metrics_plugin_is_discovered_as_a_builtin() {
    assert!(
        nemo_relay_plugin_host::list_plugin_kinds()
            .iter()
            .any(|kind| kind == "resource_metrics")
    );
}

#[test]
fn plugin_activation_controls_on_demand_collection_polling_cadence_and_teardown() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let _runtime_lock = crate::test_support::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    test_close_plugin_host().unwrap();
    *global_context().write().unwrap() = NemoRelayContextState::new();
    set_thread_scope_stack(create_scope_stack());

    let (poll_tx, poll_rx) = mpsc::channel::<Instant>();
    let plugin_scope_events = Arc::new(Mutex::new(0_usize));
    let plugin_scope_events_for_subscriber = Arc::clone(&plugin_scope_events);
    register_subscriber(
        "resource-metrics-polling-test",
        Arc::new(move |event| {
            if event.name()
                == nemo_relay_runtime::api::resource_metrics::RESOURCE_METRICS_EVENT_NAME
            {
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
    assert!(
        runtime
            .block_on(nemo_relay_resource_metrics::manager::collect_on_demand())
            .is_err()
    );
    test_close_plugin_host().unwrap();

    futures::executor::block_on(test_initialize_plugin_host_exact(
        resource_metrics_host_config(true, false, 250),
    ))
    .unwrap();
    let snapshot = runtime
        .block_on(nemo_relay_resource_metrics::manager::collect_on_demand())
        .unwrap();
    assert_ne!(
        snapshot.operating_system,
        nemo_relay_runtime::api::resource_metrics::ResourceOperatingSystem::Unsupported
    );
    assert_eq!(
        snapshot.measurement_scope,
        nemo_relay_runtime::api::resource_metrics::ResourceMeasurementScope::ProcessTree
    );
    assert!(poll_rx.recv_timeout(Duration::from_millis(100)).is_err());
    test_close_plugin_host().unwrap();
    assert!(
        runtime
            .block_on(nemo_relay_resource_metrics::manager::collect_on_demand())
            .is_err()
    );

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
    assert!(
        runtime
            .block_on(nemo_relay_resource_metrics::manager::collect_on_demand())
            .is_err()
    );
    assert!(*plugin_scope_events.lock().unwrap() >= 2);

    deregister_subscriber("resource-metrics-polling-test").unwrap();
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

#[test]
#[cfg(not(windows))]
fn cli_launch_guards_select_the_owned_tree_and_restore_the_application_target() {
    struct Child(std::process::Child);
    impl Drop for Child {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let _runtime_lock = crate::test_support::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    test_close_plugin_host().unwrap();
    *global_context().write().unwrap() = NemoRelayContextState::new();
    set_thread_scope_stack(create_scope_stack());
    let launch = nemo_relay_resource_metrics::manager::prepare_cli_owned_process_tree().unwrap();
    assert!(nemo_relay_resource_metrics::manager::prepare_cli_owned_process_tree().is_err());
    let child = Child(
        std::process::Command::new("/bin/sleep")
            .arg("60")
            .spawn()
            .unwrap(),
    );
    assert!(
        nemo_relay_resource_metrics::manager::target_owned_process_tree(child.0.id())
            .unwrap()
            .is_none()
    );
    assert!(
        nemo_relay_resource_metrics::manager::target_owned_process_tree(std::process::id())
            .is_err()
    );
    let mut config = resource_metrics_host_config(true, false, 250);
    config.components[0]
        .config
        .insert("measurement_scope".into(), json!("process_tree"));
    futures::executor::block_on(test_initialize_plugin_host_exact(config)).unwrap();
    let target = nemo_relay_resource_metrics::manager::target_owned_process_tree(child.0.id())
        .unwrap()
        .unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let snapshot = runtime
        .block_on(nemo_relay_resource_metrics::manager::collect_on_demand())
        .unwrap();
    assert_eq!(
        snapshot.measurement_scope,
        nemo_relay_runtime::api::resource_metrics::ResourceMeasurementScope::ProcessTree
    );
    assert!(snapshot.process.unwrap().active_count.unwrap().value >= 1_u64);
    drop(target);
    assert!(
        runtime
            .block_on(nemo_relay_resource_metrics::manager::collect_on_demand())
            .is_err()
    );
    let target = nemo_relay_resource_metrics::manager::target_owned_process_tree(child.0.id())
        .unwrap()
        .unwrap();
    drop(launch);
    drop(target);
    let snapshot = runtime
        .block_on(nemo_relay_resource_metrics::manager::collect_on_demand())
        .unwrap();
    assert_eq!(
        snapshot.measurement_scope,
        nemo_relay_runtime::api::resource_metrics::ResourceMeasurementScope::ProcessTree
    );
    test_close_plugin_host().unwrap();
}
