// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Checked native runtime logging and scope propagation.
use nemo_relay_plugin::{PluginLogLevel, PluginRuntime};
use nemo_relay_plugin_host::dynamic::native_checked::HostRuntime;
use std::sync::Mutex;
static LEVELS: Mutex<Vec<log::Level>> = Mutex::new(Vec::new());
struct Logger;
impl log::Log for Logger {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.target() == "nemo_relay.plugin.native.review" {
            LEVELS.lock().unwrap().push(record.level());
        }
    }
    fn flush(&self) {}
}
#[tokio::test]
async fn sdk_log_levels_reach_the_host_unchanged() {
    static LOGGER: Logger = Logger;
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Trace);
    let owner = HostRuntime::new("review".into()).unwrap();
    let runtime = PluginRuntime::from_abi(owner.runtime_box());
    let results = [
        PluginLogLevel::Error,
        PluginLogLevel::Warn,
        PluginLogLevel::Info,
        PluginLogLevel::Debug,
        PluginLogLevel::Trace,
    ]
    .map(|level| runtime.log(level, "review", "message", None));
    let observed = LEVELS.lock().unwrap().clone();
    owner.shutdown().await.unwrap();
    assert_eq!(
        observed,
        [
            log::Level::Error,
            log::Level::Warn,
            log::Level::Info,
            log::Level::Debug,
            log::Level::Trace
        ],
        "SDK results: {results:?}"
    );
}

#[tokio::test]
async fn asynchronous_runtime_calls_preserve_explicit_and_implicit_scopes() {
    use nemo_relay_plugin::abi::Json;
    use nemo_relay_runtime::api::runtime::{create_scope_stack, with_scope_stack};
    let owner = HostRuntime::new("scope-test".into()).unwrap();
    let raw = owner.runtime_box();
    let implicit = raw.call_sync("scope.snapshot".into(), Json::Null);
    assert_eq!(
        implicit,
        raw.call("scope.snapshot".into(), Json::Null).await
    );
    let (synchronous, asynchronous) = with_scope_stack(create_scope_stack(), || {
        (
            raw.call_sync("scope.snapshot".into(), Json::Null),
            raw.call("scope.snapshot".into(), Json::Null),
        )
    });
    assert_ne!(synchronous, implicit);
    assert_eq!(synchronous, asynchronous.await);
    owner.shutdown().await.unwrap();
}

#[tokio::test]
async fn native_scope_timestamps_preserve_signed_microseconds() {
    use nemo_relay_runtime::api::{
        event::Event,
        runtime::{create_scope_stack, with_scope_stack},
        scope::ScopeType,
        subscriber::{deregister_subscriber, flush_subscribers, register_subscriber},
    };
    use std::{
        sync::Arc,
        time::{Duration, UNIX_EPOCH},
    };
    let owner = HostRuntime::new("signed-timestamps".into()).unwrap();
    let runtime = PluginRuntime::from_abi(owner.runtime_box());
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "signed-native-timestamps",
        Arc::new(move |event| {
            if event.name() == "signed-native-scope" {
                captured.lock().unwrap().push(event.clone());
            }
        }),
    )
    .unwrap();
    with_scope_stack(create_scope_stack(), || {
        let mut guard = runtime
            .scope_at(
                "signed-native-scope",
                ScopeType::Custom,
                None,
                None,
                None,
                UNIX_EPOCH - Duration::from_nanos(1),
            )
            .unwrap();
        guard
            .close_at(None, None, UNIX_EPOCH - Duration::from_nanos(999))
            .unwrap();
    });
    flush_subscribers().unwrap();
    deregister_subscriber("signed-native-timestamps").unwrap();
    owner.shutdown().await.unwrap();
    let events = events.lock().unwrap();
    assert_eq!(events.len(), 2);
    for event in events.iter() {
        assert_eq!(event.timestamp().timestamp_micros(), -1);
    }
}

#[tokio::test]
async fn native_scope_and_mark_options_preserve_flags_parents_and_times() {
    use nemo_relay_plugin::{
        DataSchema, EmitMarkOptions, LogSeverity, ScopeAttributes, ScopeOptions, ScopeType,
    };
    use nemo_relay_runtime::api::{
        event::Event,
        runtime::{create_scope_stack, with_scope_stack},
        scope::get_handle,
        subscriber::{deregister_subscriber, flush_subscribers, register_subscriber},
    };
    use std::{
        sync::Arc,
        time::{Duration, UNIX_EPOCH},
    };
    let owner = HostRuntime::new("native-options".into()).unwrap();
    let runtime = PluginRuntime::from_abi(owner.runtime_box());
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "native-options-events",
        Arc::new(move |event| {
            if event.name().starts_with("native-options-") {
                captured.lock().unwrap().push(event.clone());
            }
        }),
    )
    .unwrap();
    let schema = DataSchema::builder()
        .name("native-options.schema")
        .version("1")
        .build();
    let parent_uuid = with_scope_stack(create_scope_stack(), || {
        let parent = runtime
            .push_scope("native-options-parent", ScopeType::Custom, None, None, None)
            .unwrap();
        let parent_uuid = get_handle().unwrap().uuid;
        let mut current = runtime
            .scope(
                "native-options-current",
                ScopeType::Custom,
                None,
                None,
                None,
            )
            .unwrap();
        let mut child = runtime
            .scope_with_options(
                "native-options-child",
                ScopeType::Custom,
                None,
                None,
                None,
                ScopeOptions {
                    parent: Some(&parent),
                    attributes: ScopeAttributes::PARALLEL | ScopeAttributes::RELOCATABLE,
                    timestamp: Some(UNIX_EPOCH + Duration::from_micros(100)),
                },
            )
            .unwrap();
        let handle = get_handle().unwrap();
        assert_eq!(
            handle.attributes,
            ScopeAttributes::PARALLEL | ScopeAttributes::RELOCATABLE
        );
        assert_eq!(handle.parent_uuid, Some(parent_uuid));
        assert_eq!(handle.started_at.timestamp_micros(), 100);
        runtime
            .emit_mark_with_context(
                "native-options-mark",
                Some(&serde_json::json!({"allowed":true})),
                None,
                EmitMarkOptions {
                    parent: Some(&parent),
                    timestamp: Some(UNIX_EPOCH + Duration::from_micros(123)),
                    data_schema: Some(schema.clone()),
                    severity: Some(LogSeverity::Warn),
                },
            )
            .unwrap();
        child
            .close_at(None, None, UNIX_EPOCH + Duration::from_micros(200))
            .unwrap();
        current.close(None, None).unwrap();
        runtime.pop_scope(&parent, None, None).unwrap();
        // An explicitly retained identity can parent events after its scope closes.
        runtime
            .emit_mark_with_context(
                "native-options-closed-parent",
                None,
                None,
                EmitMarkOptions {
                    parent: Some(&parent),
                    timestamp: Some(UNIX_EPOCH - Duration::from_nanos(1)),
                    ..Default::default()
                },
            )
            .unwrap();
        parent_uuid
    });
    flush_subscribers().unwrap();
    deregister_subscriber("native-options-events").unwrap();
    owner.shutdown().await.unwrap();
    let events = events.lock().unwrap();
    let mark = events
        .iter()
        .find(|e| e.name() == "native-options-mark")
        .unwrap();
    assert_eq!(mark.parent_uuid(), Some(parent_uuid));
    assert_eq!(mark.timestamp().timestamp_micros(), 123);
    assert_eq!(mark.data_schema(), Some(&schema));
    assert_eq!(
        mark.metadata().unwrap()[nemo_relay_types::api::event::LOG_SEVERITY_METADATA_KEY],
        "warn"
    );
    let closed = events
        .iter()
        .find(|e| e.name() == "native-options-closed-parent")
        .unwrap();
    assert_eq!(closed.parent_uuid(), Some(parent_uuid));
    assert_eq!(closed.timestamp().timestamp_micros(), -1);
}

#[tokio::test]
async fn malformed_native_scope_options_do_not_push_scopes() {
    use nemo_relay_plugin::{abi::Json, abi_value::into_abi};
    let owner = HostRuntime::new("invalid-options".into()).unwrap();
    let raw = owner.runtime_box();
    nemo_relay_runtime::api::runtime::with_scope_stack(
        nemo_relay_runtime::api::runtime::create_scope_stack(),
        || {
            let before = raw.call_sync("scope.snapshot".into(), Json::Null);
            for attributes in [
                serde_json::json!(-1),
                serde_json::json!(4),
                serde_json::json!(1_u64 << 32),
                serde_json::json!("parallel"),
            ] {
                assert!(raw.call_sync("scope.push".into(), into_abi(serde_json::json!({"name":"invalid-options", "scope_type":"custom","attributes":attributes}))).is_err());
            }
            assert!(raw.call_sync("scope.push".into(),into_abi(serde_json::json!({"name":"invalid-options", "scope_type":"custom","parent":{"uuid":"invalid"}}))).is_err());
            assert_eq!(before, raw.call_sync("scope.snapshot".into(), Json::Null));
        },
    );
    owner.shutdown().await.unwrap();
}

#[tokio::test]
async fn sdk_runtime_diagnostics_preserve_empty_and_populated_reports() {
    use nemo_relay_plugin_host::{
        PluginConfig, RuntimeDiagnostic, initialize_exact, record_active_plugin_runtime_diagnostic,
    };
    let mut activation = initialize_exact(PluginConfig::default()).await.unwrap();
    let owner = HostRuntime::new("runtime-diagnostic-report".into()).unwrap();
    let runtime = PluginRuntime::from_abi(owner.runtime_box());
    let empty = runtime.runtime_diagnostics();
    for (code, message, count) in [
        ("native.z", "first failure", 2),
        ("native.a", "another failure", 1),
        ("native.z", "latest failure", 3),
    ] {
        record_active_plugin_runtime_diagnostic(RuntimeDiagnostic {
            code: code.into(),
            component: "runtime-diagnostic-report".into(),
            field: None,
            message: message.into(),
            session_id: None,
            count,
        });
    }
    let populated = runtime.runtime_diagnostics();
    activation.close().unwrap();
    let cleared = runtime.runtime_diagnostics();
    owner.shutdown().await.unwrap();

    assert!(empty.unwrap().entries().is_empty());
    let populated = populated.unwrap();
    assert_eq!(
        populated
            .entries()
            .iter()
            .map(|entry| entry.code.as_str())
            .collect::<Vec<_>>(),
        ["native.a", "native.z"],
    );
    assert_eq!(populated.get("native.a").unwrap().count, 1);
    let updated = populated.get("native.z").unwrap();
    assert_eq!(updated.count, 5);
    assert_eq!(updated.message, "latest failure");
    assert!(populated.get("missing").is_none());
    assert!(cleared.unwrap().entries().is_empty());
}
