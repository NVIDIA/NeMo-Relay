// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SDK constructors bind runtime services without exposing exporter hosts.
use exporter::OpenTelemetryType;
use exporter::otel::{OpenTelemetryFileSinkConfig, OpenTelemetrySubscriber, OtlpFileSinkSettings};
use nemo_relay::{context, exporter, scope, subscriber};

#[test]
fn associated_and_factory_constructors_deliver_scopes_to_file() {
    let directory = tempfile::tempdir().unwrap();
    for associated in [true, false] {
        let filename = if associated {
            "associated.jsonl"
        } else {
            "factory.jsonl"
        };
        let sink = OtlpFileSinkSettings::from_parts(
            directory.path().to_str().unwrap(),
            Some(filename),
            None,
            None,
        )
        .unwrap();
        let config = OpenTelemetryFileSinkConfig::new(OpenTelemetryType::Full, sink);
        let exporter = if associated {
            OpenTelemetrySubscriber::new_file_sink(config).unwrap()
        } else {
            exporter::opentelemetry_file(config).unwrap()
        };
        let name = format!("sdk-exporter-{associated}");
        exporter.register(&name).unwrap();
        context::with_scope_stack(context::create_scope_stack(), || {
            let handle = scope::push_scope(
                scope::PushScopeParams::builder()
                    .name(&name)
                    .scope_type(scope::ScopeType::Agent)
                    .build(),
            )
            .unwrap();
            scope::pop_scope(
                scope::PopScopeParams::builder()
                    .handle_uuid(&handle.uuid)
                    .build(),
            )
            .unwrap();
        });
        subscriber::flush_subscribers().unwrap();
        assert!(exporter.deregister(&name).unwrap());
        exporter.force_flush().unwrap();
        exporter.shutdown().unwrap();
        let output = std::fs::read_to_string(directory.path().join(filename)).unwrap();
        assert!(output.contains(&name), "missing delivered scope: {output}");
    }
}

#[test]
fn trajectory_constructor_and_provider_options_bind_runtime_services() {
    use exporter::atif::{AtifAgentInfo, AtifExporter};
    let atif = AtifExporter::new_for_trajectory(
        "session".into(),
        "trajectory".into(),
        AtifAgentInfo {
            name: "sdk-agent".into(),
            version: "1".into(),
            model_name: None,
            tool_definitions: None,
            extra: None,
        },
    );
    // Export flushes through the bound runtime even when there are no events.
    let trajectory = atif.export().unwrap();
    assert_eq!(trajectory.trajectory_id.as_deref(), Some("trajectory"));
    assert_eq!(trajectory.session_id, "session");
    let subscriber = OpenTelemetrySubscriber::from_tracer_provider_with_type_and_options(
        opentelemetry_sdk::trace::SdkTracerProvider::builder().build(),
        "sdk-contract",
        OpenTelemetryType::Full,
        Default::default(),
    )
    .unwrap();
    subscriber.register("sdk-provider-options").unwrap();
    assert!(subscriber.deregister("sdk-provider-options").unwrap());
    subscriber.force_flush().unwrap();
    subscriber.shutdown_provider().unwrap();
}
