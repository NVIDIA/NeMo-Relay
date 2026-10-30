// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::MarkProjection;
use crate::otel::RelayIdGenerator;
use nemo_relay_types::Json;
use nemo_relay_types::api::event::{
    BaseEvent, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
};
use nemo_relay_types::api::scope::ScopeType;
use opentelemetry_sdk::trace::InMemorySpanExporterBuilder;
use opentelemetry_sdk::trace::SdkTracerProvider;
use serde_json::json;
use std::collections::HashMap;
use uuid::Uuid;

#[test]
fn config_defaults_and_builder_overrides_are_applied() {
    let config = crate::otel::OpenTelemetryConfig::new(
        crate::OpenTelemetryType::OpenInference,
        "http://localhost:4318/v1/traces",
    )
    .with_service_name("demo-agent")
    .with_header("authorization", "Bearer token")
    .with_resource_attribute("deployment.environment", "test")
    .with_service_namespace("agents")
    .with_service_version("1.2.3")
    .with_instrumentation_scope("demo-scope")
    .with_timeout(Duration::from_millis(1250));
    let subscriber = crate::otel::OpenTelemetrySubscriber::new(config).expect("valid config");
    subscriber.force_flush().unwrap();
    subscriber.shutdown().unwrap();
}

#[test]
fn grpc_config_owns_its_tokio_runtime() {
    let subscriber = crate::otel::OpenTelemetrySubscriber::new(
        crate::otel::OpenTelemetryConfig::new(
            crate::OpenTelemetryType::OpenInference,
            "http://127.0.0.1:4317",
        )
        .with_service_name("demo-agent")
        .with_transport(crate::otel::OtlpTransport::Grpc),
    )
    .expect("gRPC construction should not require an ambient Tokio runtime");
    subscriber.shutdown().unwrap();
}

#[test]
fn invalid_grpc_headers_are_rejected() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let err = runtime.block_on(async {
        match crate::otel::OpenTelemetrySubscriber::new(
            crate::otel::OpenTelemetryConfig::new(
                crate::OpenTelemetryType::OpenInference,
                "http://127.0.0.1:4317",
            )
            .with_transport(crate::otel::OtlpTransport::Grpc)
            .with_header("bad key", "value"),
        ) {
            Ok(_) => panic!("invalid metadata key should fail"),
            Err(err) => err,
        }
    });
    assert!(matches!(
        err,
        crate::otel::OpenTelemetryError::InvalidHeader { .. }
    ));
}

#[test]
fn mapped_aliases_are_typed_and_cannot_replace_projected_span_fields() {
    let (provider, exporter) = make_provider();
    let subscriber =
        crate::otel::OpenTelemetrySubscriber::from_tracer_provider_with_type_and_options(
            provider,
            "mapping-scope",
            crate::OpenTelemetryType::OpenInference,
            crate::otel::OpenTelemetrySubscriberOptions {
                mark_projection: MarkProjection::Tool,
                mark_exclude_names: vec!["custom.mark".to_string()],
                attribute_mappings: vec![
                    crate::OtlpAttributeMapping::new("nemo_relay.start.data.tenant", "tenant.id"),
                    crate::OtlpAttributeMapping::new(
                        "nemo_relay.end.data.tenant",
                        "nemo_relay.start.data.tenant",
                    ),
                    crate::OtlpAttributeMapping::new(
                        "nemo_relay.start.data.tenant",
                        "nemo_relay.start.data.existing",
                    ),
                    crate::OtlpAttributeMapping::new("missing.source", "ignored.alias"),
                ],
                promote_metadata_prefixes: Vec::new(),
                completed_span_context_ttl: crate::otel::DEFAULT_COMPLETED_SPAN_CONTEXT_TTL,
            },
        )
        .unwrap();
    let callback = subscriber.subscriber();
    let uuid = Uuid::now_v7();
    callback(&make_start_event(
        uuid,
        None,
        "mapped-scope",
        ScopeType::Agent,
        Some(json!({"tenant": 7, "existing": 9})),
    ));
    callback(&make_end_event(
        uuid,
        None,
        "mapped-scope",
        ScopeType::Agent,
        Some(json!({"tenant": 8})),
    ));
    subscriber.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let span = spans
        .iter()
        .find(|span| span.name.as_ref() == "mapped-scope")
        .unwrap();
    assert_eq!(
        span.attributes
            .iter()
            .filter(|attribute| attribute.key.as_str() == "nemo_relay.start.data.tenant")
            .count(),
        1
    );
    assert_eq!(
        span.attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == "tenant.id")
            .map(|attribute| &attribute.value),
        Some(&opentelemetry::Value::I64(7))
    );
    assert_eq!(
        span.attributes
            .iter()
            .filter(|attribute| attribute.key.as_str() == "nemo_relay.start.data.existing")
            .count(),
        1
    );
    assert_eq!(
        span.attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == "nemo_relay.start.data.existing")
            .map(|attribute| &attribute.value),
        Some(&opentelemetry::Value::I64(9))
    );
    assert!(
        !span
            .attributes
            .iter()
            .any(|attribute| attribute.key.as_str() == "ignored.alias")
    );
}

#[test]
fn session_identity_is_projected_on_trace_roots_and_marks_only() {
    let (provider, exporter) = make_provider();
    let subscriber = crate::otel::OpenTelemetrySubscriber::from_tracer_provider_with_type(
        provider,
        "session-identity",
        crate::OpenTelemetryType::OpenInference,
    );
    let callback = subscriber.subscriber();
    let root_uuid = Uuid::now_v7();
    let child_uuid = Uuid::now_v7();
    let second_root_uuid = Uuid::now_v7();
    let instance_id = root_uuid.to_string();
    let second_instance_id = second_root_uuid.to_string();
    let identity = json!({
        "session_id": "logical-session",
        "user_id": "alice",
        "agent_kind": "claude-code"
    });

    callback(&with_propagation_root(
        make_start_event_with_metadata(root_uuid, None, "identity-root", identity.clone()),
        root_uuid,
    ));
    callback(&with_propagation_root(
        make_start_event_with_metadata(
            child_uuid,
            Some(root_uuid),
            "identity-child",
            identity.clone(),
        ),
        root_uuid,
    ));
    callback(&with_propagation_root(
        make_mark_event_with_metadata(Some(root_uuid), identity.clone()),
        root_uuid,
    ));
    callback(&with_propagation_root(
        make_end_event(
            child_uuid,
            Some(root_uuid),
            "identity-child",
            ScopeType::Agent,
            None,
        ),
        root_uuid,
    ));
    callback(&with_propagation_root(
        make_end_event(root_uuid, None, "identity-root", ScopeType::Agent, None),
        root_uuid,
    ));
    callback(&with_propagation_root(
        make_start_event_with_metadata(
            second_root_uuid,
            None,
            "identity-second-root",
            json!({"session_id": "logical-session", "user_id": 42}),
        ),
        second_root_uuid,
    ));
    callback(&with_propagation_root(
        make_end_event(
            second_root_uuid,
            None,
            "identity-second-root",
            ScopeType::Agent,
            None,
        ),
        second_root_uuid,
    ));
    callback(&with_propagation_root(
        make_mark_event_with_metadata(None, identity),
        root_uuid,
    ));
    subscriber.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_session_root_and_child_identity(&spans, &instance_id, &second_instance_id);
    assert_session_mark_identity(&spans);
}

#[test]
fn mapped_orphan_mark_alias_cannot_replace_intrinsic_mark_fields() {
    let (provider, exporter) = make_provider();
    let subscriber =
        crate::otel::OpenTelemetrySubscriber::from_tracer_provider_with_type_and_options(
            provider,
            "mapping-scope",
            crate::OpenTelemetryType::OpenInference,
            crate::otel::OpenTelemetrySubscriberOptions {
                attribute_mappings: vec![crate::OtlpAttributeMapping::new(
                    "nemo_relay.mark.data.value",
                    "nemo_relay.mark.orphan",
                )],
                ..Default::default()
            },
        )
        .unwrap();
    let callback = subscriber.subscriber();
    callback(&make_mark_event(
        None,
        "mapped-orphan-mark",
        Some(json!({"value": "not-an-orphan-flag"})),
    ));
    subscriber.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let span = spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:mapped-orphan-mark")
        .unwrap();
    assert_eq!(
        span.attributes
            .iter()
            .filter(|attribute| attribute.key.as_str() == "nemo_relay.mark.orphan")
            .count(),
        1
    );
    assert_eq!(
        span.attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == "nemo_relay.mark.orphan")
            .map(|attribute| &attribute.value),
        Some(&opentelemetry::Value::Bool(true))
    );
}

#[test]
fn provider_builders_cover_success_paths() {
    let subscriber = crate::otel::OpenTelemetrySubscriber::new(
        crate::otel::OpenTelemetryConfig::new(
            crate::OpenTelemetryType::OpenInference,
            "http://127.0.0.1:4318/v1/traces",
        )
        .with_service_name("http-success")
        .with_header("authorization", "Bearer token")
        .with_resource_attribute("deployment.environment", "test")
        .with_service_namespace("agents")
        .with_service_version("1.2.3"),
    )
    .unwrap();
    subscriber.force_flush().unwrap();
    subscriber.shutdown().unwrap();
}

#[test]
fn grpc_metadata_and_runtime_builder_paths_succeed() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let subscriber = crate::otel::OpenTelemetrySubscriber::new(
            crate::otel::OpenTelemetryConfig::new(
                crate::OpenTelemetryType::OpenInference,
                "http://127.0.0.1:4317",
            )
            .with_service_name("grpc-demo")
            .with_transport(crate::otel::OtlpTransport::Grpc)
            .with_header("authorization", "Bearer token"),
        )
        .unwrap();
        subscriber.force_flush().ok();
        subscriber.shutdown().ok();
    });
}

fn assert_session_mark_identity(spans: &[opentelemetry_sdk::trace::SpanData]) {
    let root = finished_span_named(spans, "identity-root");
    let orphan_mark = finished_span_named(spans, "mark:session.start");
    let root_attributes = attr_map(&root.attributes);
    let mark_attributes = attr_map(&root.events.events[0].attributes);
    assert_eq!(mark_attributes["session.id"], "logical-session");
    assert_eq!(mark_attributes["user.id"], "alice");
    assert_eq!(mark_attributes["nemo_relay.agent.kind"], "claude-code");
    assert_eq!(
        mark_attributes["nemo_relay.session.instance_id"],
        root_attributes["nemo_relay.session.instance_id"]
    );
    let orphan_attributes = attr_map(&orphan_mark.attributes);
    assert_eq!(orphan_attributes["session.id"], "logical-session");
    assert_eq!(orphan_attributes["user.id"], "alice");
    assert_eq!(orphan_attributes["nemo_relay.agent.kind"], "claude-code");
    assert_eq!(
        orphan_attributes["nemo_relay.session.instance_id"],
        root_attributes["nemo_relay.session.instance_id"]
    );
}

fn assert_session_root_and_child_identity(
    spans: &[opentelemetry_sdk::trace::SpanData],
    instance_id: &str,
    second_instance_id: &str,
) {
    let root = finished_span_named(spans, "identity-root");
    let child = finished_span_named(spans, "identity-child");
    let second_root = finished_span_named(spans, "identity-second-root");

    let root_attributes = attr_map(&root.attributes);
    assert_eq!(root_attributes["session.id"], "logical-session");
    assert_eq!(root_attributes["user.id"], "alice");
    assert_eq!(root_attributes["nemo_relay.agent.kind"], "claude-code");
    assert_eq!(
        root_attributes["nemo_relay.session.instance_id"],
        instance_id
    );
    assert_eq!(
        root.attributes
            .iter()
            .filter(|attribute| attribute.key.as_str() == "session.id")
            .count(),
        1
    );
    let child_attributes = attr_map(&child.attributes);
    assert!(!child_attributes.contains_key("session.id"));
    assert!(!child_attributes.contains_key("user.id"));
    assert!(!child_attributes.contains_key("nemo_relay.session.instance_id"));
    assert!(!child_attributes.contains_key("nemo_relay.agent.kind"));
    assert_eq!(child_attributes["openinference.metadata.user_id"], "alice");
    let second_root_attributes = attr_map(&second_root.attributes);
    assert_eq!(second_root_attributes["session.id"], "logical-session");
    assert!(!second_root_attributes.contains_key("user.id"));
    assert_eq!(
        second_root_attributes["openinference.metadata.user_id"],
        "42"
    );
    assert_eq!(
        second_root_attributes["nemo_relay.session.instance_id"],
        second_instance_id
    );
    assert_ne!(
        root.span_context.trace_id(),
        second_root.span_context.trace_id()
    );
}

fn attr_map(attributes: &[KeyValue]) -> HashMap<String, String> {
    attributes
        .iter()
        .map(|attribute| {
            (
                attribute.key.as_str().to_string(),
                attribute.value.to_string(),
            )
        })
        .collect()
}

fn finished_span_named<'a>(
    spans: &'a [opentelemetry_sdk::trace::SpanData],
    name: &str,
) -> &'a opentelemetry_sdk::trace::SpanData {
    spans
        .iter()
        .find(|span| span.name.as_ref() == name)
        .unwrap_or_else(|| panic!("missing span {name}"))
}

fn make_end_event(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    output: Option<Json>,
) -> Event {
    make_scope_event(
        ScopeCategory::End,
        uuid,
        parent_uuid,
        name,
        scope_type,
        output,
    )
}

fn make_mark_event(parent_uuid: Option<Uuid>, name: &str, data: Option<Json>) -> Event {
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .name(name)
            .data_opt(data)
            .build(),
        None,
        None,
    ))
}

fn make_mark_event_with_metadata(parent_uuid: Option<Uuid>, metadata: Json) -> Event {
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .name("session.start")
            .metadata(metadata)
            .build(),
        None,
        None,
    ))
}

fn make_provider() -> (
    SdkTracerProvider,
    opentelemetry_sdk::trace::InMemorySpanExporter,
) {
    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = SdkTracerProvider::builder()
        .with_id_generator(RelayIdGenerator)
        .with_simple_exporter(exporter.clone())
        .build();
    (provider, exporter)
}

fn make_scope_event(
    scope_category: ScopeCategory,
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    data: Option<Json>,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(uuid)
            .name(name)
            .data_opt(data)
            .build(),
        scope_category,
        Vec::new(),
        EventCategory::from(scope_type),
        None,
    ))
}

fn make_start_event(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    input: Option<Json>,
) -> Event {
    make_scope_event(
        ScopeCategory::Start,
        uuid,
        parent_uuid,
        name,
        scope_type,
        input,
    )
}

fn make_start_event_with_metadata(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    metadata: Json,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(uuid)
            .name(name)
            .metadata(metadata)
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::agent(),
        None,
    ))
}

fn with_propagation_root(mut event: Event, root_uuid: Uuid) -> Event {
    event.set_propagation_root_uuid(Some(root_uuid));
    event
}
