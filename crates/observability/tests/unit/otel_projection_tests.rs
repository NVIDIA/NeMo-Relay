// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_relay_model_pricing::pricing_test_mutex;
use nemo_relay_model_pricing::{
    PricingCatalog, PricingResolver, reset_active_pricing_resolver, set_active_pricing_resolver,
};
use nemo_relay_types::Json;
use nemo_relay_types::api::event::*;
use nemo_relay_types::api::llm::*;
use nemo_relay_types::api::scope::*;
use nemo_relay_types::api::tool::*;
use nemo_relay_types::codec::request::*;
use nemo_relay_types::codec::response::*;
use opentelemetry::trace::TraceContextExt;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use opentelemetry_proto::tonic::common::v1::{KeyValue as OtlpKeyValue, any_value};
use opentelemetry_sdk::trace::{BatchConfigBuilder, InMemorySpanExporterBuilder};
use prost::Message;
use serde_json::json;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use uuid::Uuid;

#[test]
fn local_unexported_parent_starts_a_root_trace() {
    let parent_uuid = Uuid::now_v7();
    let agent_uuid = Uuid::now_v7();
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider, "test".into());
    let mut start = make_start_event(
        agent_uuid,
        Some(parent_uuid),
        "receiver-agent",
        ScopeType::Agent,
        None,
    );
    start.set_propagation_root_uuid(Some(agent_uuid));

    processor.process(&crate::test_support::event_view(&start));
    processor.process(&crate::test_support::event_view(&make_end_event(
        agent_uuid,
        Some(parent_uuid),
        "receiver-agent",
        ScopeType::Agent,
        None,
    )));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let span = &spans[0];
    assert_eq!(span.span_context.trace_id(), relay_trace_id(agent_uuid));
    assert_eq!(span.parent_span_id, SpanId::INVALID);
    assert!(!span.parent_span_is_remote);
}

#[test]
fn orphan_mark_with_a_local_unexported_parent_starts_a_new_trace() {
    let parent_uuid = Uuid::now_v7();
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider, "test".into());
    let mut mark = make_mark_event(Some(parent_uuid), "detached", None);
    let mark_uuid = mark.uuid();
    mark.set_propagation_root_uuid(Some(Uuid::now_v7()));

    processor.process(&crate::test_support::event_view(&mark));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let span = &spans[0];
    assert_eq!(span.span_context.trace_id(), relay_trace_id(mark_uuid));
    assert_eq!(span.parent_span_id, SpanId::INVALID);
    assert!(!span.parent_span_is_remote);
}

#[test]
fn promotes_final_scope_and_mark_metadata_without_duplicate_span_keys() {
    for otel_type in [OpenTelemetryType::Full, OpenTelemetryType::OpenInference] {
        let (provider, exporter) = make_provider();
        let mut processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
                provider,
                "metadata-promotion-test".into(),
                otel_type,
                MarkProjection::default(),
                default_mark_exclude_names(),
                Vec::new(),
                vec!["nv.".to_string(), "nemo_relay.".to_string()],
                SignalRuntimeDiagnostics::new(None).with_host(crate::test_support::detached_host()),
            );
        let uuid = Uuid::now_v7();
        processor.process(&crate::test_support::event_view(
            &make_start_event_with_metadata(
                uuid,
                None,
                "metadata-promotion-scope",
                json!({
                    "nv.source": "start",
                    "nemo_relay.scope_type": "attempted-overwrite"
                }),
            ),
        ));
        processor.process(&crate::test_support::event_view(
            &make_mark_event_with_metadata(Some(uuid), json!({"nv.source": "mark"})),
        ));
        processor.process(&crate::test_support::event_view(
            &make_end_event_with_metadata(
                uuid,
                None,
                "metadata-promotion-scope",
                ScopeType::Agent,
                json!({
                    "nv.source": "end",
                    "nv.completed": true,
                    "nemo_relay.scope_type": "attempted-overwrite"
                }),
            ),
        ));
        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(
            span.attributes
                .iter()
                .filter(|attribute| attribute.key.as_str() == "nv.source")
                .count(),
            1
        );
        let attributes = attr_map(&span.attributes);
        assert_eq!(attributes.get("nv.source"), Some(&"end".to_string()));
        assert_eq!(attributes.get("nv.completed"), Some(&"true".to_string()));
        assert_eq!(
            attributes.get("nemo_relay.scope_type"),
            Some(&"agent".to_string())
        );
        assert!(
            !attributes
                .keys()
                .any(|key| key.ends_with(".metadata.nv.source"))
        );
        let mark_attributes = attr_map(&span.events.events[0].attributes);
        assert_eq!(mark_attributes.get("nv.source"), Some(&"mark".to_string()));
        assert!(
            !mark_attributes
                .keys()
                .any(|key| key.ends_with(".metadata.nv.source"))
        );
    }
}

#[test]
fn removes_resource_promoted_metadata_from_all_otlp_event_projections() {
    for otel_type in [OpenTelemetryType::Full, OpenTelemetryType::OpenInference] {
        let (provider, exporter) = make_provider();
        let mut processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
                provider,
                "resource-metadata-filter-test".into(),
                otel_type,
                MarkProjection::default(),
                default_mark_exclude_names(),
                Vec::new(),
                Vec::new(),
                SignalRuntimeDiagnostics::new(None).with_host(crate::test_support::detached_host()),
        );
        processor.resource_metadata_prefixes = vec!["tenant.".to_string()];
        processor.resource_metadata_protected_keys = HashSet::from(["tenant.region".to_string()]);
        let uuid = Uuid::now_v7();
        processor.process(&crate::test_support::event_view(
            &make_start_event_with_metadata(
                uuid,
                None,
                "resource-metadata-filter-scope",
                json!({"tenant.id": "root", "tenant.region": "metadata", "keep": "start"}),
            ),
        ));
        processor.process(&crate::test_support::event_view(
            &make_mark_event_with_metadata(
                Some(uuid),
                json!({"tenant.id": "mark", "tenant.region": "metadata", "keep": "mark"}),
            ),
        ));
        processor.process(&crate::test_support::event_view(
            &make_end_event_with_metadata(
                uuid,
                None,
                "resource-metadata-filter-scope",
                ScopeType::Agent,
                json!({"tenant.id": "end", "tenant.region": "metadata", "keep": "end"}),
            ),
        ));
        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        let span = finished_span_named(&spans, "resource-metadata-filter-scope");
        let serialized_attributes = span
            .attributes
            .iter()
            .chain(
                span.events
                    .events
                    .iter()
                    .flat_map(|event| event.attributes.iter()),
            )
            .map(|attribute| format!("{}={}", attribute.key, attribute.value))
            .collect::<Vec<_>>();
        assert!(
            serialized_attributes
                .iter()
                .any(|attribute| attribute.contains("keep"))
        );
        assert!(
            serialized_attributes
                .iter()
                .any(|attribute| attribute.contains("tenant.region"))
        );
        assert!(serialized_attributes.iter().all(
            |attribute| !attribute.contains("tenant.id") && !attribute.contains("tenant\":\"")
        ));
    }
}

#[test]
fn promotes_orphan_and_tool_projection_mark_metadata() {
    for otel_type in [OpenTelemetryType::Full, OpenTelemetryType::OpenInference] {
        let (provider, exporter) = make_provider();
        let mut processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
                provider,
                "metadata-promotion-mark-test".into(),
                otel_type,
                MarkProjection::Tool,
                default_mark_exclude_names(),
                Vec::new(),
                vec!["nv.".to_string()],
                SignalRuntimeDiagnostics::new(None).with_host(crate::test_support::detached_host()),
            );
        let parent_uuid = Uuid::now_v7();
        processor.process(&crate::test_support::event_view(&Event::Mark(
            MarkEvent::new(
                BaseEvent::builder()
                    .name("metadata.orphan")
                    .metadata(json!({"nv.source": "orphan"}))
                    .build(),
                None,
                None,
            ),
        )));
        processor.process(&crate::test_support::event_view(&make_start_event(
            parent_uuid,
            None,
            "metadata-promotion-parent",
            ScopeType::Agent,
            None,
        )));
        processor.process(&crate::test_support::event_view(&Event::Mark(
            MarkEvent::new(
                BaseEvent::builder()
                    .parent_uuid(parent_uuid)
                    .name("metadata.projected")
                    .metadata(json!({"nv.source": "projected"}))
                    .build(),
                None,
                None,
            ),
        )));
        processor.process(&crate::test_support::event_view(&make_end_event(
            parent_uuid,
            None,
            "metadata-promotion-parent",
            ScopeType::Agent,
            None,
        )));
        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 3);
        let parent = finished_span_named(&spans, "metadata-promotion-parent");
        let orphan = finished_span_named(&spans, "mark:metadata.orphan");
        let projected = finished_span_named(&spans, "mark:metadata.projected");
        assert!(!attr_map(&parent.attributes).contains_key("nv.source"));
        assert_eq!(
            attr_map(&orphan.attributes).get("nv.source"),
            Some(&"orphan".to_string())
        );
        assert_eq!(
            attr_map(&projected.attributes).get("nv.source"),
            Some(&"projected".to_string())
        );
        assert_eq!(projected.parent_span_id, parent.span_context.span_id());
    }
}

#[test]
fn promotes_final_scope_metadata_across_trace_projections() {
    for otel_type in [
        OpenTelemetryType::Full,
        OpenTelemetryType::GenAi,
        OpenTelemetryType::OpenInference,
    ] {
        let (provider, exporter) = make_provider();
        let mut processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
                provider,
                "metadata-promotion-projection-test".into(),
                otel_type,
                MarkProjection::default(),
                default_mark_exclude_names(),
                Vec::new(),
                vec!["nv.".to_string()],
                SignalRuntimeDiagnostics::new(None).with_host(crate::test_support::detached_host()),
            );
        let uuid = Uuid::now_v7();
        processor.process(&crate::test_support::event_view(
            &make_start_event_with_metadata(
                uuid,
                None,
                "metadata-promotion-projection-scope",
                json!({"nv.source": "start"}),
            ),
        ));
        processor.process(&crate::test_support::event_view(
            &make_end_event_with_metadata(
                uuid,
                None,
                "metadata-promotion-projection-scope",
                ScopeType::Agent,
                json!({"nv.source": "end"}),
            ),
        ));
        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(
            attr_map(&spans[0].attributes).get("nv.source"),
            Some(&"end".to_string())
        );
    }
}

#[test]
fn omits_scope_metadata_when_final_value_is_unsupported_across_trace_projections() {
    for otel_type in [
        OpenTelemetryType::Full,
        OpenTelemetryType::GenAi,
        OpenTelemetryType::OpenInference,
    ] {
        let (provider, exporter) = make_provider();
        let runtime_diagnostics =
            SignalRuntimeDiagnostics::new(None).with_host(crate::test_support::detached_host());
        let mut processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
                provider,
                "unsupported-final-metadata-promotion-test".into(),
                otel_type,
                MarkProjection::default(),
                default_mark_exclude_names(),
                Vec::new(),
                vec!["nv.".to_string()],
                runtime_diagnostics.clone(),
            );
        let uuid = Uuid::now_v7();
        processor.process(&crate::test_support::event_view(
            &make_start_event_with_metadata(
                uuid,
                None,
                "unsupported-final-metadata-promotion-scope",
                json!({"nv.source": "start"}),
            ),
        ));
        processor.process(&crate::test_support::event_view(
            &make_end_event_with_metadata(
                uuid,
                None,
                "unsupported-final-metadata-promotion-scope",
                ScopeType::Agent,
                json!({"nv.source": {"unsupported": true}}),
            ),
        ));
        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert!(!attr_map(&spans[0].attributes).contains_key("nv.source"));

        let diagnostics = runtime_diagnostics.snapshot();
        let diagnostic = diagnostics
            .get("otel.metadata_promotion_value_unsupported.nv.source")
            .expect("unsupported final metadata diagnostic");
        assert_eq!(diagnostic.count, 1);
        assert!(diagnostic.message.contains("nv.source"));
    }
}

#[test]
fn reports_each_unsupported_metadata_key_with_a_deterministic_diagnostic_code() {
    let (provider, exporter) = make_provider();
    let runtime_diagnostics =
        SignalRuntimeDiagnostics::new(None).with_host(crate::test_support::detached_host());
    let mut processor =
        OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
            provider,
            "unsupported-metadata-promotion-keys-test".into(),
            OpenTelemetryType::Full,
            MarkProjection::default(),
            default_mark_exclude_names(),
            Vec::new(),
            vec!["tenant.".to_string()],
            runtime_diagnostics.clone(),
        );
    let uuid = Uuid::now_v7();
    let unsupported_metadata = json!({
        "tenant.plan": {"name": "enterprise"},
        "tenant.flags": null,
        "tenant.tags": [["nested"]],
        "tenant.mixed": [1, "string"],
    });

    processor.process(&crate::test_support::event_view(
        &make_start_event_with_metadata(
            uuid,
            None,
            "unsupported-metadata-promotion-keys-scope",
            unsupported_metadata.clone(),
        ),
    ));
    processor.process(&crate::test_support::event_view(
        &make_end_event_with_metadata(
            uuid,
            None,
            "unsupported-metadata-promotion-keys-scope",
            ScopeType::Agent,
            unsupported_metadata,
        ),
    ));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let span_attributes = attr_map(&spans[0].attributes);
    for key in ["tenant.flags", "tenant.mixed", "tenant.plan", "tenant.tags"] {
        assert!(!span_attributes.contains_key(key));
    }

    let diagnostics = runtime_diagnostics.snapshot();
    let expected_keys = ["tenant.flags", "tenant.mixed", "tenant.plan", "tenant.tags"];
    assert_eq!(
        diagnostics
            .entries()
            .iter()
            .map(|diagnostic| diagnostic.code.as_str())
            .collect::<Vec<_>>(),
        expected_keys
            .iter()
            .map(|key| format!("otel.metadata_promotion_value_unsupported.{key}"))
            .collect::<Vec<_>>()
    );
    for key in expected_keys {
        let code = format!("otel.metadata_promotion_value_unsupported.{key}");
        let diagnostic = diagnostics
            .get(&code)
            .expect("metadata-key-specific promotion diagnostic");
        assert_eq!(diagnostic.count, 2);
        assert!(diagnostic.message.contains(key));
    }
}

#[test]
fn promotes_start_only_scope_metadata_across_trace_projections() {
    for otel_type in [
        OpenTelemetryType::Full,
        OpenTelemetryType::GenAi,
        OpenTelemetryType::OpenInference,
    ] {
        let (provider, exporter) = make_provider();
        let mut processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings_and_runtime_diagnostics(
                provider,
                "start-metadata-promotion-projection-test".into(),
                otel_type,
                MarkProjection::default(),
                default_mark_exclude_names(),
                Vec::new(),
                vec!["nv.".to_string()],
                SignalRuntimeDiagnostics::new(None).with_host(crate::test_support::detached_host()),
            );
        let uuid = Uuid::now_v7();
        processor.process(&crate::test_support::event_view(
            &make_start_event_with_metadata(
                uuid,
                None,
                "start-metadata-promotion-projection-scope",
                json!({"nv.start_only": "configured"}),
            ),
        ));
        processor.process(&crate::test_support::event_view(
            &make_end_event_with_metadata(
                uuid,
                None,
                "start-metadata-promotion-projection-scope",
                ScopeType::Agent,
                json!({}),
            ),
        ));
        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(
            attr_map(&spans[0].attributes).get("nv.start_only"),
            Some(&"configured".to_string())
        );
    }
}

#[test]
fn gen_ai_projection_is_fixed_and_preserves_all_scope_parentage() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings(
        provider,
        "gen-ai-test".to_string(),
        OpenTelemetryType::GenAi,
        MarkProjection::Tool,
        Vec::new(),
        Vec::new(),
    );
    assert!(
        processor
            .mark_attributes(&crate::test_support::event_view(&make_mark_event(
                None,
                "omitted-mark",
                None
            )))
            .is_empty()
    );
    let agent_uuid = Uuid::now_v7();
    let reranker_uuid = Uuid::now_v7();
    let tool_uuid = Uuid::now_v7();

    processor.process(&crate::test_support::event_view(&make_start_event(
        agent_uuid,
        None,
        "research-agent",
        ScopeType::Agent,
        Some(json!({"secret": "must-not-export"})),
    )));
    processor.process(&crate::test_support::event_view(&make_start_event(
        reranker_uuid,
        Some(agent_uuid),
        "rerank",
        ScopeType::Reranker,
        None,
    )));
    processor.process(&crate::test_support::event_view(&make_mark_event(
        Some(agent_uuid),
        "checkpoint",
        Some(json!({"secret": "must-not-export"})),
    )));
    processor.process(&crate::test_support::event_view(&make_start_event(
        tool_uuid,
        Some(reranker_uuid),
        "web-search",
        ScopeType::Tool,
        Some(json!({"query": "exported-tool-input"})),
    )));
    processor.process(&crate::test_support::event_view(&make_end_event(
        tool_uuid,
        Some(reranker_uuid),
        "web-search",
        ScopeType::Tool,
        Some(json!({"result": "exported-tool-result"})),
    )));
    processor.process(&crate::test_support::event_view(&make_end_event(
        reranker_uuid,
        Some(agent_uuid),
        "rerank",
        ScopeType::Reranker,
        None,
    )));
    processor.process(&crate::test_support::event_view(&make_end_event(
        agent_uuid,
        None,
        "research-agent",
        ScopeType::Agent,
        None,
    )));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 3);
    let agent = spans
        .iter()
        .find(|span| span.name.as_ref() == "invoke_agent research-agent")
        .unwrap();
    let tool = spans
        .iter()
        .find(|span| span.name.as_ref() == "execute_tool web-search")
        .unwrap();
    let reranker = spans
        .iter()
        .find(|span| span.name.as_ref() == "rerank")
        .unwrap();
    assert_eq!(agent.span_kind, SpanKind::Internal);
    assert_eq!(reranker.span_kind, SpanKind::Internal);
    assert_eq!(tool.span_kind, SpanKind::Internal);
    assert_eq!(reranker.parent_span_id, agent.span_context.span_id());
    assert_eq!(tool.parent_span_id, reranker.span_context.span_id());
    assert!(agent.events.events.is_empty());
    assert!(spans.iter().all(|span| {
        span.attributes.iter().all(|attribute| {
            !attribute.key.as_str().starts_with("nemo_relay.")
                && !attribute.value.to_string().contains("must-not-export")
        })
    }));
    let agent_attributes = attr_map(&agent.attributes);
    assert!(reranker.attributes.is_empty());
    let tool_attributes = attr_map(&tool.attributes);
    assert_eq!(
        agent_attributes.get("gen_ai.operation.name"),
        Some(&"invoke_agent".to_string())
    );
    assert_eq!(
        tool_attributes.get("gen_ai.operation.name"),
        Some(&"execute_tool".to_string())
    );
}

#[test]
fn gen_ai_projection_uses_standard_operation_names_and_span_kinds() {
    for (scope_type, name, expected_name, expected_kind) in [
        (
            ScopeType::Agent,
            "planner",
            "invoke_agent planner",
            SpanKind::Internal,
        ),
        (ScopeType::Llm, "chat", "chat", SpanKind::Client),
        (
            ScopeType::Llm,
            "generate_content",
            "generate_content",
            SpanKind::Client,
        ),
        (
            ScopeType::Llm,
            "text_completion",
            "text_completion",
            SpanKind::Client,
        ),
        (
            ScopeType::Tool,
            "search",
            "execute_tool search",
            SpanKind::Internal,
        ),
        (ScopeType::Embedder, "embed", "embeddings", SpanKind::Client),
        (
            ScopeType::Retriever,
            "retrieve",
            "retrieval",
            SpanKind::Client,
        ),
    ] {
        let event = make_start_event(Uuid::now_v7(), None, name, scope_type, None);
        assert_eq!(
            crate::test_support::otel_genai::span_name(&event),
            expected_name
        );
        assert_eq!(
            crate::test_support::otel_genai::span_kind(&event),
            expected_kind
        );
    }

    for generic in [
        ScopeType::Function,
        ScopeType::Reranker,
        ScopeType::Guardrail,
        ScopeType::Evaluator,
        ScopeType::Custom,
        ScopeType::Unknown,
    ] {
        let event = make_start_event(Uuid::now_v7(), None, "generic", generic, None);
        assert_eq!(
            crate::test_support::otel_genai::span_name(&event),
            "generic"
        );
        assert_eq!(
            crate::test_support::otel_genai::span_kind(&event),
            SpanKind::Internal
        );
        assert!(crate::test_support::otel_genai::start_attributes(&event).is_empty());
    }
}

#[test]
fn gen_ai_projection_classifies_marked_turn_roots_as_internal_agent_invocations() {
    let mut turn = make_start_event(
        Uuid::now_v7(),
        None,
        "claude-code-turn",
        ScopeType::Custom,
        None,
    );
    if let Event::Scope(scope) = &mut turn {
        scope.base.metadata = Some(json!({
            "nemo_relay_scope_role": "turn",
            "conversation_id": "conversation-1"
        }));
    }

    assert_eq!(
        crate::test_support::otel_genai::span_name(&turn),
        "invoke_agent"
    );
    assert_eq!(
        crate::test_support::otel_genai::span_kind(&turn),
        SpanKind::Internal
    );
    let attributes = attr_map(&crate::test_support::otel_genai::start_attributes(&turn));
    assert_eq!(attributes["gen_ai.operation.name"], "invoke_agent");
    assert_eq!(attributes["gen_ai.conversation.id"], "conversation-1");
    assert!(!attributes.contains_key("gen_ai.agent.name"));

    let mut payload_marked_turn = make_start_event(
        Uuid::now_v7(),
        None,
        "claude-code-turn",
        ScopeType::Custom,
        None,
    );
    if let Event::Scope(scope) = &mut payload_marked_turn {
        scope.base.data = Some(json!({"nemo_relay_scope_role": "turn"}));
    }
    assert_eq!(
        crate::test_support::otel_genai::span_name(&payload_marked_turn),
        "claude-code-turn"
    );
    assert!(crate::test_support::otel_genai::start_attributes(&payload_marked_turn).is_empty());

    let mut named_turn = make_start_event(
        Uuid::now_v7(),
        None,
        "claude-code-turn",
        ScopeType::Custom,
        None,
    );
    if let Event::Scope(scope) = &mut named_turn {
        scope.base.metadata = Some(json!({
            "nemo_relay_scope_role": "turn",
            "gen_ai.agent.name": "Claude Code",
            "gen_ai.agent.description": "Coding agent"
        }));
    }
    assert_eq!(
        crate::test_support::otel_genai::span_name(&named_turn),
        "invoke_agent Claude Code"
    );
    let attributes = attr_map(&crate::test_support::otel_genai::start_attributes(
        &named_turn,
    ));
    assert_eq!(attributes["gen_ai.agent.name"], "Claude Code");
    assert_eq!(attributes["gen_ai.agent.description"], "Coding agent");
}

#[test]
fn gen_ai_projection_emits_source_backed_registry_attributes() {
    for (metadata, expected) in [
        (json!({"model": "ambiguous-model"}), None),
        (
            json!({"gen_ai.request.model": "single-configured-model"}),
            Some("single-configured-model"),
        ),
    ] {
        let mut agent = make_start_event(Uuid::now_v7(), None, "planner", ScopeType::Agent, None);
        if let Event::Scope(scope) = &mut agent {
            scope.base.metadata = Some(metadata);
        }
        let attributes = attr_map(&crate::test_support::otel_genai::start_attributes(&agent));
        assert_eq!(
            attributes.get("gen_ai.request.model").map(String::as_str),
            expected
        );
    }

    let mut embedder = make_start_event(Uuid::now_v7(), None, "embed", ScopeType::Embedder, None);
    if let Event::Scope(scope) = &mut embedder {
        scope.base.metadata = Some(json!({
            "dimensions": 1024,
            "encoding_format": "base64"
        }));
    }
    let attributes = attr_map(&crate::test_support::otel_genai::start_attributes(
        &embedder,
    ));
    assert_eq!(attributes["gen_ai.embeddings.dimension.count"], "1024");
    assert_eq!(
        attributes["gen_ai.request.encoding_formats"],
        "[\"base64\"]"
    );

    let chat_request: AnnotatedLlmRequest = serde_json::from_value(json!({
        "model": "gpt-5",
        "stream": false,
        "api_specific": {
            "api": "openai_chat",
            "modalities": ["audio"],
            "reasoning_effort": "high"
        }
    }))
    .unwrap();
    let chat = make_scope_event_with_profile(
        ScopeCategory::Start,
        Uuid::now_v7(),
        None,
        "openai.chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_request(Arc::new(chat_request))
                .build(),
        ),
    );
    let attributes = attr_map(&crate::test_support::otel_genai::start_attributes(&chat));
    assert_eq!(attributes["gen_ai.output.type"], "speech");
    assert_eq!(attributes["gen_ai.request.reasoning.level"], "high");
    assert!(!attributes.contains_key("gen_ai.request.stream"));

    let responses_request: AnnotatedLlmRequest = serde_json::from_value(json!({
        "previous_response_id": "resp-previous",
        "reasoning": {"effort": "medium"},
        "api_specific": {"api": "openai_responses"}
    }))
    .unwrap();
    let responses = make_scope_event_with_profile(
        ScopeCategory::Start,
        Uuid::now_v7(),
        None,
        "openai.responses",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_request(Arc::new(responses_request))
                .build(),
        ),
    );
    let attributes = attr_map(&crate::test_support::otel_genai::start_attributes(
        &responses,
    ));
    assert_eq!(
        attributes["gen_ai.request.previous_response.id"],
        "resp-previous"
    );
    assert_eq!(attributes["gen_ai.request.reasoning.level"], "medium");
}

#[test]
fn gen_ai_output_type_requires_an_explicit_source_backed_request() {
    for (api_specific, expected) in [
        (
            json!({"api": "openai_chat", "modalities": ["audio"]}),
            Some("speech"),
        ),
        (
            json!({"api": "openai_chat", "modalities": ["text"]}),
            Some("text"),
        ),
        (
            json!({"api": "openai_chat", "response_format": {"type": "json_object"}}),
            Some("json"),
        ),
        (
            json!({"api": "openai_responses", "text": {"format": {"type": "json_schema"}}}),
            Some("json"),
        ),
        (json!({"api": "openai_chat", "modalities": ["image"]}), None),
        (
            json!({"api": "openai_chat", "modalities": ["text", "audio"]}),
            None,
        ),
        (
            json!({"api": "openai_responses", "text": {"verbosity": "low"}}),
            None,
        ),
    ] {
        let request: AnnotatedLlmRequest = serde_json::from_value(json!({
            "model": "test-model",
            "api_specific": api_specific,
        }))
        .unwrap();
        let event = make_scope_event_with_profile(
            ScopeCategory::Start,
            Uuid::now_v7(),
            None,
            "test.request",
            ScopeType::Llm,
            None,
            Some(
                CategoryProfile::builder()
                    .annotated_request(Arc::new(request))
                    .build(),
            ),
        );
        let attributes = attr_map(&crate::test_support::otel_genai::start_attributes(&event));
        assert_eq!(
            attributes.get("gen_ai.output.type").map(String::as_str),
            expected
        );
    }
}

#[test]
fn gen_ai_projection_emits_provider_reasoning_token_usage() {
    for (api_specific, expected) in [
        (
            json!({
                "api": "openai_responses",
                "output_tokens_details": {"reasoning_tokens": 17}
            }),
            "17",
        ),
        (
            json!({
                "api": "gemini_generate_content",
                "thoughts_tokens": 23
            }),
            "23",
        ),
    ] {
        let response: AnnotatedLlmResponse = serde_json::from_value(json!({
            "api_specific": api_specific
        }))
        .unwrap();
        let event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
            None,
            "chat",
            ScopeType::Llm,
            None,
            Some(
                CategoryProfile::builder()
                    .annotated_response(Arc::new(response))
                    .build(),
            ),
        );
        let attributes = attr_map(&crate::test_support::otel_genai::end_attributes(&event));
        assert_eq!(attributes["gen_ai.usage.reasoning.output_tokens"], expected);
    }
}

#[test]
fn gen_ai_projection_emits_only_span_specific_attributes() {
    let common = json!({
        "provider": "openai",
        "conversation_id": "conversation-1",
        "server_address": "api.example.test",
        "server_port": 443,
        "model": "model-1",
        "agent_description": "helpful",
        "tool_type": "function",
        "tool_call_id": "call-1",
        "tool_description": "searches",
        "data_source_id": "docs",
        "top_k": 5
    });
    let cases = [
        (
            ScopeType::Agent,
            "agent",
            [
                "gen_ai.agent.description",
                "gen_ai.agent.name",
                "gen_ai.conversation.id",
                "gen_ai.operation.name",
            ]
            .as_slice(),
        ),
        (
            ScopeType::Llm,
            "chat",
            [
                "gen_ai.conversation.id",
                "gen_ai.operation.name",
                "gen_ai.provider.name",
                "gen_ai.request.model",
                "server.address",
                "server.port",
            ]
            .as_slice(),
        ),
        (
            ScopeType::Tool,
            "search",
            [
                "gen_ai.conversation.id",
                "gen_ai.operation.name",
                "gen_ai.tool.call.arguments",
                "gen_ai.tool.call.id",
                "gen_ai.tool.description",
                "gen_ai.tool.name",
                "gen_ai.tool.type",
            ]
            .as_slice(),
        ),
        (
            ScopeType::Embedder,
            "embed",
            [
                "gen_ai.operation.name",
                "gen_ai.provider.name",
                "gen_ai.request.model",
                "server.address",
                "server.port",
            ]
            .as_slice(),
        ),
        (
            ScopeType::Retriever,
            "retrieve",
            [
                "gen_ai.data_source.id",
                "gen_ai.operation.name",
                "gen_ai.provider.name",
                "gen_ai.request.model",
                "gen_ai.retrieval.top_k",
                "server.address",
                "server.port",
            ]
            .as_slice(),
        ),
    ];
    for (scope_type, name, expected) in cases {
        let mut event =
            make_start_event(Uuid::now_v7(), None, name, scope_type, Some(common.clone()));
        if let Event::Scope(scope) = &mut event {
            scope.base.metadata = Some(common.clone());
        }
        let attributes = crate::test_support::otel_genai::start_attributes(&event);
        let actual = attributes
            .iter()
            .map(|attribute| attribute.key.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(actual, expected.iter().copied().collect());
        assert!(attributes.iter().all(|attribute| {
            attribute.key.as_str() != "gen_ai.conversation.id"
                || matches!(
                    scope_type,
                    ScopeType::Agent | ScopeType::Llm | ScopeType::Tool
                )
        }));
        if scope_type == ScopeType::Retriever {
            let top_k = attributes
                .iter()
                .find(|attribute| attribute.key.as_str() == "gen_ai.retrieval.top_k")
                .unwrap();
            assert_eq!(top_k.value, opentelemetry::Value::I64(5));
        }
    }

    let embed_end = make_end_event(
        Uuid::now_v7(),
        None,
        "embed",
        ScopeType::Embedder,
        Some(json!({"input_tokens": 7, "output_tokens": 11})),
    );
    let embed_attributes = attr_map(&crate::test_support::otel_genai::end_attributes(&embed_end));
    assert_eq!(
        embed_attributes.get("gen_ai.usage.input_tokens"),
        Some(&"7".to_string())
    );
    assert!(!embed_attributes.contains_key("gen_ai.usage.output_tokens"));
}

#[test]
fn gen_ai_tool_content_is_captured_by_default_and_preserves_json() {
    use crate::test_support::otel_genai::{end_attributes, start_attributes};
    for (payload, expected) in [
        (
            json!({"query": "sanitized"}),
            Some(json!({"query": "sanitized"})),
        ),
        (
            json!("{\"query\":\"sanitized\"}"),
            Some(json!({"query": "sanitized"})),
        ),
        (json!("\"serialized text\""), Some(json!("serialized text"))),
        (json!({}), Some(json!({}))),
        (json!("plain text"), Some(json!("plain text"))),
        (json!("[1,2]"), Some(json!([1, 2]))),
        (json!([1, 2]), Some(json!([1, 2]))),
        (json!(null), None),
        (json!(false), Some(json!(false))),
        (json!(42), Some(json!(42))),
    ] {
        let start = make_start_event(
            Uuid::now_v7(),
            None,
            "search",
            ScopeType::Tool,
            Some(payload.clone()),
        );
        let mut end = make_end_event(start.uuid(), None, "search", ScopeType::Tool, Some(payload));
        for (attrs, key) in [
            (start_attributes(&start), "gen_ai.tool.call.arguments"),
            (end_attributes(&end), "gen_ai.tool.call.result"),
        ] {
            let attrs = attr_map(&attrs);
            let actual = attrs
                .get(key)
                .map(|value| serde_json::from_str::<Json>(value).unwrap());
            assert_eq!(actual, expected.clone());
        }
        if let Event::Scope(scope) = &mut end {
            scope.base.metadata = Some(json!({"otel.status_code": "ERROR"}));
        }
        let attrs = attr_map(&end_attributes(&end));
        assert!(attrs.contains_key("error.type"));
        assert!(!attrs.contains_key("gen_ai.tool.call.result"));
    }
}

#[test]
fn gen_ai_tool_identity_ignores_payload_keys_and_blank_metadata() {
    let mut event = make_start_event(
        Uuid::now_v7(),
        None,
        "search",
        ScopeType::Tool,
        Some(json!({
            "gen_ai.tool.name": "spoofed", "description": "private argument", "tool_type": "spoofed",
            "tool_call_id": "spoofed", "gen_ai.agent.name": "spoofed"
        })),
    );
    let attrs = attr_map(&crate::test_support::otel_genai::start_attributes(&event));
    assert_eq!(
        attrs.get("gen_ai.tool.name").map(String::as_str),
        Some("search")
    );
    assert_eq!(attrs.len(), 3);
    if let Event::Scope(scope) = &mut event {
        scope.base.metadata = Some(json!({
            "gen_ai.tool.name": " ", "gen_ai.tool.call.id": "", "tool_call_id": "call-1",
            "gen_ai.tool.type": false, "tool_type": "function", "tool_description": "Search docs",
            "gen_ai.agent.name": "researcher"
        }));
    }
    let attrs = attr_map(&crate::test_support::otel_genai::start_attributes(&event));
    assert_eq!(
        attrs.get("gen_ai.tool.name").map(String::as_str),
        Some("search")
    );
    assert_eq!(
        attrs.get("gen_ai.tool.call.id").map(String::as_str),
        Some("call-1")
    );
    assert_eq!(
        attrs.get("gen_ai.tool.type").map(String::as_str),
        Some("function")
    );
    assert_eq!(
        attrs.get("gen_ai.tool.description").map(String::as_str),
        Some("Search docs")
    );
    assert_eq!(
        attrs.get("gen_ai.agent.name").map(String::as_str),
        Some("researcher")
    );
}

#[test]
fn gen_ai_tool_definitions_use_minimal_standard_schema() {
    let request = serde_json::from_value::<AnnotatedLlmRequest>(json!({
        "messages": [],
        "tools": [
            {"type": "function", "function": {
                "name": "search", "description": "large private description",
                "parameters": {"type": "object"}
            }},
            {"type": "provider_native", "provider": "anthropic", "kind": "web_search",
             "value": {"type": "web_search_20250305", "name": "web_search", "private": "omitted"}},
            {"type": "provider_native", "provider": "openai_responses", "kind": "web_search",
             "value": {"type": "web_search"}},
            {"type": "provider_native", "provider": "gemini", "kind": "codeExecution",
             "value": {"codeExecution": {}, "googleSearch": {}, "unknownTool": {}}},
            {"type": "provider_native", "provider": "unknown", "kind": "web_search",
             "value": {"type": "web_search"}},
            {"type": "provider_native", "provider": "openai_responses", "kind": "unknown",
             "value": {"type": "unknown"}},
            {"type": "function", "function": {"name": " "}}
        ]
    }))
    .unwrap();
    let event = make_scope_event_with_profile(
        ScopeCategory::Start,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_request(Arc::new(request))
                .build(),
        ),
    );
    let enabled = attr_map(&crate::test_support::otel_genai::start_attributes(&event));
    let definitions: Json =
        serde_json::from_str(enabled.get("gen_ai.tool.definitions").unwrap()).unwrap();
    assert_eq!(
        definitions,
        json!([
            {"type": "function", "name": "search"},
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "web_search", "name": "web_search"},
            {"type": "codeExecution", "name": "codeExecution"},
            {"type": "googleSearch", "name": "googleSearch"}
        ])
    );
}

#[test]
fn gen_ai_projection_emits_normalized_response_attributes() {
    let event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"answer": "ok"})),
        Some(
            CategoryProfile::builder()
                .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                    id: Some("response-1".to_string()),
                    model: Some("model-1".to_string()),
                    finish_reason: Some(FinishReason::ToolUse),
                    usage: Some(Usage {
                        prompt_tokens: Some(13),
                        completion_tokens: Some(8),
                        total_tokens: Some(21),
                        cache_read_tokens: Some(5),
                        cache_write_tokens: Some(3),
                        uncached_input_tokens: None,
                        cost: None,
                    }),
                    ..empty_annotated_response()
                }))
                .build(),
        ),
    );

    let attributes = attr_map(&crate::test_support::otel_genai::end_attributes(&event));
    assert_eq!(
        attributes.get("gen_ai.response.id"),
        Some(&"response-1".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.response.model"),
        Some(&"model-1".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.response.finish_reasons"),
        Some(&"[\"tool_call\"]".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.usage.input_tokens"),
        Some(&"13".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.usage.output_tokens"),
        Some(&"8".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.usage.cache_read.input_tokens"),
        Some(&"5".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.usage.cache_write.input_tokens"),
        Some(&"3".to_string())
    );
}

#[test]
fn gen_ai_end_projection_preserves_explicit_error_type() {
    let event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name("chat")
            .metadata(json!({
                "otel.status_code": "ERROR",
                "otel.status_description": "invalid argument: invalid value",
                "error.type": "invalid_argument",
            }))
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::from(ScopeType::Llm),
        None,
    ));

    let attributes = attr_map(&crate::test_support::otel_genai::end_attributes(&event));
    assert_eq!(
        attributes.get("error.type"),
        Some(&"invalid_argument".to_string())
    );
}

#[test]
fn failed_descendant_classification_and_exception_propagate_to_agent_span() {
    for otel_type in [OpenTelemetryType::Full, OpenTelemetryType::GenAi] {
        let (provider, exporter) = make_provider();
        let mut processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings(
                provider,
                "error-propagation-test".to_string(),
                otel_type,
                MarkProjection::default(),
                default_mark_exclude_names(),
                Vec::new(),
            );
        let agent_uuid = Uuid::now_v7();
        let function_uuid = Uuid::now_v7();
        let llm_uuid = Uuid::now_v7();
        processor.process(&crate::test_support::event_view(&make_start_event(
            agent_uuid,
            None,
            "agent",
            ScopeType::Agent,
            None,
        )));
        let llm_parent_uuid = if otel_type == OpenTelemetryType::GenAi {
            processor.process(&crate::test_support::event_view(&make_start_event(
                function_uuid,
                Some(agent_uuid),
                "function",
                ScopeType::Function,
                None,
            )));
            function_uuid
        } else {
            agent_uuid
        };
        processor.process(&crate::test_support::event_view(&make_start_event(
            llm_uuid,
            Some(llm_parent_uuid),
            "chat",
            ScopeType::Llm,
            None,
        )));
        processor.process(&crate::test_support::event_view(
            &make_end_event_with_metadata(
                llm_uuid,
                Some(llm_parent_uuid),
                "chat",
                ScopeType::Llm,
                json!({
                    "otel.status_code": "ERROR",
                    "otel.status_description": "internal error: ValueError: boom",
                    "error.type": "internal_error",
                    "exception.type": "ValueError",
                }),
            ),
        ));
        if otel_type == OpenTelemetryType::GenAi {
            processor.process(&crate::test_support::event_view(
                &make_end_event_with_metadata(
                    function_uuid,
                    Some(agent_uuid),
                    "function",
                    ScopeType::Function,
                    json!({
                        "otel.status_code": "ERROR",
                        "otel.status_description": "internal error: ValueError: boom",
                    }),
                ),
            ));
        }
        processor.process(&crate::test_support::event_view(
            &make_end_event_with_metadata(
                agent_uuid,
                None,
                "agent",
                ScopeType::Agent,
                json!({
                    "otel.status_code": "ERROR",
                    "otel.status_description": "internal error: ValueError: boom",
                }),
            ),
        ));
        processor.force_flush().unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        let expected_span_count = if otel_type == OpenTelemetryType::GenAi {
            3
        } else {
            2
        };
        assert_eq!(spans.len(), expected_span_count);
        for span in &spans {
            assert_eq!(
                attr_map(&span.attributes).get("error.type"),
                Some(&"internal_error".to_string())
            );
            let exception = span
                .events
                .events
                .iter()
                .find(|event| event.name.as_ref() == "exception")
                .expect("expected exception event");
            assert_eq!(
                attr_map(&exception.attributes).get("exception.type"),
                Some(&"ValueError".to_string())
            );
        }
    }
}

#[test]
fn generic_function_error_propagates_to_agent_span() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings(
        provider,
        "suppressed-error-propagation-test".to_string(),
        OpenTelemetryType::GenAi,
        MarkProjection::default(),
        default_mark_exclude_names(),
        Vec::new(),
    );
    let agent_uuid = Uuid::now_v7();
    let function_uuid = Uuid::now_v7();
    processor.process(&crate::test_support::event_view(&make_start_event(
        agent_uuid,
        None,
        "agent",
        ScopeType::Agent,
        None,
    )));
    processor.process(&crate::test_support::event_view(&make_start_event(
        function_uuid,
        Some(agent_uuid),
        "function",
        ScopeType::Function,
        None,
    )));
    processor.process(&crate::test_support::event_view(
        &make_end_event_with_metadata(
            function_uuid,
            Some(agent_uuid),
            "function",
            ScopeType::Function,
            json!({
                "otel.status_code": "ERROR",
                "error.type": "internal_error",
                "exception.type": "ValueError",
            }),
        ),
    ));
    processor.process(&crate::test_support::event_view(
        &make_end_event_with_metadata(
            agent_uuid,
            None,
            "agent",
            ScopeType::Agent,
            json!({"otel.status_code": "ERROR"}),
        ),
    ));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 2);
    let agent_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "invoke_agent agent")
        .expect("expected agent span");
    let function_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "function")
        .expect("expected generic function span");
    assert_eq!(
        function_span.parent_span_id,
        agent_span.span_context.span_id()
    );
    assert_eq!(
        attr_map(&agent_span.attributes).get("error.type"),
        Some(&"internal_error".to_string())
    );
    let exception = agent_span
        .events
        .events
        .iter()
        .find(|event| event.name.as_ref() == "exception")
        .expect("expected propagated exception event");
    assert_eq!(
        attr_map(&exception.attributes).get("exception.type"),
        Some(&"ValueError".to_string())
    );
}

#[test]
fn generic_parent_error_propagation_isolated_by_trace_id() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings(
        provider,
        "error-trace-isolation-test".to_string(),
        OpenTelemetryType::GenAi,
        MarkProjection::default(),
        default_mark_exclude_names(),
        Vec::new(),
    );
    let shared_span_id = [0xAB; 8];
    let mut first_agent_bytes = [0x11; 16];
    first_agent_bytes[8..].copy_from_slice(&shared_span_id);
    let mut second_agent_bytes = [0x22; 16];
    second_agent_bytes[8..].copy_from_slice(&shared_span_id);
    let cases = [
        (
            Uuid::from_bytes(first_agent_bytes),
            Uuid::now_v7(),
            Uuid::now_v7(),
            "first_error",
            "FirstException",
        ),
        (
            Uuid::from_bytes(second_agent_bytes),
            Uuid::now_v7(),
            Uuid::now_v7(),
            "second_error",
            "SecondException",
        ),
    ];
    assert_eq!(
        relay_span_id(cases[0].0),
        relay_span_id(cases[1].0),
        "fixture agents must share a span ID"
    );
    assert_ne!(
        relay_trace_id(cases[0].0),
        relay_trace_id(cases[1].0),
        "fixture agents must belong to different traces"
    );

    for (agent_uuid, function_uuid, llm_uuid, _, _) in cases {
        processor.process(&crate::test_support::event_view(&make_start_event(
            agent_uuid,
            None,
            "agent",
            ScopeType::Agent,
            None,
        )));
        processor.process(&crate::test_support::event_view(&make_start_event(
            function_uuid,
            Some(agent_uuid),
            "function",
            ScopeType::Function,
            None,
        )));
        processor.process(&crate::test_support::event_view(&make_start_event(
            llm_uuid,
            Some(function_uuid),
            "chat",
            ScopeType::Llm,
            None,
        )));
    }
    for (agent_uuid, function_uuid, llm_uuid, error_type, exception_type) in cases {
        processor.process(&crate::test_support::event_view(
            &make_end_event_with_metadata(
                llm_uuid,
                Some(function_uuid),
                "chat",
                ScopeType::Llm,
                json!({
                    "otel.status_code": "ERROR",
                    "error.type": error_type,
                    "exception.type": exception_type,
                }),
            ),
        ));
        processor.process(&crate::test_support::event_view(
            &make_end_event_with_metadata(
                function_uuid,
                Some(agent_uuid),
                "function",
                ScopeType::Function,
                json!({"otel.status_code": "ERROR"}),
            ),
        ));
    }
    for (agent_uuid, _, _, _, _) in cases {
        processor.process(&crate::test_support::event_view(
            &make_end_event_with_metadata(
                agent_uuid,
                None,
                "agent",
                ScopeType::Agent,
                json!({"otel.status_code": "ERROR"}),
            ),
        ));
    }
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 6);
    for (agent_uuid, _, _, error_type, exception_type) in cases {
        let agent_span = spans
            .iter()
            .find(|span| {
                span.span_context.trace_id() == relay_trace_id(agent_uuid)
                    && span.parent_span_id == SpanId::INVALID
            })
            .expect("expected agent span for trace");
        assert_eq!(
            attr_map(&agent_span.attributes).get("error.type"),
            Some(&error_type.to_string())
        );
        let exception = agent_span
            .events
            .events
            .iter()
            .find(|event| event.name.as_ref() == "exception")
            .expect("expected propagated exception event");
        assert_eq!(
            attr_map(&exception.attributes).get("exception.type"),
            Some(&exception_type.to_string())
        );
    }
}

#[test]
fn gen_ai_projection_exports_stream_time_to_first_chunk_without_response_decoding() {
    let event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "openai.chat.completions",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .time_to_first_chunk(0.125)
                .build(),
        ),
    );

    let attributes = attr_map(&crate::test_support::otel_genai::end_attributes(&event));
    assert_eq!(
        attributes.get("gen_ai.response.time_to_first_chunk"),
        Some(&"0.125".to_string())
    );
}

#[test]
fn gen_ai_projection_covers_message_variants_and_empty_input() {
    let annotated_request = serde_json::from_value::<AnnotatedLlmRequest>(json!({
        "instructions": "Be concise.",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{not-json"
                    }
                }]
            },
            {
                "role": "tool",
                "content": "result",
                "tool_call_id": "call-1"
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "call-2",
                    "content": "claude result"
                }]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-3",
                        "content": "mixed result"
                    },
                    {
                        "type": "text",
                        "text": "continue"
                    }
                ]
            },
            {
                "role": "user",
                "content": []
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "provider_native",
                        "provider": "example",
                        "kind": "reasoning",
                        "value": {"content": "provider payload"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"}
                    }
                ]
            }
        ]
    }))
    .unwrap();
    let event = make_scope_event_with_profile(
        ScopeCategory::Start,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_request(std::sync::Arc::new(annotated_request))
                .build(),
        ),
    );
    let attributes = attr_map(&crate::test_support::otel_genai::start_attributes(&event));
    assert_eq!(
        serde_json::from_str::<Json>(&attributes["gen_ai.system_instructions"]).unwrap(),
        json!([{"type": "text", "content": "Be concise."}])
    );
    assert_eq!(
        serde_json::from_str::<Json>(&attributes["gen_ai.input.messages"]).unwrap(),
        json!([
            {
                "role": "assistant",
                "parts": [{
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "lookup",
                    "arguments": "{not-json"
                }]
            },
            {
                "role": "tool",
                "parts": [{
                    "type": "tool_call_response",
                    "id": "call-1",
                    "response": "result"
                }]
            },
            {
                "role": "tool",
                "parts": [{
                    "type": "tool_call_response",
                    "id": "call-2",
                    "response": "claude result"
                }]
            },
            {
                "role": "user",
                "parts": [
                    {
                        "type": "tool_call_response",
                        "id": "call-3",
                        "response": "mixed result"
                    },
                    {
                        "type": "text",
                        "content": "continue"
                    }
                ]
            },
            {
                "role": "user",
                "parts": []
            },
            {
                "role": "user",
                "parts": [
                    {"type": "reasoning", "content": "provider payload"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"}
                    }
                ]
            }
        ])
    );

    let empty_event = make_scope_event_with_profile(
        ScopeCategory::Start,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_request(std::sync::Arc::new(AnnotatedLlmRequest::default()))
                .build(),
        ),
    );
    let attributes = attr_map(&crate::test_support::otel_genai::start_attributes(
        &empty_event,
    ));
    assert!(!attributes.contains_key("gen_ai.input.messages"));
    assert!(!attributes.contains_key("gen_ai.system_instructions"));
}

#[test]
fn gen_ai_projection_covers_output_tool_calls_and_empty_output() {
    let tool_call_event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                    tool_calls: Some(vec![ResponseToolCall {
                        id: "call-1".to_string(),
                        name: "lookup".to_string(),
                        arguments: json!({"city": "Paris"}),
                    }]),
                    finish_reason: Some(FinishReason::ToolUse),
                    ..empty_annotated_response()
                }))
                .build(),
        ),
    );
    let attributes = attr_map(&crate::test_support::otel_genai::end_attributes(
        &tool_call_event,
    ));
    assert_eq!(
        serde_json::from_str::<Json>(&attributes["gen_ai.output.messages"]).unwrap(),
        json!([{
            "role": "assistant",
            "parts": [{
                "type": "tool_call",
                "id": "call-1",
                "name": "lookup",
                "arguments": {"city": "Paris"}
            }],
            "finish_reason": "tool_call"
        }])
    );

    let empty_event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_response(std::sync::Arc::new(empty_annotated_response()))
                .build(),
        ),
    );
    let attributes = attr_map(&crate::test_support::otel_genai::end_attributes(
        &empty_event,
    ));
    assert!(!attributes.contains_key("gen_ai.output.messages"));
}

#[test]
fn gen_ai_projection_reads_nested_scalar_fallbacks() {
    let event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name("embed")
            .metadata(json!({
                "request": {"provider": true, "server_port": 8443},
                "response": {"response_model": 17},
                "usage": {"input_tokens": u64::MAX}
            }))
            .data(json!({"usage": {"prompt_tokens": 23}}))
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::from(ScopeType::Embedder),
        None,
    ));
    let attributes = attr_map(&crate::test_support::otel_genai::end_attributes(&event));
    assert_eq!(
        attributes.get("gen_ai.response.model"),
        Some(&"17".to_string())
    );
    assert_eq!(
        attributes.get("gen_ai.usage.input_tokens"),
        Some(&"23".to_string())
    );
}

#[test]
fn records_span_start_mark_and_end() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());
    let root_uuid = Uuid::now_v7();

    let start = make_start_event(
        root_uuid,
        None,
        "search",
        ScopeType::Tool,
        Some(json!({"query": "hello"})),
    );
    processor.process(&crate::test_support::event_view(&start));

    let mark = make_mark_event(Some(root_uuid), "checkpoint", Some(json!({"step": 1})));
    processor.process(&crate::test_support::event_view(&mark));

    let end = make_end_event(
        root_uuid,
        None,
        "search",
        ScopeType::Tool,
        Some(json!({"result": "ok"})),
    );
    processor.process(&crate::test_support::event_view(&end));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let span = &spans[0];
    assert_eq!(span.name.as_ref(), "search");
    assert_eq!(span.events.events.len(), 1);
    assert_eq!(span.events.events[0].name.as_ref(), "checkpoint");

    let attributes = attr_map(&span.attributes);
    assert_eq!(
        attributes.get("nemo_relay.uuid"),
        Some(&root_uuid.to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.start.input.query"),
        Some(&"hello".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.end.output.result"),
        Some(&"ok".to_string())
    );
}

#[test]
fn metric_schema_marks_are_not_projected_to_direct_traces() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());
    let root_uuid = Uuid::now_v7();

    processor.process(&crate::test_support::event_view(&make_start_event(
        root_uuid,
        None,
        "agent",
        ScopeType::Agent,
        None,
    )));
    processor.process(&crate::test_support::event_view(&Event::Mark(
        MarkEvent::new(
            BaseEvent::builder()
                .parent_uuid(root_uuid)
                .name("tokens-saved")
                .data(json!({
                    "measurements": [{
                        "name": "example.tokens.saved",
                        "kind": "counter",
                        "value_type": "u64",
                        "value": 42
                    }]
                }))
                .data_schema(
                    DataSchema::builder()
                        .name(METRIC_DATA_SCHEMA_NAME)
                        .version(METRIC_DATA_SCHEMA_VERSION)
                        .build(),
                )
                .build(),
            None,
            None,
        ),
    )));
    processor.process(&crate::test_support::event_view(&Event::Mark(
        MarkEvent::new(
            BaseEvent::builder()
                .parent_uuid(root_uuid)
                .name("future-metric")
                .data(json!({"measurements": []}))
                .data_schema(
                    DataSchema::builder()
                        .name(METRIC_DATA_SCHEMA_NAME)
                        .version("999")
                        .build(),
                )
                .build(),
            None,
            None,
        ),
    )));
    processor.process(&crate::test_support::event_view(&Event::Mark(
        MarkEvent::new(
            BaseEvent::builder()
                .parent_uuid(root_uuid)
                .name("invalid-metric")
                .data(json!({"measurements": []}))
                .data_schema(
                    DataSchema::builder()
                        .name(METRIC_DATA_SCHEMA_NAME)
                        .version(METRIC_DATA_SCHEMA_VERSION)
                        .build(),
                )
                .build(),
            None,
            None,
        ),
    )));
    processor.process(&crate::test_support::event_view(&make_mark_event(
        Some(root_uuid),
        "routing-decision",
        None,
    )));
    processor.process(&crate::test_support::event_view(&make_end_event(
        root_uuid,
        None,
        "agent",
        ScopeType::Agent,
        None,
    )));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].events.events.len(), 1);
    assert_eq!(spans[0].events.events[0].name.as_ref(), "routing-decision");
    assert_eq!(processor.invalid_metric_count, 2);
}

#[test]
fn derives_span_ids_from_relay_uuids() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection(
        provider.clone(),
        "test-scope".to_string(),
        MarkProjection::Tool,
    );

    let root_uuid = Uuid::from_u128(0x018f_0f0f_0f0f_7000_8123_4567_89ab_cdef);
    let child_uuid = Uuid::from_u128(0x018f_0f0f_0f10_7000_8fed_cba9_8765_4321);
    let orphan_uuid = Uuid::from_u128(0x018f_0f0f_0f11_7000_8011_2233_4455_6677);

    processor.process(&crate::test_support::event_view(&make_start_event(
        root_uuid,
        None,
        "agent",
        ScopeType::Agent,
        None,
    )));
    processor.process(&crate::test_support::event_view(&make_start_event(
        child_uuid,
        Some(root_uuid),
        "model-call",
        ScopeType::Llm,
        None,
    )));
    processor.process(&crate::test_support::event_view(&make_end_event(
        child_uuid,
        Some(root_uuid),
        "model-call",
        ScopeType::Llm,
        None,
    )));
    processor.process(&crate::test_support::event_view(&make_end_event(
        root_uuid,
        None,
        "agent",
        ScopeType::Agent,
        None,
    )));
    processor.process(&crate::test_support::event_view(&Event::Mark(
        MarkEvent::new(
            BaseEvent::builder()
                .uuid(orphan_uuid)
                .name("checkpoint")
                .build(),
            None,
            None,
        ),
    )));

    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 3);
    let parent = spans
        .iter()
        .find(|span| span.name.as_ref() == "agent")
        .unwrap();
    let child = spans
        .iter()
        .find(|span| span.name.as_ref() == "model-call")
        .unwrap();
    let orphan = spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:checkpoint")
        .unwrap();

    assert_eq!(
        parent.span_context.trace_id().to_bytes(),
        *root_uuid.as_bytes()
    );
    assert_eq!(
        parent.span_context.span_id().to_bytes(),
        root_uuid.as_bytes()[8..]
    );
    assert_eq!(
        child.span_context.trace_id(),
        parent.span_context.trace_id()
    );
    assert_eq!(
        child.span_context.span_id().to_bytes(),
        child_uuid.as_bytes()[8..]
    );
    assert_eq!(child.parent_span_id, parent.span_context.span_id());
    assert!(!child.parent_span_is_remote);
    assert_eq!(
        orphan.span_context.trace_id().to_bytes(),
        *orphan_uuid.as_bytes()
    );
    assert_eq!(
        orphan.span_context.span_id().to_bytes(),
        orphan_uuid.as_bytes()[8..]
    );
}

#[test]
fn atif_lineage_correlates_with_otel_span_attributes() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());

    let agent_uuid = Uuid::now_v7();
    let llm_uuid = Uuid::now_v7();
    let atif_exporter = AtifExporter::new(
        agent_uuid.to_string(),
        AtifAgentInfo {
            name: "test-agent".to_string(),
            version: "1.0.0".to_string(),
            model_name: None,
            tool_definitions: None,
            extra: None,
        },
    )
    .with_host(crate::test_support::detached_host());
    let atif_subscriber = atif_exporter.subscriber();

    let events = vec![
        make_start_event(agent_uuid, None, "agent", ScopeType::Agent, None),
        make_start_event(
            llm_uuid,
            Some(agent_uuid),
            "model-call",
            ScopeType::Llm,
            Some(json!({"messages": [{"role": "user", "content": "hello"}]})),
        ),
        make_end_event(
            llm_uuid,
            Some(agent_uuid),
            "model-call",
            ScopeType::Llm,
            Some(json!({"content": "hi", "role": "assistant"})),
        ),
        make_end_event(agent_uuid, None, "agent", ScopeType::Agent, None),
    ];

    for event in &events {
        processor.process(&crate::test_support::event_view(event));
        atif_subscriber(event);
    }
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let agent_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "agent")
        .unwrap();
    let llm_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "model-call")
        .unwrap();
    let agent_attributes = attr_map(&agent_span.attributes);
    let llm_attributes = attr_map(&llm_span.attributes);

    assert_eq!(
        agent_attributes.get("nemo_relay.uuid"),
        Some(&agent_uuid.to_string())
    );
    assert_eq!(
        llm_attributes.get("nemo_relay.uuid"),
        Some(&llm_uuid.to_string())
    );
    assert_eq!(
        llm_attributes.get("nemo_relay.parent_uuid"),
        Some(&agent_uuid.to_string())
    );

    let trajectory = atif_exporter.export().unwrap();
    assert_eq!(trajectory.session_id, agent_uuid.to_string());
    let agent_step = trajectory
        .steps
        .iter()
        .find(|step| step.source == "agent")
        .unwrap();
    let extra: AtifStepExtra = serde_json::from_value(agent_step.extra.clone().unwrap()).unwrap();

    assert_eq!(
        llm_attributes.get("nemo_relay.uuid"),
        Some(&extra.ancestry.function_id)
    );
    assert_eq!(extra.ancestry.parent_id, Some(trajectory.session_id));
}

#[test]
fn orphan_marks_become_zero_duration_spans() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());
    let mark = make_mark_event(None, "detached", Some(json!({"kind": "standalone"})));

    processor.process(&crate::test_support::event_view(&mark));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let span = &spans[0];
    assert_eq!(span.name.as_ref(), "mark:detached");
    assert_eq!(span.start_time, span.end_time);

    let attributes = attr_map(&span.attributes);
    assert_eq!(
        attributes.get("nemo_relay.mark.orphan"),
        Some(&"true".to_string())
    );
}

#[test]
fn tool_projection_emits_generic_mark_as_parented_zero_duration_span() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection(
        provider.clone(),
        "test-scope".to_string(),
        MarkProjection::Tool,
    );
    let parent_uuid = Uuid::now_v7();
    let start = make_start_event(parent_uuid, None, "agent-turn", ScopeType::Agent, None);
    let mark = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid(parent_uuid)
            .name("plugin.output_compacted")
            .data(json!({"count": 3}))
            .build(),
        Some(EventCategory::custom()),
        Some(
            CategoryProfile::builder()
                .subtype("example.compaction")
                .build(),
        ),
    ));
    let end = make_end_event(parent_uuid, None, "agent-turn", ScopeType::Agent, None);

    processor.process(&crate::test_support::event_view(&start));
    processor.process(&crate::test_support::event_view(&mark));
    processor.process(&crate::test_support::event_view(&end));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 2);
    let parent = spans
        .iter()
        .find(|span| span.name.as_ref() == "agent-turn")
        .unwrap();
    let projected = spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:plugin.output_compacted")
        .unwrap();
    assert!(parent.events.events.is_empty());
    assert_eq!(projected.parent_span_id, parent.span_context.span_id());
    assert_eq!(projected.start_time, projected.end_time);
    assert!(parent.start_time <= projected.start_time);
    assert!(projected.end_time <= parent.end_time);

    let attributes = attr_map(&projected.attributes);
    assert_eq!(
        attributes.get("nemo_relay.mark.projection"),
        Some(&"tool".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.scope_type"),
        Some(&"tool".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.mark.data.count"),
        Some(&"3".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.mark.category"),
        Some(&"custom".to_string())
    );
    assert_eq!(
        attributes.get("nemo_relay.mark.category_profile.subtype"),
        Some(&"example.compaction".to_string())
    );
    assert!(!attributes.contains_key("nemo_relay.mark.orphan"));
}

#[test]
fn tool_projection_exclusion_keeps_custom_mark_as_native_event() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection_and_exclusions(
        provider.clone(),
        "test-scope".to_string(),
        MarkProjection::Tool,
        vec!["plugin.excluded".to_string()],
    );
    processor.process(&crate::test_support::event_view(&make_mark_event(
        None,
        "plugin.excluded",
        Some(json!({"count": 3})),
    )));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    let attributes = attr_map(&spans[0].attributes);
    assert!(!attributes.contains_key("nemo_relay.mark.projection"));
    assert_eq!(
        attributes.get("nemo_relay.mark.orphan"),
        Some(&"true".to_string())
    );
}

#[test]
fn tool_projection_reuses_completed_parent_context_for_late_marks() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection(
        provider.clone(),
        "test-scope".to_string(),
        MarkProjection::Tool,
    );
    let parent_uuid = Uuid::now_v7();

    processor.process(&crate::test_support::event_view(&make_start_event(
        parent_uuid,
        None,
        "completed-tool",
        ScopeType::Tool,
        None,
    )));
    processor.process(&crate::test_support::event_view(&make_end_event(
        parent_uuid,
        None,
        "completed-tool",
        ScopeType::Tool,
        None,
    )));
    processor.process(&crate::test_support::event_view(&make_mark_event(
        Some(parent_uuid),
        "plugin.late_checkpoint",
        Some(json!({"status": "complete"})),
    )));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    let parent = spans
        .iter()
        .find(|span| span.name.as_ref() == "completed-tool")
        .unwrap();
    let projected = spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:plugin.late_checkpoint")
        .unwrap();
    assert_eq!(projected.parent_span_id, parent.span_context.span_id());
    assert_eq!(
        projected.span_context.trace_id(),
        parent.span_context.trace_id()
    );
    assert_eq!(
        attr_map(&projected.attributes).get("nemo_relay.mark.orphan"),
        Some(&"true".to_string())
    );
}

#[test]
fn tool_projection_keeps_llm_chunks_as_parent_span_events() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new_with_mark_projection(
        provider.clone(),
        "test-scope".to_string(),
        MarkProjection::Tool,
    );
    let parent_uuid = Uuid::now_v7();
    let start = make_start_event(parent_uuid, None, "llm", ScopeType::Llm, None);
    let chunk = make_mark_event(Some(parent_uuid), "llm.chunk", Some(json!({"delta": "x"})));
    let aliased_chunk = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid(parent_uuid)
            .name("hook_mark")
            .metadata(json!({"hook_event_name": "llm.chunk"}))
            .data(json!({"delta": "y"}))
            .build(),
        None,
        None,
    ));
    let end = make_end_event(parent_uuid, None, "llm", ScopeType::Llm, None);

    processor.process(&crate::test_support::event_view(&start));
    processor.process(&crate::test_support::event_view(&chunk));
    processor.process(&crate::test_support::event_view(&aliased_chunk));
    processor.process(&crate::test_support::event_view(&end));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].events.events.len(), 2);
    assert_eq!(spans[0].events.events[0].name.as_ref(), "llm.chunk");
    assert_eq!(spans[0].events.events[1].name.as_ref(), "hook_mark");
}

#[test]
fn late_parented_marks_reuse_completed_parent_trace_context() {
    let (provider, exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider.clone(), "test-scope".to_string());
    let tool_uuid = Uuid::now_v7();

    processor.process(&crate::test_support::event_view(&make_start_event(
        tool_uuid,
        None,
        "terminal",
        ScopeType::Tool,
        None,
    )));
    processor.process(&crate::test_support::event_view(&make_end_event(
        tool_uuid,
        None,
        "terminal",
        ScopeType::Tool,
        Some(json!({"status": "done"})),
    )));
    processor.process(&crate::test_support::event_view(&make_mark_event(
        Some(tool_uuid),
        "plugin.output_compacted",
        Some(json!({"estimated_tokens_saved": 42})),
    )));
    processor.force_flush().unwrap();

    let spans = exporter.get_finished_spans().unwrap();
    assert_eq!(spans.len(), 2);
    let tool_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "terminal")
        .unwrap();
    let mark_span = spans
        .iter()
        .find(|span| span.name.as_ref() == "mark:plugin.output_compacted")
        .unwrap();

    assert_eq!(
        mark_span.span_context.trace_id(),
        tool_span.span_context.trace_id()
    );
    assert_eq!(mark_span.parent_span_id, tool_span.span_context.span_id());
    assert!(!mark_span.parent_span_is_remote);

    let attributes = attr_map(&mark_span.attributes);
    assert_eq!(
        attributes.get("nemo_relay.mark.orphan"),
        Some(&"true".to_string())
    );
}

#[test]
fn process_start_removes_completed_span_expiry_index_entry() {
    let (provider, _exporter) = make_provider();
    let mut processor = OtelEventProcessor::new(provider, "test-scope".to_string());
    let tool_uuid = Uuid::now_v7();

    processor.process(&crate::test_support::event_view(&make_start_event(
        tool_uuid,
        None,
        "terminal",
        ScopeType::Tool,
        None,
    )));
    processor.process(&crate::test_support::event_view(&make_end_event(
        tool_uuid,
        None,
        "terminal",
        ScopeType::Tool,
        Some(json!({"status": "done"})),
    )));
    assert!(processor.completed_span_contexts.contains_key(&tool_uuid));
    assert_eq!(
        processor
            .completed_span_expiry_index
            .values()
            .filter(|uuids| uuids.contains(&tool_uuid))
            .count(),
        1
    );

    processor.process(&crate::test_support::event_view(&make_start_event(
        tool_uuid,
        None,
        "terminal",
        ScopeType::Tool,
        None,
    )));
    assert!(!processor.completed_span_contexts.contains_key(&tool_uuid));
    assert!(
        processor
            .completed_span_expiry_index
            .values()
            .all(|uuids| !uuids.contains(&tool_uuid))
    );

    processor.process(&crate::test_support::event_view(&make_end_event(
        tool_uuid,
        None,
        "terminal",
        ScopeType::Tool,
        Some(json!({"status": "done"})),
    )));
    assert!(processor.completed_span_contexts.contains_key(&tool_uuid));
    assert_eq!(
        processor
            .completed_span_expiry_index
            .values()
            .filter(|uuids| uuids.contains(&tool_uuid))
            .count(),
        1
    );
}

#[test]
fn semantic_scope_type_and_span_kind_follow_event_variants() {
    let scope_event = make_start_event(
        Uuid::now_v7(),
        None,
        "guardrail",
        ScopeType::Guardrail,
        Some(json!({"input": true})),
    );
    assert_eq!(
        semantic_scope_type(&scope_event),
        Some(ScopeType::Guardrail)
    );
    assert_eq!(span_kind(&scope_event), SpanKind::Internal);

    let remote_tool = make_scope_event_with_attributes(
        ScopeCategory::Start,
        Uuid::now_v7(),
        None,
        "search",
        ScopeType::Tool,
        Some(json!({"query": "hello"})),
        tool_attributes_to_strings(ToolAttributes::REMOTE),
    );
    assert_eq!(semantic_scope_type(&remote_tool), Some(ScopeType::Tool));
    assert_eq!(span_kind(&remote_tool), SpanKind::Client);
    let remote_tool_attributes = attr_map(&start_attributes(&remote_tool));
    assert_eq!(
        remote_tool_attributes.get("nemo_relay.handle_attributes"),
        Some(&"[\"remote\"]".to_string())
    );
    assert!(!remote_tool_attributes.contains_key("nemo_relay.handle_attributes_json"));

    let llm_event = make_end_event(
        Uuid::now_v7(),
        None,
        "model-call",
        ScopeType::Llm,
        Some(json!({"result": "hello"})),
    );
    assert_eq!(semantic_scope_type(&llm_event), Some(ScopeType::Llm));
    assert_eq!(span_kind(&llm_event), SpanKind::Client);

    let mark = make_mark_event(None, "checkpoint", None);
    assert_eq!(semantic_scope_type(&mark), None);
    assert_eq!(span_kind(&mark), SpanKind::Internal);
}

#[test]
fn helper_functions_cover_additional_otel_branches() {
    assert_otel_scope_and_model_attribute_branches();
    assert_otel_tool_attribute_branches();
    assert_otel_catalog_cost_branches();
    assert_otel_normalized_cost_branches();
    assert_otel_manual_cost_branches();
}

#[test]
fn gen_ai_projection_preserves_named_messages_and_portable_tool_correlations() {
    let request = serde_json::from_value::<AnnotatedLlmRequest>(json!({
        "messages": [
            {"role": "system", "name": "policy", "content": "safe"},
            {"role": "developer", "name": "application", "content": "brief"},
            {"role": "function", "name": "legacy_lookup", "content": null},
            {"role": "tool_call", "id": "item-1", "call_id": "call-1",
             "name": "lookup", "arguments": {"city": "Paris"}},
            {"role": "tool_result", "id": "item-2", "call_id": "call-1",
             "output": {"temperature": 21}},
            {"role": "provider_native", "provider": "example", "kind": "reasoning",
             "value": {"role": "critic", "content": "check"}},
            {"role": "provider_native", "provider": "example", "kind": "opaque", "value": 42},
            {"role": "tool", "tool_call_id": "call-2", "content": [{"type": "text", "text": "done"}]}
        ]
    })).unwrap();
    let event = make_scope_event_with_profile(
        ScopeCategory::Start,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_request(Arc::new(request))
                .build(),
        ),
    );
    let attributes = attr_map(&crate::test_support::otel_genai::start_attributes(&event));
    assert_eq!(
        serde_json::from_str::<Json>(&attributes["gen_ai.input.messages"]).unwrap(),
        json!([
            {"role": "system", "name": "policy", "parts": [{"type": "text", "content": "safe"}]},
            {"role": "developer", "name": "application", "parts": [{"type": "text", "content": "brief"}]},
            {"role": "tool", "name": "legacy_lookup", "parts": [{"type": "tool_call_response", "response": null}]},
            {"role": "assistant", "parts": [{"type": "tool_call", "id": "call-1", "name": "lookup", "arguments": {"city": "Paris"}}]},
            {"role": "tool", "parts": [{"type": "tool_call_response", "id": "call-1", "response": {"temperature": 21}}]},
            {"role": "critic", "parts": [{"type": "reasoning", "role": "critic", "content": "check"}]},
            {"role": "provider_native", "parts": [{"type": "opaque", "content": 42}]},
            {"role": "tool", "parts": [{"type": "tool_call_response", "id": "call-2", "response": [{"type": "text", "text": "done"}]}]}
        ])
    );
}

#[test]
fn gen_ai_projection_preserves_multimodal_inputs_and_anthropic_tool_arguments() {
    let request = serde_json::from_value::<AnnotatedLlmRequest>(json!({
        "messages": [{"role": "assistant", "name": "helper", "content": [
            {"type": "refusal", "refusal": "cannot comply"},
            {"type": "tool_use", "id": "use-1", "name": "lookup", "input": {"city": "Paris"}},
            {"type": "image", "image": {"url": "https://example.com/image"}, "detail": "high"},
            {"type": "audio", "audio": {"data": "encoded", "format": "wav"}},
            {"type": "file", "file": {"id": "file-1"}}
        ]}]
    }))
    .unwrap();
    let event = make_scope_event_with_profile(
        ScopeCategory::Start,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        None,
        Some(
            CategoryProfile::builder()
                .annotated_request(Arc::new(request))
                .build(),
        ),
    );
    let attributes = attr_map(&crate::test_support::otel_genai::start_attributes(&event));
    assert_eq!(
        serde_json::from_str::<Json>(&attributes["gen_ai.input.messages"]).unwrap(),
        json!([{
            "role": "assistant", "name": "helper", "parts": [
                {"type": "text", "content": "cannot comply"},
                {"type": "tool_call", "id": "use-1", "name": "lookup", "arguments": {"city": "Paris"}},
                {"type": "image", "image": {"url": "https://example.com/image"}, "detail": "high"},
                {"type": "audio", "audio": {"data": "encoded", "format": "wav"}},
                {"type": "file", "file": {"id": "file-1"}}
            ]
        }])
    );
}

#[test]
fn gen_ai_embedding_encoding_formats_filter_blanks_and_reject_mixed_arrays() {
    for (value, expected) in [
        (json!("float"), Some("[\"float\"]")),
        (
            json!(["float", " ", "base64"]),
            Some("[\"float\",\"base64\"]"),
        ),
        (json!(["", " "]), None),
        (json!(["float", 3]), None),
        (json!(null), None),
    ] {
        let event = make_scope_event_with_profile(
            ScopeCategory::Start,
            Uuid::now_v7(),
            None,
            "embed",
            ScopeType::Embedder,
            Some(json!({"encoding_formats": value})),
            None,
        );
        let attributes = attr_map(&crate::test_support::otel_genai::start_attributes(&event));
        assert_eq!(
            attributes
                .get("gen_ai.request.encoding_formats")
                .map(String::as_str),
            expected
        );
    }
}

#[test]
fn otlp_timestamp_range_drops_invalid_events_and_preserves_boundaries() {
    let scope_at = |uuid, category, timestamp| {
        Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .uuid(uuid)
                .name("replay")
                .timestamp(timestamp)
                .build(),
            category,
            Vec::new(),
            EventCategory::from(ScopeType::Custom),
            None,
        ))
    };
    for otel_type in [
        OpenTelemetryType::Full,
        OpenTelemetryType::GenAi,
        OpenTelemetryType::OpenInference,
    ] {
        let (provider, exporter) = make_provider();
        let mut processor =
            OtelEventProcessor::new_with_mark_projection_and_exclusions_and_mappings(
                provider,
                "timestamp-range-test".to_string(),
                otel_type,
                MarkProjection::Event,
                Vec::new(),
                Vec::new(),
            );
        let valid_uuid = Uuid::now_v7();
        let epoch = DateTime::from_timestamp(0, 0).unwrap();
        processor.process(&crate::test_support::event_view(&scope_at(
            valid_uuid,
            ScopeCategory::Start,
            epoch,
        )));
        for value in [
            "1969-12-31T23:59:59Z",
            "1969-12-31T23:59:59.999999999Z",
            "2554-07-21T23:34:33.709551616Z",
            "9999-12-31T23:59:59.999999Z",
        ] {
            let timestamp = DateTime::parse_from_rfc3339(value)
                .unwrap()
                .with_timezone(&Utc);
            let invalid_uuid = Uuid::now_v7();
            processor.process(&crate::test_support::event_view(&scope_at(
                invalid_uuid,
                ScopeCategory::Start,
                timestamp,
            )));
            processor.process(&crate::test_support::event_view(&scope_at(
                invalid_uuid,
                ScopeCategory::End,
                timestamp,
            )));
            processor.process(&crate::test_support::event_view(&scope_at(
                valid_uuid,
                ScopeCategory::End,
                timestamp,
            )));
            for parent in [None, Some(valid_uuid)] {
                let mark = Event::Mark(MarkEvent::new(
                    BaseEvent::builder()
                        .parent_uuid_opt(parent)
                        .name("invalid-mark")
                        .timestamp(timestamp)
                        .build(),
                    None,
                    None,
                ));
                processor.process(&crate::test_support::event_view(&mark));
            }
            assert!(!processor.active_spans.contains_key(&invalid_uuid));
            assert!(processor.active_spans.contains_key(&valid_uuid));
            assert!(exporter.get_finished_spans().unwrap().is_empty());
        }
        assert_eq!(
            processor
                .runtime_diagnostics
                .snapshot()
                .get("otel.timestamp_out_of_range")
                .unwrap()
                .count,
            20,
        );
        let maximum = DateTime::parse_from_rfc3339("2554-07-21T23:34:33.709551615Z")
            .unwrap()
            .with_timezone(&Utc);
        processor.process(&crate::test_support::event_view(&scope_at(
            valid_uuid,
            ScopeCategory::End,
            maximum,
        )));
        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start_time, UNIX_EPOCH);
        // SystemTime uses 100-nanosecond ticks on Windows. Range validation
        // above still checks the original timestamp at nanosecond precision.
        assert_eq!(
            spans[0].end_time,
            UNIX_EPOCH + Duration::from_nanos(u64::MAX)
        );
        assert!(spans[0].events.is_empty());
        for value in [
            "2554-07-21T23:34:33.709551615Z",
            "2554-01-01T00:00:00Z",
            "2020-01-01T05:30:00.000001+05:30",
        ] {
            let timestamp = DateTime::parse_from_rfc3339(value)
                .unwrap()
                .with_timezone(&Utc);
            let uuid = Uuid::now_v7();
            processor.process(&crate::test_support::event_view(&scope_at(
                uuid,
                ScopeCategory::Start,
                timestamp,
            )));
            processor.process(&crate::test_support::event_view(&scope_at(
                uuid,
                ScopeCategory::End,
                timestamp,
            )));
            let spans = exporter.get_finished_spans().unwrap();
            let span = spans.last().unwrap();
            assert_eq!(span.start_time, to_system_time(timestamp));
            assert_eq!(span.end_time, to_system_time(timestamp));
        }
    }
}

#[test]
fn gen_ai_projection_reads_cache_tokens_by_the_declared_api_shape() {
    let usage = Usage {
        prompt_tokens: Some(150),
        completion_tokens: Some(20),
        total_tokens: Some(170),
        cache_read_tokens: Some(40),
        cache_write_tokens: None,
        uncached_input_tokens: Some(110),
        cost: None,
    };
    let openai_shaped = ApiSpecificResponse::OpenAIChat {
        logprobs: None,
        system_fingerprint: None,
        service_tier: None,
    };
    for (api_specific, input_tokens) in [(Some(openai_shaped), "150"), (None, "190")] {
        let event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
            None,
            "anthropic",
            ScopeType::Llm,
            Some(json!({"answer": "ok"})),
            Some(
                CategoryProfile::builder()
                    .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                        usage: Some(usage.clone()),
                        api_specific,
                        ..empty_annotated_response()
                    }))
                    .build(),
            ),
        );

        let attributes = attr_map(&crate::test_support::otel_genai::end_attributes(&event));
        assert_eq!(
            attributes.get("gen_ai.usage.input_tokens"),
            Some(&input_tokens.to_string())
        );
    }
}

fn assert_otel_catalog_cost_branches() {
    {
        let _pricing_guard = pricing_test_mutex().lock().unwrap();
        install_test_pricing("priced-model");
        let _reset_guard = ResetPricingResolverGuard;
        let llm_cost_event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
            None,
            "chat",
            ScopeType::Llm,
            Some(json!({"answer": "ok"})),
            Some(
                CategoryProfile::builder()
                    .model_name("priced-model")
                    .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                        usage: Some(Usage {
                            prompt_tokens: Some(1_000),
                            completion_tokens: Some(500),
                            total_tokens: Some(1_500),
                            cache_read_tokens: Some(200),
                            cache_write_tokens: None,
                            uncached_input_tokens: None,
                            cost: None,
                        }),
                        ..empty_annotated_response()
                    }))
                    .build(),
            ),
        );
        let llm_cost_attributes = attr_map(&end_attributes(&llm_cost_event));
        assert_eq!(
            llm_cost_attributes.get("nemo_relay.llm.cost.total"),
            Some(&"0.000435".to_string())
        );
        assert_eq!(
            llm_cost_attributes.get("nemo_relay.llm.cost.currency"),
            Some(&"USD".to_string())
        );
    }

    {
        let _pricing_guard = pricing_test_mutex().lock().unwrap();
        install_provider_disambiguation_pricing("priced-model");
        let _reset_guard = ResetPricingResolverGuard;
        let provider_qualified_cost_event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
            None,
            "test",
            ScopeType::Llm,
            Some(json!({"answer": "ok"})),
            Some(
                CategoryProfile::builder()
                    .model_name("priced-model")
                    .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                        usage: Some(Usage {
                            prompt_tokens: Some(1_000),
                            completion_tokens: Some(500),
                            total_tokens: Some(1_500),
                            cache_read_tokens: Some(200),
                            cache_write_tokens: None,
                            uncached_input_tokens: None,
                            cost: None,
                        }),
                        ..empty_annotated_response()
                    }))
                    .build(),
            ),
        );
        let provider_qualified_cost_attributes =
            attr_map(&end_attributes(&provider_qualified_cost_event));
        assert_eq!(
            provider_qualified_cost_attributes.get("nemo_relay.llm.cost.total"),
            Some(&"0.000435".to_string())
        );
        assert_eq!(
            provider_qualified_cost_attributes.get("nemo_relay.llm.cost.currency"),
            Some(&"USD".to_string())
        );
    }

    {
        let _pricing_guard = pricing_test_mutex().lock().unwrap();
        install_test_pricing("priced-model");
        let _reset_guard = ResetPricingResolverGuard;
        let partial_cost_event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
            None,
            "test",
            ScopeType::Llm,
            Some(json!({"answer": "ok"})),
            Some(
                CategoryProfile::builder()
                    .model_name("priced-model")
                    .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                        usage: Some(Usage {
                            prompt_tokens: Some(1_000),
                            completion_tokens: Some(500),
                            total_tokens: Some(1_500),
                            cache_read_tokens: Some(200),
                            cache_write_tokens: Some(10),
                            uncached_input_tokens: None,
                            cost: None,
                        }),
                        ..empty_annotated_response()
                    }))
                    .build(),
            ),
        );
        let partial_cost_attributes = attr_map(&end_attributes(&partial_cost_event));
        assert!(!partial_cost_attributes.contains_key("nemo_relay.llm.cost.total"));
        assert!(!partial_cost_attributes.contains_key("nemo_relay.llm.cost.currency"));
    }
}

fn assert_otel_manual_cost_branches() {
    {
        let _pricing_guard = pricing_test_mutex().lock().unwrap();
        install_test_pricing("priced-model");
        let _reset_guard = ResetPricingResolverGuard;
        let manual_cost_event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
            None,
            "chat",
            ScopeType::Llm,
            Some(json!({
                "model": "priced-model",
                "usage": {
                    "prompt_tokens": 1_000,
                    "completion_tokens": 500,
                    "total_tokens": 1_500,
                    "prompt_tokens_details": {"cached_tokens": 200}
                }
            })),
            None,
        );
        let manual_cost_attributes = attr_map(&end_attributes(&manual_cost_event));
        assert_eq!(
            manual_cost_attributes.get("nemo_relay.llm.cost.total"),
            Some(&"0.000435".to_string())
        );
        assert_eq!(
            manual_cost_attributes.get("nemo_relay.llm.cost.currency"),
            Some(&"USD".to_string())
        );

        let manual_component_cost_event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
            None,
            "chat",
            ScopeType::Llm,
            Some(json!({
                "model": "unknown-model",
                "usage": {
                    "prompt_tokens": 1_000,
                    "completion_tokens": 500,
                    "cost": {
                        "currency": "EUR",
                        "input": 0.25,
                        "output": 0.5,
                        "cache_read": 0.125
                    }
                }
            })),
            None,
        );
        let manual_component_cost_attributes =
            attr_map(&end_attributes(&manual_component_cost_event));
        assert_eq!(
            manual_component_cost_attributes.get("nemo_relay.llm.cost.total"),
            Some(&"0.875".to_string())
        );
        assert_eq!(
            manual_component_cost_attributes.get("nemo_relay.llm.cost.currency"),
            Some(&"EUR".to_string())
        );

        let annotated_without_model_event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
            None,
            "chat",
            ScopeType::Llm,
            Some(json!({
                "model": "priced-model",
                "usage": {
                    "prompt_tokens": 1_000,
                    "completion_tokens": 500,
                    "total_tokens": 1_500,
                    "prompt_tokens_details": {"cached_tokens": 200}
                }
            })),
            Some(
                CategoryProfile::builder()
                    .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                        usage: Some(Usage {
                            prompt_tokens: Some(1_000),
                            completion_tokens: Some(500),
                            total_tokens: Some(1_500),
                            cache_read_tokens: Some(200),
                            ..Usage::default()
                        }),
                        ..empty_annotated_response()
                    }))
                    .build(),
            ),
        );
        let annotated_without_model_attributes =
            attr_map(&end_attributes(&annotated_without_model_event));
        assert_eq!(
            annotated_without_model_attributes.get("nemo_relay.llm.cost.total"),
            Some(&"0.000435".to_string())
        );
        assert_eq!(
            annotated_without_model_attributes.get("nemo_relay.llm.cost.currency"),
            Some(&"USD".to_string())
        );
    }

    let mark = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid(Uuid::now_v7())
            .name("checkpoint")
            .data(json!({"kind": "aux"}))
            .metadata(json!({"source": "unit"}))
            .build(),
        None,
        None,
    ));
    let mark_attributes = attr_map(&mark_attributes(&mark));
    assert_eq!(
        mark_attributes.get("nemo_relay.mark.data.kind"),
        Some(&"aux".to_string())
    );
    assert_eq!(
        mark_attributes.get("nemo_relay.mark.metadata.source"),
        Some(&"unit".to_string())
    );

    let mut processor = OtelEventProcessor::new(make_provider().0, "test".into());
    processor.process(&crate::test_support::event_view(&make_end_event(
        Uuid::now_v7(),
        None,
        "missing",
        ScopeType::Agent,
        None,
    )));
    assert!(processor.active_spans.is_empty());

    let local_context = local_parent_span_context(&SpanContext::empty_context());
    assert!(!local_context.is_remote());

    let whole_second_pre_epoch = DateTime::parse_from_rfc3339("1969-12-31T23:59:58Z")
        .unwrap()
        .with_timezone(&Utc);
    assert_eq!(
        to_system_time(whole_second_pre_epoch),
        UNIX_EPOCH - Duration::from_secs(2)
    );
}

fn assert_otel_normalized_cost_branches() {
    let normalized_cost_event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"answer": "ok"})),
        Some(
            CategoryProfile::builder()
                .model_name("unknown-model")
                .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                    usage: Some(Usage {
                        prompt_tokens: Some(1_000),
                        completion_tokens: Some(500),
                        cost: Some(CostEstimate {
                            total: Some(0.42),
                            currency: "USD".into(),
                            input: None,
                            output: None,
                            cache_read: None,
                            cache_write: None,
                            source: CostSource::ProviderReported,
                            pricing_provider: Some("external".to_string()),
                            pricing_model: Some("external-model".to_string()),
                            pricing_as_of: Some("2026-06-04".to_string()),
                            pricing_source: None,
                        }),
                        ..Usage::default()
                    }),
                    ..empty_annotated_response()
                }))
                .build(),
        ),
    );
    let normalized_cost_attributes = attr_map(&end_attributes(&normalized_cost_event));
    assert_eq!(
        normalized_cost_attributes.get("nemo_relay.llm.cost.total"),
        Some(&"0.42".to_string())
    );
    assert_eq!(
        normalized_cost_attributes.get("nemo_relay.llm.cost.currency"),
        Some(&"USD".to_string())
    );

    {
        let _pricing_guard = pricing_test_mutex().lock().unwrap();
        install_test_pricing("priced-model");
        let _reset_guard = ResetPricingResolverGuard;
        let reported_cost_without_total_event = make_scope_event_with_profile(
            ScopeCategory::End,
            Uuid::now_v7(),
            None,
            "test",
            ScopeType::Llm,
            Some(json!({"answer": "ok"})),
            Some(
                CategoryProfile::builder()
                    .model_name("priced-model")
                    .annotated_response(std::sync::Arc::new(AnnotatedLlmResponse {
                        usage: Some(Usage {
                            prompt_tokens: Some(1_000),
                            completion_tokens: Some(500),
                            cost: Some(CostEstimate {
                                total: None,
                                currency: "EUR".into(),
                                input: Some(0.10),
                                output: None,
                                cache_read: None,
                                cache_write: None,
                                source: CostSource::ProviderReported,
                                pricing_provider: Some("external".to_string()),
                                pricing_model: Some("external-model".to_string()),
                                pricing_as_of: Some("2026-06-04".to_string()),
                                pricing_source: None,
                            }),
                            ..Usage::default()
                        }),
                        ..empty_annotated_response()
                    }))
                    .build(),
            ),
        );
        let reported_cost_without_total_attributes =
            attr_map(&end_attributes(&reported_cost_without_total_event));
        assert_eq!(
            reported_cost_without_total_attributes.get("nemo_relay.llm.cost.total"),
            Some(&"0.1".to_string())
        );
        assert_eq!(
            reported_cost_without_total_attributes.get("nemo_relay.llm.cost.currency"),
            Some(&"EUR".to_string())
        );
    }
}

fn assert_otel_scope_and_model_attribute_branches() {
    let function_end = make_end_event(Uuid::now_v7(), None, "fn-scope", ScopeType::Function, None);
    assert_eq!(span_name(&function_end), "fn-scope");
    assert_eq!(
        semantic_scope_type(&function_end),
        Some(ScopeType::Function)
    );

    assert_eq!(scope_type_name(Some(ScopeType::Retriever)), "retriever");
    assert_eq!(scope_type_name(Some(ScopeType::Embedder)), "embedder");
    assert_eq!(scope_type_name(Some(ScopeType::Reranker)), "reranker");
    assert_eq!(scope_type_name(Some(ScopeType::Guardrail)), "guardrail");
    assert_eq!(scope_type_name(Some(ScopeType::Evaluator)), "evaluator");
    assert_eq!(scope_type_name(Some(ScopeType::Custom)), "custom");
    assert_eq!(scope_type_name(Some(ScopeType::Unknown)), "unknown");
    assert_eq!(scope_type_name(None), "unknown");

    let llm_event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"answer": "ok"})),
        Some(CategoryProfile::builder().model_name("demo-model").build()),
    );
    let llm_attributes = attr_map(&common_attributes(&llm_event));
    assert_eq!(
        llm_attributes.get("nemo_relay.model_name"),
        Some(&"demo-model".to_string())
    );
    let raw_model_event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({"model": "raw-model", "answer": "ok"})),
        None,
    );
    let raw_model_attributes = attr_map(&common_attributes(&raw_model_event));
    assert_eq!(
        raw_model_attributes.get("nemo_relay.model_name"),
        Some(&"raw-model".to_string())
    );
    let response_model_event = make_scope_event_with_profile(
        ScopeCategory::End,
        Uuid::now_v7(),
        None,
        "chat",
        ScopeType::Llm,
        Some(json!({
            "id": "chatcmpl-response-model",
            "model": "response-model",
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }]
        })),
        Some(
            CategoryProfile::builder()
                .model_name("requested-model")
                .build(),
        ),
    );
    let response_model_attributes = attr_map(&common_attributes(&response_model_event));
    assert_eq!(
        response_model_attributes.get("nemo_relay.model_name"),
        Some(&"response-model".to_string())
    );
}

fn assert_otel_tool_attribute_branches() {
    let tool_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("lookup")
            .data(json!({"query": "hello"}))
            .metadata(json!({"meta": true}))
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::tool(),
        Some(CategoryProfile::builder().tool_call_id("call-123").build()),
    ));
    let tool_attributes = attr_map(&common_attributes(&tool_event));
    assert_eq!(
        tool_attributes.get("nemo_relay.tool_call_id"),
        Some(&"call-123".to_string())
    );

    let start_attributes = attr_map(&start_attributes(&tool_event));
    assert_eq!(
        start_attributes.get("nemo_relay.start.data.query"),
        Some(&"hello".to_string())
    );
    assert_eq!(
        start_attributes.get("nemo_relay.start.metadata.meta"),
        Some(&"true".to_string())
    );

    let tool_end_profile = CategoryProfile::builder()
        .tool_call_id("call-456")
        .tool_result_annotation(json!({"opaque": {"rank": 1}}))
        .build();
    let tool_end_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("lookup")
            .metadata(json!({"phase": "complete"}))
            .data(json!({"result": true}))
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::tool(),
        Some(tool_end_profile),
    ));
    let tool_end_attributes = attr_map(&end_attributes(&tool_end_event));
    assert_eq!(
        tool_end_attributes.get("nemo_relay.end.data.result"),
        Some(&"true".to_string())
    );
    assert_eq!(
        tool_end_attributes.get("nemo_relay.tool.result.annotation"),
        Some(&r#"{"opaque":{"rank":1}}"#.to_string())
    );

    let llm_profile = CategoryProfile::builder()
        .tool_result_annotation(json!({"must_not_project": true}))
        .build();
    let llm_end_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("chat").build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::llm(),
        Some(llm_profile),
    ));
    assert!(
        !attr_map(&end_attributes(&llm_end_event))
            .contains_key("nemo_relay.tool.result.annotation")
    );

    let gen_ai_attributes = attr_map(&crate::test_support::otel_genai::end_attributes(
        &tool_end_event,
    ));
    assert!(!gen_ai_attributes.contains_key("nemo_relay.tool.result.annotation"));
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

fn empty_annotated_response() -> AnnotatedLlmResponse {
    AnnotatedLlmResponse {
        id: None,
        model: None,
        message: None,
        tool_calls: None,
        finish_reason: None,
        usage: None,
        optimization_summary: None,
        api_specific: None,
        extra: serde_json::Map::new(),
    }
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

fn install_disambiguation_pricing(model_id: &str, preferred_provider: &str) {
    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [
                {
                    "provider": "other",
                    "model_id": model_id,
                    "pricing_as_of": "2026-06-05",
                    "pricing_source": "test",
                    "rates": {
                        "input_per_million": 1000.0,
                        "output_per_million": 1000.0
                    },
                    "prompt_cache": {
                        "read_accounting": "included_in_prompt_tokens"
                    }
                },
                {
                    "provider": preferred_provider,
                    "model_id": model_id,
                    "pricing_as_of": "2026-06-05",
                    "pricing_source": "test",
                    "rates": {
                        "input_per_million": 0.15,
                        "output_per_million": 0.60,
                        "cache_read_per_million": 0.075
                    },
                    "prompt_cache": {
                        "read_accounting": "included_in_prompt_tokens"
                    }
                }
            ]
        })
        .to_string(),
    )
    .unwrap();
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();
}

fn install_provider_disambiguation_pricing(model_id: &str) {
    install_disambiguation_pricing(model_id, "test");
}

fn install_test_pricing(model_id: &str) {
    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [
                {
                    "provider": "test",
                    "model_id": model_id,
                    "pricing_as_of": "2026-06-05",
                    "pricing_source": "test",
                    "rates": {
                        "input_per_million": 0.15,
                        "output_per_million": 0.60,
                        "cache_read_per_million": 0.075
                    },
                    "prompt_cache": {
                        "read_accounting": "included_in_prompt_tokens"
                    }
                }
            ]
        })
        .to_string(),
    )
    .unwrap();
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();
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

fn make_end_event_with_metadata(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    metadata: Json,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(uuid)
            .name(name)
            .metadata(metadata)
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::from(scope_type),
        None,
    ))
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

fn make_scope_event_with_attributes(
    scope_category: ScopeCategory,
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    data: Option<Json>,
    attributes: Vec<String>,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .uuid(uuid)
            .name(name)
            .data_opt(data)
            .build(),
        scope_category,
        attributes,
        EventCategory::from(scope_type),
        None,
    ))
}

fn make_scope_event_with_profile(
    scope_category: ScopeCategory,
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    scope_type: ScopeType,
    data: Option<Json>,
    category_profile: Option<CategoryProfile>,
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
        category_profile,
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

struct ResetPricingResolverGuard;

impl Drop for ResetPricingResolverGuard {
    fn drop(&mut self) {
        let _ = reset_active_pricing_resolver();
    }
}
