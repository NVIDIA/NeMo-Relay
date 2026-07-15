// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-exporter parity tests for the NeMo Relay core crate.
//!
//! Each test feeds the same event stream to the ATIF, OpenTelemetry, and
//! OpenInference exporters and asserts that the facts they project (cost,
//! usage, model name, tool calls) agree. Where an exporter projects more or
//! less than the others, the divergence is asserted explicitly so drift from
//! the documented behavior fails loudly instead of silently.

use std::collections::HashMap;

use opentelemetry::KeyValue;
use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, SdkTracerProvider, SpanData};
use serde_json::json;
use uuid::Uuid;

use crate::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, EventNormalizationExt, ScopeCategory,
    ScopeEvent,
};
use crate::codec::model_pricing::pricing_test_mutex;
use crate::codec::response::{
    PricingCatalog, PricingResolver, reset_active_pricing_resolver, set_active_pricing_resolver,
};
use crate::json::Json;
use crate::observability::atif::{
    AtifAgentInfo, AtifExporter, AtifStep, AtifStepExtra, AtifTrajectory,
};
use crate::observability::openinference::OpenInferenceSubscriber;
use crate::observability::otel::OpenTelemetrySubscriber;

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

struct ResetPricingResolverGuard;

impl Drop for ResetPricingResolverGuard {
    fn drop(&mut self) {
        let _ = reset_active_pricing_resolver();
    }
}

fn install_parity_pricing(model_id: &str) {
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

fn make_agent_info() -> AtifAgentInfo {
    AtifAgentInfo {
        name: "parity-agent".to_string(),
        version: "1.0.0".to_string(),
        model_name: None,
        tool_definitions: None,
        extra: None,
    }
}

fn make_provider() -> (
    SdkTracerProvider,
    opentelemetry_sdk::trace::InMemorySpanExporter,
) {
    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    (provider, exporter)
}

fn attr_map(attributes: &[KeyValue]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for attribute in attributes {
        let key = attribute.key.as_str().to_string();
        let previous = map.insert(key.clone(), attribute.value.to_string());
        assert!(previous.is_none(), "duplicate span attribute key: {key}");
    }
    map
}

fn assert_no_attribute_key_contains(attributes: &HashMap<String, String>, fragment: &str) {
    let offending: Vec<&String> = attributes
        .keys()
        .filter(|key| key.contains(fragment))
        .collect();
    assert!(
        offending.is_empty(),
        "expected no attribute key containing {fragment:?}, found {offending:?}",
    );
}

fn make_scope_event(
    scope_category: ScopeCategory,
    uuid: Uuid,
    name: &str,
    category: EventCategory,
    data: Option<Json>,
    profile: Option<CategoryProfile>,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(uuid)
            .name(name)
            .data_opt(data)
            .build(),
        scope_category,
        Vec::new(),
        category,
        profile,
    ))
}

/// LLM start event carrying the managed-pipeline `LlmRequest` envelope.
fn llm_start(uuid: Uuid, name: &str, request_content: Json) -> Event {
    make_scope_event(
        ScopeCategory::Start,
        uuid,
        name,
        EventCategory::llm(),
        Some(json!({"headers": {}, "content": request_content})),
        None,
    )
}

fn llm_end(uuid: Uuid, name: &str, output: Json) -> Event {
    make_scope_event(
        ScopeCategory::End,
        uuid,
        name,
        EventCategory::llm(),
        Some(output),
        None,
    )
}

fn llm_event_with_model(
    scope_category: ScopeCategory,
    uuid: Uuid,
    name: &str,
    data: Json,
    model_name: &str,
) -> Event {
    make_scope_event(
        scope_category,
        uuid,
        name,
        EventCategory::llm(),
        Some(data),
        Some(CategoryProfile::builder().model_name(model_name).build()),
    )
}

fn tool_event(
    scope_category: ScopeCategory,
    uuid: Uuid,
    name: &str,
    data: Json,
    tool_call_id: &str,
) -> Event {
    make_scope_event(
        scope_category,
        uuid,
        name,
        EventCategory::tool(),
        Some(data),
        Some(
            CategoryProfile::builder()
                .tool_call_id(tool_call_id)
                .build(),
        ),
    )
}

struct ParityExports {
    trajectory: AtifTrajectory,
    otel_spans: Vec<SpanData>,
    openinference_spans: Vec<SpanData>,
}

impl ParityExports {
    fn otel_attrs(&self, span_name: &str) -> HashMap<String, String> {
        span_attrs(&self.otel_spans, span_name)
    }

    fn openinference_attrs(&self, span_name: &str) -> HashMap<String, String> {
        span_attrs(&self.openinference_spans, span_name)
    }

    /// The ATIF `agent` step projected from the LLM end event.
    fn agent_step(&self) -> &AtifStep {
        self.trajectory
            .steps
            .iter()
            .find(|step| step.source == "agent")
            .expect("trajectory should contain an agent step")
    }
}

fn span_attrs(spans: &[SpanData], span_name: &str) -> HashMap<String, String> {
    let span = spans
        .iter()
        .find(|span| span.name.as_ref() == span_name)
        .unwrap_or_else(|| panic!("missing span {span_name}"));
    attr_map(&span.attributes)
}

/// Feed the same events to all three exporters and collect their outputs.
fn export_through_all_exporters(events: &[Event]) -> ParityExports {
    let atif = AtifExporter::new("parity-session".to_string(), make_agent_info());
    let atif_record = atif.subscriber();

    let (otel_provider, otel_exporter) = make_provider();
    let otel = OpenTelemetrySubscriber::from_tracer_provider(otel_provider, "parity-otel");
    let otel_record = otel.subscriber();

    let (openinference_provider, openinference_exporter) = make_provider();
    let openinference = OpenInferenceSubscriber::from_tracer_provider(
        openinference_provider,
        "parity-openinference",
    );
    let openinference_record = openinference.subscriber();

    for event in events {
        atif_record(event);
        otel_record(event);
        openinference_record(event);
    }

    let trajectory = atif.export().unwrap();
    otel.force_flush().unwrap();
    openinference.force_flush().unwrap();

    ParityExports {
        trajectory,
        otel_spans: otel_exporter.get_finished_spans().unwrap(),
        openinference_spans: openinference_exporter.get_finished_spans().unwrap(),
    }
}

fn run_llm_scenario(request_content: Json, output: Json) -> ParityExports {
    let uuid = Uuid::now_v7();
    export_through_all_exporters(&[
        llm_start(uuid, "model-call", request_content),
        llm_end(uuid, "model-call", output),
    ])
}

// Shared provider-shaped payloads: the same logical call (one user message,
// one assistant reply, 1000/500/200-cached usage) in two provider schemas so
// codec x exporter drift is caught.

fn chat_request_content(model: &str) -> Json {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": "price this"}]
    })
}

fn chat_response_output(model: &str) -> Json {
    json!({
        "id": "chatcmpl-parity",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
            "prompt_tokens_details": {"cached_tokens": 200}
        }
    })
}

fn anthropic_request_content(model: &str) -> Json {
    json!({
        "model": model,
        "system": "be terse",
        "messages": [{"role": "user", "content": "price this"}]
    })
}

fn anthropic_response_output(model: &str, usage: Json) -> Json {
    json!({
        "id": "msg_parity",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": usage
    })
}

// ===================================================================
// Cost parity
// ===================================================================

#[test]
fn test_exporters_agree_on_cost_total_for_openai_chat_payload() {
    let _pricing_guard = pricing_test_mutex()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    install_parity_pricing("parity-priced-model");
    let _reset_guard = ResetPricingResolverGuard;

    let exports = run_llm_scenario(
        chat_request_content("parity-priced-model"),
        chat_response_output("parity-priced-model"),
    );

    // 800 billable prompt (1000 - 200 cached) * 0.15/M
    //   + 500 completion * 0.60/M + 200 cache-read * 0.075/M = 0.000435 USD.
    assert_eq!(
        exports.agent_step().metrics.as_ref().unwrap().cost_usd,
        Some(0.000_435)
    );

    let otel = exports.otel_attrs("model-call");
    assert_eq!(
        otel.get("nemo_relay.llm.cost.total"),
        Some(&"0.000435".to_string())
    );
    assert_eq!(
        otel.get("nemo_relay.llm.cost.currency"),
        Some(&"USD".to_string())
    );

    let openinference = exports.openinference_attrs("model-call");
    assert_eq!(
        openinference.get("llm.cost.total"),
        Some(&"0.000435".to_string())
    );
}

#[test]
fn test_exporters_agree_on_cost_total_for_anthropic_payload() {
    let _pricing_guard = pricing_test_mutex()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    install_parity_pricing("parity-priced-model");
    let _reset_guard = ResetPricingResolverGuard;

    let exports = run_llm_scenario(
        anthropic_request_content("parity-priced-model"),
        anthropic_response_output(
            "parity-priced-model",
            json!({
                "input_tokens": 1000,
                "output_tokens": 500,
                "cache_read_input_tokens": 200
            }),
        ),
    );

    // Identical normalized usage as the OpenAI Chat scenario, so the cost
    // must match the chat-shaped run exactly: codec differences must not
    // leak into exporter cost facts.
    assert_eq!(
        exports.agent_step().metrics.as_ref().unwrap().cost_usd,
        Some(0.000_435)
    );

    let otel = exports.otel_attrs("model-call");
    assert_eq!(
        otel.get("nemo_relay.llm.cost.total"),
        Some(&"0.000435".to_string())
    );
    assert_eq!(
        otel.get("nemo_relay.llm.cost.currency"),
        Some(&"USD".to_string())
    );

    let openinference = exports.openinference_attrs("model-call");
    assert_eq!(
        openinference.get("llm.cost.total"),
        Some(&"0.000435".to_string())
    );
}

// ===================================================================
// Usage parity
// ===================================================================

#[test]
fn test_usage_facts_parity_for_openai_chat_payload() {
    // "parity-usage-model" never appears in a pricing catalog, so usage facts
    // are deterministic regardless of the process-wide pricing resolver.
    let exports = run_llm_scenario(
        chat_request_content("parity-usage-model"),
        chat_response_output("parity-usage-model"),
    );

    let metrics = exports.agent_step().metrics.as_ref().unwrap();
    assert_eq!(metrics.prompt_tokens, Some(1000));
    assert_eq!(metrics.completion_tokens, Some(500));
    assert_eq!(metrics.cached_tokens, Some(200));

    let openinference = exports.openinference_attrs("model-call");
    assert_eq!(
        openinference.get("llm.token_count.prompt"),
        Some(&"1000".to_string())
    );
    assert_eq!(
        openinference.get("llm.token_count.completion"),
        Some(&"500".to_string())
    );
    assert_eq!(
        openinference.get("llm.token_count.total"),
        Some(&"1500".to_string())
    );
    assert_eq!(
        openinference.get("llm.token_count.prompt_details.cache_read"),
        Some(&"200".to_string())
    );

    let otel = exports.otel_attrs("model-call");
    assert_no_attribute_key_contains(&otel, "token_count");
    assert!(
        otel.keys()
            .any(|key| key.starts_with("nemo_relay.end.output."))
    );
}

#[test]
fn test_usage_facts_parity_for_anthropic_payload_with_cache_write() {
    let exports = run_llm_scenario(
        anthropic_request_content("parity-usage-model"),
        anthropic_response_output(
            "parity-usage-model",
            json!({
                "input_tokens": 1000,
                "output_tokens": 500,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 64
            }),
        ),
    );

    let metrics = exports.agent_step().metrics.as_ref().unwrap();
    assert_eq!(metrics.prompt_tokens, Some(1000));
    assert_eq!(metrics.completion_tokens, Some(500));
    // ATIF folds cache reads and writes into a single cached_tokens sum
    // (200 + 64).
    assert_eq!(metrics.cached_tokens, Some(264));

    let openinference = exports.openinference_attrs("model-call");
    assert_eq!(
        openinference.get("llm.token_count.prompt"),
        Some(&"1000".to_string())
    );
    assert_eq!(
        openinference.get("llm.token_count.completion"),
        Some(&"500".to_string())
    );
    // Anthropic reports no total; the shared codec computes 1000 + 500.
    assert_eq!(
        openinference.get("llm.token_count.total"),
        Some(&"1500".to_string())
    );
    assert_eq!(
        openinference.get("llm.token_count.prompt_details.cache_read"),
        Some(&"200".to_string())
    );
    assert_eq!(
        openinference.get("llm.token_count.prompt_details.cache_write"),
        Some(&"64".to_string())
    );

    let otel = exports.otel_attrs("model-call");
    assert_no_attribute_key_contains(&otel, "token_count");
}

// ===================================================================
// Model-name parity
// ===================================================================

#[test]
fn test_exporters_agree_on_model_name() {
    let exports = run_llm_scenario(
        chat_request_content("parity-name-model"),
        chat_response_output("parity-name-model"),
    );

    assert_eq!(
        exports.agent_step().model_name.as_deref(),
        Some("parity-name-model")
    );
    assert_eq!(
        exports
            .otel_attrs("model-call")
            .get("nemo_relay.model_name"),
        Some(&"parity-name-model".to_string())
    );
    assert_eq!(
        exports
            .openinference_attrs("model-call")
            .get("llm.model_name"),
        Some(&"parity-name-model".to_string())
    );
}

// ===================================================================
// Tool-call parity
// ===================================================================

#[test]
fn test_tool_call_projection_parity() {
    let llm_uuid = Uuid::now_v7();
    let tool_uuid = Uuid::now_v7();
    let output = json!({
        "id": "chatcmpl-parity-tools",
        "model": "parity-tool-model",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_parity_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });

    let exports = export_through_all_exporters(&[
        llm_start(
            llm_uuid,
            "model-call",
            chat_request_content("parity-tool-model"),
        ),
        llm_end(llm_uuid, "model-call", output),
        tool_event(
            ScopeCategory::Start,
            tool_uuid,
            "get_weather",
            json!({"city": "NYC"}),
            "call_parity_1",
        ),
        tool_event(
            ScopeCategory::End,
            tool_uuid,
            "get_weather",
            json!({"temp_f": 72}),
            "call_parity_1",
        ),
    ]);

    // ATIF promotes the tool call onto the agent step and correlates the tool
    // result observation by the same id.
    let agent_step = exports.agent_step();
    let tool_calls = agent_step.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].tool_call_id, "call_parity_1");
    assert_eq!(tool_calls[0].function_name, "get_weather");
    assert_eq!(tool_calls[0].arguments, json!({"city": "NYC"}));
    let observation = agent_step.observation.as_ref().unwrap();
    assert_eq!(
        observation.results[0].source_call_id.as_deref(),
        Some("call_parity_1")
    );

    // OpenInference flattens the same facts onto the LLM span.
    let openinference_llm = exports.openinference_attrs("model-call");
    assert_eq!(
        openinference_llm.get("llm.output_messages.0.message.tool_calls.0.tool_call.id"),
        Some(&"call_parity_1".to_string())
    );
    assert_eq!(
        openinference_llm.get("llm.output_messages.0.message.tool_calls.0.tool_call.function.name"),
        Some(&"get_weather".to_string())
    );
    let openinference_arguments = openinference_llm
        .get("llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments")
        .expect("openinference tool-call arguments");
    assert_eq!(
        serde_json::from_str::<Json>(openinference_arguments).unwrap(),
        tool_calls[0].arguments,
    );
    // The tool span carries the same correlation id.
    assert_eq!(
        exports
            .openinference_attrs("get_weather")
            .get("tool_call.id"),
        Some(&"call_parity_1".to_string())
    );

    assert_eq!(
        exports
            .otel_attrs("get_weather")
            .get("nemo_relay.tool_call_id"),
        Some(&"call_parity_1".to_string())
    );
    // OpenTelemetry retains the raw response as JSON; it does not add
    // OpenInference's semantic tool-call attributes.
    assert_no_attribute_key_contains(&exports.otel_attrs("model-call"), "tool_call");
}

// ===================================================================
// Reasoning projection
// ===================================================================

#[test]
fn test_reasoning_projected_by_atif_only() {
    let request_content = json!({
        "model": "parity-reasoning-model",
        "reasoning_effort": "high",
        "messages": [{"role": "user", "content": "think hard"}]
    });
    let mut output = chat_response_output("parity-reasoning-model");
    output.as_object_mut().unwrap().insert(
        "reasoning".into(),
        Json::String("I compared both options step by step.".to_string()),
    );

    let exports = run_llm_scenario(request_content, output);

    // ATIF projects reasoning effort (from the request) and reasoning content
    // (from the response) onto the agent step.
    let agent_step = exports.agent_step();
    assert_eq!(agent_step.reasoning_effort, Some(json!("high")));
    assert_eq!(
        agent_step.reasoning_content.as_deref(),
        Some("I compared both options step by step.")
    );

    // The raw end data retains reasoning without adding exporter-specific
    // reasoning semantic conventions.
    assert_eq!(
        exports
            .otel_attrs("model-call")
            .get("nemo_relay.end.data.reasoning"),
        Some(&"I compared both options step by step.".to_string())
    );
    assert_eq!(
        exports
            .openinference_attrs("model-call")
            .get("nemo_relay.end.data.reasoning"),
        Some(&"I compared both options step by step.".to_string())
    );
}

// ===================================================================
// Replay-payload preservation
// ===================================================================

#[test]
fn test_replay_payload_preservation_across_exporters() {
    let request_content = chat_request_content("parity-replay-model");
    let output = chat_response_output("parity-replay-model");
    let exports = run_llm_scenario(request_content.clone(), output.clone());

    // ATIF preserves the full request on the user step and the full response
    // on the agent step; both round-trip the original payloads exactly.
    let user_step = exports
        .trajectory
        .steps
        .iter()
        .find(|step| step.source == "user")
        .expect("user step");
    let user_extra: AtifStepExtra =
        serde_json::from_value(user_step.extra.clone().unwrap()).unwrap();
    assert_eq!(user_extra.llm_request, Some(request_content.clone()));

    let agent_extra: AtifStepExtra =
        serde_json::from_value(exports.agent_step().extra.clone().unwrap()).unwrap();
    assert_eq!(agent_extra.llm_response, Some(output.clone()));

    // OTel projects the LLM request wrapper into typed top-level attributes,
    // keeping the provider payload in its nested content field.
    let otel = exports.otel_attrs("model-call");
    let otel_input: Json = serde_json::from_str(
        otel.get("nemo_relay.start.input.content")
            .expect("projected request content"),
    )
    .unwrap();
    assert_eq!(otel_input, request_content);
    assert_eq!(
        otel.get("nemo_relay.start.input.headers"),
        Some(&"{}".to_string())
    );
    assert!(!otel.contains_key("nemo_relay.start.input_json"));
    assert!(
        otel.keys()
            .any(|key| key.starts_with("nemo_relay.end.output."))
    );

    // OpenInference preserves the raw response payload but omits raw
    // LLM request attributes in favor of flattened message fields plus a
    // display input.value.
    let openinference = exports.openinference_attrs("model-call");
    assert!(
        openinference
            .keys()
            .any(|key| key.starts_with("nemo_relay.end.output."))
    );
    assert!(!openinference.contains_key("nemo_relay.start.input.content"));
    assert!(openinference.contains_key("llm.input_messages.0.message.content"));
}

// ===================================================================
// Manual-extraction fallback parity
// ===================================================================

#[test]
fn test_manual_fallback_payload_parity() {
    let _pricing_guard = pricing_test_mutex()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    install_parity_pricing("parity-manual-model");
    let _reset_guard = ResetPricingResolverGuard;

    // A non-provider-shaped payload: no codec detects it, so every exporter
    // must agree via the shared manual-extraction fallback.
    let uuid = Uuid::now_v7();
    let output = json!({
        "content": "manual answer",
        "model": "parity-manual-model",
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500
        }
    });
    let end = llm_event_with_model(
        ScopeCategory::End,
        uuid,
        "model-call",
        output,
        "parity-manual-model",
    );
    assert!(
        end.normalized_llm_response().is_none(),
        "payload must exercise the manual fallback, not a codec",
    );
    let start = llm_event_with_model(
        ScopeCategory::Start,
        uuid,
        "model-call",
        json!({"prompt": "manual prompt"}),
        "parity-manual-model",
    );
    let exports = export_through_all_exporters(&[start, end]);

    // Cost: 1000 * 0.15/M + 500 * 0.60/M = 0.00045 USD from all exporters.
    let metrics = exports.agent_step().metrics.as_ref().unwrap();
    assert_eq!(metrics.prompt_tokens, Some(1000));
    assert_eq!(metrics.completion_tokens, Some(500));
    assert_eq!(metrics.cost_usd, Some(0.000_45));
    assert_eq!(
        exports.agent_step().model_name.as_deref(),
        Some("parity-manual-model")
    );

    let otel = exports.otel_attrs("model-call");
    assert_eq!(
        otel.get("nemo_relay.llm.cost.total"),
        Some(&"0.00045".to_string())
    );
    assert_eq!(
        otel.get("nemo_relay.llm.cost.currency"),
        Some(&"USD".to_string())
    );
    assert_eq!(
        otel.get("nemo_relay.model_name"),
        Some(&"parity-manual-model".to_string())
    );

    let openinference = exports.openinference_attrs("model-call");
    assert_eq!(
        openinference.get("llm.cost.total"),
        Some(&"0.00045".to_string())
    );
    assert_eq!(
        openinference.get("llm.model_name"),
        Some(&"parity-manual-model".to_string())
    );
    assert_eq!(
        openinference.get("llm.token_count.prompt"),
        Some(&"1000".to_string())
    );
    assert_eq!(
        openinference.get("llm.token_count.completion"),
        Some(&"500".to_string())
    );
    assert_eq!(
        openinference.get("llm.token_count.total"),
        Some(&"1500".to_string())
    );
}

#[test]
fn session_instance_correlates_distinct_otel_and_openinference_traces() {
    let uuid = Uuid::now_v7();
    let metadata = json!({"session_id": "logical-session", "user_id": "alice"});
    let start = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(uuid)
            .name("session-root")
            .metadata(metadata)
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::agent(),
        None,
    ));
    let end = make_scope_event(
        ScopeCategory::End,
        uuid,
        "session-root",
        EventCategory::agent(),
        None,
        None,
    );
    let exports = export_through_all_exporters(&[start, end]);
    let otel = exports
        .otel_spans
        .iter()
        .find(|span| span.name.as_ref() == "session-root")
        .unwrap();
    let openinference = exports
        .openinference_spans
        .iter()
        .find(|span| span.name.as_ref() == "session-root")
        .unwrap();
    let otel_attributes = attr_map(&otel.attributes);
    let openinference_attributes = attr_map(&openinference.attributes);

    assert_eq!(otel_attributes["nemo_relay.uuid"], uuid.to_string());
    assert_eq!(
        openinference_attributes["nemo_relay.uuid"],
        uuid.to_string()
    );
    assert_eq!(otel_attributes["session.id"], "logical-session");
    assert_eq!(openinference_attributes["session.id"], "logical-session");
    assert_eq!(otel_attributes["user.id"], "alice");
    assert_eq!(openinference_attributes["user.id"], "alice");
    assert_eq!(
        otel_attributes["nemo_relay.session.instance_id"],
        openinference_attributes["nemo_relay.session.instance_id"]
    );
    assert_ne!(
        otel.span_context.trace_id(),
        openinference.span_context.trace_id()
    );
}
