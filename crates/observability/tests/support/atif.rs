// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unused_imports)]
use crate::atif as implementation;
pub(crate) use crate::atif::{AtifAgentInfo, AtifExporter, AtifStepExtra};
pub(crate) use crate::otel::RelayIdGenerator;
pub(crate) use crate::otel_signal::SignalRuntimeDiagnostics;
pub(crate) use crate::{
    MarkProjection, OpenTelemetryType, default_mark_exclude_names, relay_span_id, relay_trace_id,
};
use chrono::{DateTime, Utc};
pub(crate) use implementation::*;
use nemo_relay_types::Json;
use nemo_relay_types::api::event::*;
use nemo_relay_types::api::scope::*;
use nemo_relay_types::api::tool::*;
use nemo_relay_types::codec::request::*;
use nemo_relay_types::codec::response::*;
use opentelemetry::logs::AnyValue;
use opentelemetry::logs::{LogRecord, Logger, LoggerProvider, Severity};
use opentelemetry::metrics::MeterProvider;
use opentelemetry::{
    Key,
    trace::{SpanContext, SpanId, TraceFlags},
};
use opentelemetry::{KeyValue, trace::SpanKind};
use opentelemetry_sdk::{metrics::SdkMeterProvider, trace::SdkTracerProvider};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::UNIX_EPOCH;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

#[allow(dead_code)]
fn observation_extra(event: &Event, output: Option<&Json>, content: Option<&Json>) -> Json {
    implementation::observation_extra(&crate::test_support::event_view(event), output, content)
}

#[allow(dead_code)]
fn llm_start_user_step_message(event: &Event, input: &Json, has_paired_end: bool) -> Option<Json> {
    implementation::llm_start_user_step_message(
        &crate::test_support::event_view(event),
        input,
        has_paired_end,
    )
}

#[allow(dead_code)]
fn event_extra(event: &Event) -> Json {
    implementation::event_extra(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn build_ancestry(
    event: &Event,
    name_map: &std::collections::HashMap<Uuid, String>,
) -> AtifAncestry {
    implementation::build_ancestry(&crate::test_support::event_view(event), name_map)
}

#[allow(dead_code)]
fn build_llm_dedupe(events: &[&Event]) -> LlmDedupeLookups {
    let views_events: Vec<_> = events
        .iter()
        .map(|event| crate::test_support::event_view(event))
        .collect();
    let refs_events: Vec<_> = views_events.iter().collect();
    implementation::build_llm_dedupe(&refs_events)
}

#[allow(dead_code)]
fn collect_llm_span_candidates(events: &[&Event]) -> Vec<LlmSpanCandidate> {
    let views_events: Vec<_> = events
        .iter()
        .map(|event| crate::test_support::event_view(event))
        .collect();
    let refs_events: Vec<_> = views_events.iter().collect();
    implementation::collect_llm_span_candidates(&refs_events)
}

#[allow(dead_code)]
fn llm_request_correlation_keys(start: &Event, end: &Event) -> HashSet<String> {
    implementation::llm_request_correlation_keys(
        &crate::test_support::event_view(start),
        &crate::test_support::event_view(end),
    )
}

#[allow(dead_code)]
fn collect_llm_request_correlation_keys(event: &Event, keys: &mut HashSet<String>) {
    implementation::collect_llm_request_correlation_keys(
        &crate::test_support::event_view(event),
        keys,
    )
}

#[allow(dead_code)]
fn llm_event_fidelity_score(event: &Event) -> u8 {
    implementation::llm_event_fidelity_score(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn is_hook_instrumented_llm_event(event: &Event) -> bool {
    implementation::is_hook_instrumented_llm_event(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn is_gateway_instrumented_llm_event(event: &Event) -> bool {
    implementation::is_gateway_instrumented_llm_event(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn has_non_exact_provider_payload(event: &Event) -> bool {
    implementation::has_non_exact_provider_payload(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn build_tool_call_correlations(events: &[&Event]) -> HashMap<Uuid, String> {
    let views_events: Vec<_> = events
        .iter()
        .map(|event| crate::test_support::event_view(event))
        .collect();
    let refs_events: Vec<_> = views_events.iter().collect();
    implementation::build_tool_call_correlations(&refs_events)
}

#[allow(dead_code)]
fn collect_tool_correlation_inputs(
    events: &[&Event],
) -> (
    HashMap<Uuid, String>,
    Vec<ToolExecutionRecord>,
    Vec<LlmToolCallRecord>,
) {
    let views_events: Vec<_> = events
        .iter()
        .map(|event| crate::test_support::event_view(event))
        .collect();
    let refs_events: Vec<_> = views_events.iter().collect();
    implementation::collect_tool_correlation_inputs(&refs_events)
}

#[allow(dead_code)]
fn collect_tool_correlation_event(
    event: &Event,
    explicit: &mut HashMap<Uuid, String>,
    executions: &mut Vec<ToolExecutionRecord>,
    tool_calls: &mut Vec<LlmToolCallRecord>,
) {
    implementation::collect_tool_correlation_event(
        &crate::test_support::event_view(event),
        explicit,
        executions,
        tool_calls,
    )
}

#[allow(dead_code)]
fn event_signature(
    event: &Event,
) -> (
    &str,
    Option<nemo_relay_types::api::event::ScopeCategory>,
    Option<&str>,
) {
    implementation::event_signature(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn collect_tool_execution_start(
    event: &Event,
    explicit: &mut HashMap<Uuid, String>,
    executions: &mut Vec<ToolExecutionRecord>,
) {
    implementation::collect_tool_execution_start(
        &crate::test_support::event_view(event),
        explicit,
        executions,
    )
}

#[allow(dead_code)]
fn collect_explicit_tool_call_id(event: &Event, explicit: &mut HashMap<Uuid, String>) {
    implementation::collect_explicit_tool_call_id(&crate::test_support::event_view(event), explicit)
}

#[allow(dead_code)]
fn collect_llm_tool_calls(event: &Event, tool_calls: &mut Vec<LlmToolCallRecord>) {
    implementation::collect_llm_tool_calls(&crate::test_support::event_view(event), tool_calls)
}

#[allow(dead_code)]
fn tool_execution_match_key(event: &Event) -> Option<ToolCallMatchKey> {
    implementation::tool_execution_match_key(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn normalized_response_model_name(event: &Event) -> Option<String> {
    implementation::normalized_response_model_name(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn effective_model_for_pair(start: &Event, end: &Event) -> Option<String> {
    implementation::effective_model_for_pair(
        &crate::test_support::event_view(start),
        &crate::test_support::event_view(end),
    )
}

#[allow(dead_code)]
fn effective_response_model_name(event: &Event) -> Option<String> {
    implementation::effective_response_model_name(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn subagent_dispatch_arguments(child: &AgentScopeNode, event: &Event) -> Json {
    implementation::subagent_dispatch_arguments(child, &crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn agent_scope_maps(events: &[&Event]) -> (HashMap<Uuid, Uuid>, AgentScopeRoles) {
    let views_events: Vec<_> = events
        .iter()
        .map(|event| crate::test_support::event_view(event))
        .collect();
    let refs_events: Vec<_> = views_events.iter().collect();
    implementation::agent_scope_maps(&refs_events)
}

#[allow(dead_code)]
fn agent_uuids_from_events(
    events: &[&Event],
    scope_parent_map: &HashMap<Uuid, Uuid>,
    agent_scope_roles: &AgentScopeRoles,
) -> HashSet<Uuid> {
    let views_events: Vec<_> = events
        .iter()
        .map(|event| crate::test_support::event_view(event))
        .collect();
    let refs_events: Vec<_> = views_events.iter().collect();
    implementation::agent_uuids_from_events(&refs_events, scope_parent_map, agent_scope_roles)
}

#[allow(dead_code)]
fn should_include_agent_scope(
    event: &Event,
    scope_parent_map: &HashMap<Uuid, Uuid>,
    agent_scope_roles: &AgentScopeRoles,
) -> bool {
    implementation::should_include_agent_scope(
        &crate::test_support::event_view(event),
        scope_parent_map,
        agent_scope_roles,
    )
}

#[allow(dead_code)]
fn agent_scope_nodes(
    events: &[&Event],
    scope_parent_map: &HashMap<Uuid, Uuid>,
    agent_uuids: &HashSet<Uuid>,
) -> HashMap<Uuid, AgentScopeNode> {
    let views_events: Vec<_> = events
        .iter()
        .map(|event| crate::test_support::event_view(event))
        .collect();
    let refs_events: Vec<_> = views_events.iter().collect();
    implementation::agent_scope_nodes(&refs_events, scope_parent_map, agent_uuids)
}

#[allow(dead_code)]
fn is_included_agent_scope(event: &Event, agent_uuids: &HashSet<Uuid>) -> bool {
    implementation::is_included_agent_scope(&crate::test_support::event_view(event), agent_uuids)
}

#[allow(dead_code)]
fn agent_session_id(event: &Event) -> Option<String> {
    implementation::agent_session_id(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn agent_scope_role(event: &Event) -> Option<&str> {
    implementation::agent_scope_role(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn is_agent_scope_event(event: &Event) -> bool {
    implementation::is_agent_scope_event(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn is_subagent_reference_event(event: &Event) -> bool {
    implementation::is_subagent_reference_event(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn events_to_trajectory(
    session_id: &str,
    trajectory_id: &str,
    agent_info: AtifAgentInfo,
    events: &[&Event],
) -> AtifTrajectory {
    let views_events: Vec<_> = events
        .iter()
        .map(|event| crate::test_support::event_view(event))
        .collect();
    let refs_events: Vec<_> = views_events.iter().collect();
    implementation::events_to_trajectory(session_id, trajectory_id, agent_info, &refs_events)
}

#[allow(dead_code)]
fn can_use_agent_scope_tree(tree: &AgentScopeTree, events: &[&Event]) -> bool {
    let views_events: Vec<_> = events
        .iter()
        .map(|event| crate::test_support::event_view(event))
        .collect();
    let refs_events: Vec<_> = views_events.iter().collect();
    implementation::can_use_agent_scope_tree(tree, &refs_events)
}

#[allow(dead_code)]
fn is_step_event(event: &Event) -> bool {
    implementation::is_step_event(&crate::test_support::event_view(event))
}

#[allow(dead_code)]
fn agent_scope_to_trajectory(
    tree: &AgentScopeTree,
    agent_uuid: Uuid,
    session_id: &str,
    root_trajectory_id: &str,
    agent_info: &AtifAgentInfo,
    sorted_events: &[&Event],
    is_root: bool,
) -> AtifTrajectory {
    let views_sorted_events: Vec<_> = sorted_events
        .iter()
        .map(|event| crate::test_support::event_view(event))
        .collect();
    let refs_sorted_events: Vec<_> = views_sorted_events.iter().collect();
    implementation::agent_scope_to_trajectory(
        tree,
        agent_uuid,
        session_id,
        root_trajectory_id,
        agent_info,
        &refs_sorted_events,
        is_root,
    )
}

#[allow(dead_code)]
fn events_to_steps_for_agent(
    events: &[&Event],
    tree: &AgentScopeTree,
    agent_uuid: Uuid,
) -> Vec<AtifStep> {
    let views_events: Vec<_> = events
        .iter()
        .map(|event| crate::test_support::event_view(event))
        .collect();
    let refs_events: Vec<_> = views_events.iter().collect();
    implementation::events_to_steps_for_agent(&refs_events, tree, agent_uuid)
}

#[allow(dead_code)]
fn events_to_steps(events: &[&Event]) -> Vec<AtifStep> {
    let views_events: Vec<_> = events
        .iter()
        .map(|event| crate::test_support::event_view(event))
        .collect();
    let refs_events: Vec<_> = views_events.iter().collect();
    implementation::events_to_steps(&refs_events)
}

#[allow(dead_code)]
fn is_start_event(event: &Event) -> bool {
    implementation::is_start_event(&crate::test_support::event_view(event))
}

#[path = "../unit/atif_projection_tests.rs"]
mod projection_tests;

pub(crate) fn step_tool_call_ids(step: &AtifStep) -> Vec<String> {
    step.tool_calls
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|tool_call| tool_call.tool_call_id.clone())
        .filter(|tool_call_id| !tool_call_id.is_empty())
        .collect()
}
