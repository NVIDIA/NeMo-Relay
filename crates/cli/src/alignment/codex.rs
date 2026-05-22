// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Codex-specific trace alignment and gateway normalization.
//!
//! Codex subagents run as child threads. Their hook stream can look like an independent top-level
//! session unless we recover the parent thread id from Codex metadata or the transcript's first
//! `session_meta` record. The helpers here turn that Codex-specific shape into the generic
//! [`SessionAlias`](super::SessionAlias) contract used by the session manager.

use std::io::{BufRead, BufReader};

use axum::http::HeaderMap;
use serde_json::{Map, Value, json};

use crate::alignment::{
    GatewayRouteKind, SessionAlias, insert_optional, json_string_at, merge_metadata,
};
use crate::model::{AgentKind, SessionEvent, SubagentEvent};

// ChatGPT backend base URL used by Codex when authenticated with ChatGPT-Plus OAuth. This mirrors
// Codex's own `CHATGPT_CODEX_BASE_URL`; API-key auth continues through the normal OpenAI base.
const CHATGPT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";

// The Codex fields needed to turn a child thread into a parent-owned subagent scope. Optional
// display fields are copied through because they make Phoenix traces easier to inspect, but only
// the parent session id is required for correlation.
#[derive(Debug, Clone)]
pub(crate) struct SubagentContext {
    pub(crate) parent_session_id: String,
    nickname: Option<String>,
    role: Option<String>,
    depth: Option<String>,
}

// Identifies gateway LLM providers that Codex owns for implicit session creation. Today Codex
// emits OpenAI Responses requests through the gateway, while OpenAI chat/model endpoints may come
// from generic clients and should not be labeled as Codex by route alone.
pub(crate) fn owns_gateway_provider(provider: &str) -> bool {
    provider == "openai.responses"
}

// Codex currently does not forward a stable session header on OpenAI Responses requests. When the
// request carries Codex client metadata, the `prompt_cache_key` is the rollout/thread id. The
// metadata check prevents treating arbitrary application prompt-cache keys as session ids.
pub(crate) fn prompt_cache_session_id(body: &Value, route: GatewayRouteKind) -> Option<String> {
    if route != GatewayRouteKind::OpenAiResponses {
        return None;
    }
    let has_codex_metadata = body
        .get("client_metadata")
        .and_then(|metadata| metadata.get("x-codex-installation-id"))
        .and_then(Value::as_str)
        .is_some_and(|value| !value.is_empty());
    if !has_codex_metadata {
        return None;
    }
    json_string_at(body, &[&["prompt_cache_key"][..]])
}

// Gives the gateway a Codex-native upstream only when the inbound token is the ChatGPT OAuth JWT
// shape and no `OPENAI_API_KEY` is available to substitute. The gateway stays generic by asking
// alignment for an optional override instead of knowing Codex backend URLs.
pub(crate) fn chatgpt_upstream_url_if_needed(
    headers: &HeaderMap,
    route: GatewayRouteKind,
    path_and_query: &str,
    has_replacement_key: bool,
) -> Option<String> {
    (is_openai_route(route) && has_chatgpt_oauth_jwt(headers) && !has_replacement_key)
        .then(|| chatgpt_upstream_url(path_and_query))
}

// Removes Codex ChatGPT OAuth JWTs from OpenAI-family routes when the gateway has a real API key
// to inject. Real provider keys and non-OpenAI routes are preserved.
pub(crate) fn strip_chatgpt_oauth_for_openai_route(
    headers: &HeaderMap,
    route: GatewayRouteKind,
    has_replacement_key: bool,
) -> HeaderMap {
    if !is_openai_route(route) || !has_replacement_key {
        return headers.clone();
    }
    let mut out = headers.clone();
    if has_chatgpt_oauth_jwt(&out) {
        out.remove(http::header::AUTHORIZATION);
    }
    out
}

// OpenAI API bases commonly include `/v1`, while the ChatGPT backend is rooted at
// `/backend-api/codex`. Strip any `/v1` prefix from gateway routes before appending the path.
fn chatgpt_upstream_url(path_and_query: &str) -> String {
    let path = path_and_query.strip_prefix("/v1").unwrap_or(path_and_query);
    format!("{CHATGPT_CODEX_BASE_URL}{path}")
}

// Codex stores ChatGPT-Plus OAuth tokens in `~/.codex/auth.json`; the access token is a JWT whose
// bearer value starts with `eyJ`. Provider API keys (`sk-...`) and opaque tokens do not match.
fn has_chatgpt_oauth_jwt(headers: &HeaderMap) -> bool {
    headers
        .get(http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("Bearer eyJ"))
}

// The ChatGPT OAuth transport fallback applies only to OpenAI-family routes. Anthropic routes use
// a different auth scheme and should never be redirected through Codex's ChatGPT backend.
fn is_openai_route(route: GatewayRouteKind) -> bool {
    matches!(
        route,
        GatewayRouteKind::OpenAiResponses
            | GatewayRouteKind::OpenAiChatCompletions
            | GatewayRouteKind::OpenAiModels
    )
}

// Extracts Codex subagent thread-spawn context from a SessionStart. It looks first at the hook
// payload, then hook metadata, then the first transcript line. Returning None keeps ordinary Codex
// root sessions as root sessions and prevents self-parenting loops.
pub(crate) async fn subagent_context(event: &SessionEvent) -> Option<SubagentContext> {
    if event.agent_kind != AgentKind::Codex {
        return None;
    }
    let context = match subagent_context_from_value(&event.payload)
        .or_else(|| subagent_context_from_value(&event.metadata))
    {
        Some(context) => Some(context),
        None => subagent_context_from_transcript(event).await,
    };
    context.filter(|context| context.parent_session_id != event.session_id)
}

// Codex sometimes supplies only a transcript path in the hook payload. The first transcript line
// is a `session_meta` object and carries the thread-spawn parent id that Phoenix needs for
// parentage. Reading one line keeps this cheap and avoids treating the full transcript as input.
async fn subagent_context_from_transcript(event: &SessionEvent) -> Option<SubagentContext> {
    let transcript_path = json_string_at(&event.metadata, &[&["transcript_path"][..]])
        .or_else(|| json_string_at(&event.payload, &[&["transcript_path"][..]]))?;
    tokio::task::spawn_blocking(move || {
        let file = std::fs::File::open(transcript_path).ok()?;
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            return None;
        }
        let value = serde_json::from_str::<Value>(&line).ok()?;
        subagent_context_from_value(&value)
    })
    .await
    .ok()
    .flatten()
}

// Searches the known Codex shapes for thread-spawn data. Recent traces have placed
// `parent_thread_id`, nickname, role, and depth under direct hook payloads, nested `payload`, and
// transcript `session_meta.payload`; all of those paths describe the same child-thread parentage.
fn subagent_context_from_value(value: &Value) -> Option<SubagentContext> {
    let parent_session_id = json_string_at(
        value,
        &[
            &["source", "subagent", "thread_spawn", "parent_thread_id"][..],
            &[
                "payload",
                "source",
                "subagent",
                "thread_spawn",
                "parent_thread_id",
            ][..],
            &[
                "session_meta",
                "payload",
                "source",
                "subagent",
                "thread_spawn",
                "parent_thread_id",
            ][..],
            &[
                "extra",
                "source",
                "subagent",
                "thread_spawn",
                "parent_thread_id",
            ][..],
        ],
    )?;
    Some(SubagentContext {
        parent_session_id,
        nickname: json_string_at(
            value,
            &[
                &["agent_nickname"][..],
                &["payload", "agent_nickname"][..],
                &["session_meta", "payload", "agent_nickname"][..],
                &["source", "subagent", "thread_spawn", "agent_nickname"][..],
                &[
                    "payload",
                    "source",
                    "subagent",
                    "thread_spawn",
                    "agent_nickname",
                ][..],
                &[
                    "session_meta",
                    "payload",
                    "source",
                    "subagent",
                    "thread_spawn",
                    "agent_nickname",
                ][..],
            ],
        ),
        role: json_string_at(
            value,
            &[
                &["agent_role"][..],
                &["payload", "agent_role"][..],
                &["session_meta", "payload", "agent_role"][..],
                &["source", "subagent", "thread_spawn", "agent_role"][..],
                &[
                    "payload",
                    "source",
                    "subagent",
                    "thread_spawn",
                    "agent_role",
                ][..],
                &[
                    "session_meta",
                    "payload",
                    "source",
                    "subagent",
                    "thread_spawn",
                    "agent_role",
                ][..],
            ],
        ),
        depth: json_string_at(
            value,
            &[
                &["source", "subagent", "thread_spawn", "depth"][..],
                &["payload", "source", "subagent", "thread_spawn", "depth"][..],
                &[
                    "session_meta",
                    "payload",
                    "source",
                    "subagent",
                    "thread_spawn",
                    "depth",
                ][..],
            ],
        ),
    })
}

// Stamps the child thread's hook metadata with parent-thread details before the session manager
// converts it into a subagent start. This makes the scope itself filterable even before any LLM
// ownership heuristics run.
pub(crate) fn augment_subagent_metadata(metadata: Value, context: &SubagentContext) -> Value {
    let mut object = Map::new();
    object.insert("thread_source".into(), json!("subagent"));
    object.insert(
        "codex_parent_thread_id".into(),
        json!(context.parent_session_id.clone()),
    );
    insert_optional(&mut object, "agent_nickname", context.nickname.as_deref());
    insert_optional(&mut object, "agent_role", context.role.as_deref());
    insert_optional(
        &mut object,
        "codex_subagent_depth",
        context.depth.as_deref(),
    );
    merge_metadata(metadata, Value::Object(object))
}

// Converts the child thread SessionStart into a real subagent start under the parent thread. The
// child session id becomes the subagent id because subsequent gateway/tool events can use that id
// for deterministic ownership.
pub(crate) fn subagent_start_event(
    event: &SessionEvent,
    context: &SubagentContext,
) -> SubagentEvent {
    SubagentEvent {
        session_id: context.parent_session_id.clone(),
        agent_kind: event.agent_kind,
        event_name: event.event_name.clone(),
        subagent_id: event.session_id.clone(),
        payload: event.payload.clone(),
        metadata: merge_metadata(
            event.metadata.clone(),
            json!({ "codex_subagent_session_id": event.session_id.clone() }),
        ),
    }
}

// Creates the routing alias that keeps later child-thread events under the same parent subagent.
// The alias metadata intentionally repeats parent and child thread ids so every rewritten event can
// be debugged without consulting session-manager state.
pub(crate) fn alias_for_child_session(
    child_session_id: String,
    context: &SubagentContext,
) -> SessionAlias {
    SessionAlias::new(
        context.parent_session_id.clone(),
        child_session_id.clone(),
        json!({
            "thread_source": "subagent",
            "codex_parent_thread_id": context.parent_session_id.clone(),
            "codex_subagent_session_id": child_session_id,
        }),
    )
}

// Copies Codex thread fields from the subagent scope onto LLM spans whenever ownership resolves to
// that subagent. The owner may have been explicit, sticky, inferred from a recent tool, or hinted;
// using the scope metadata gives every path the same debug/filter fields. Local transcript paths
// stay on the subagent scope only; copying them onto every LLM span would broaden filesystem path
// exposure in remote exporters.
pub(crate) fn llm_owner_metadata(scope_metadata: Option<&Value>) -> Value {
    let Some(Value::Object(scope_metadata)) = scope_metadata else {
        return Value::Null;
    };
    let mut metadata = Map::new();
    for key in [
        "thread_source",
        "codex_parent_thread_id",
        "codex_subagent_session_id",
        "codex_subagent_depth",
        "agent_nickname",
        "agent_role",
    ] {
        if let Some(value) = scope_metadata.get(key)
            && !value.is_null()
        {
            metadata.insert(key.to_string(), value.clone());
        }
    }
    if metadata.is_empty() {
        Value::Null
    } else {
        Value::Object(metadata)
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/alignment_codex_tests.rs"]
mod tests;
