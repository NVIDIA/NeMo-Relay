// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use serde_json::json;

use super::*;
use crate::agents::adapters::{claude_code, codex, hermes};

#[test]
fn maps_claude_canonical_tool_payload() {
    let headers = HeaderMap::new();
    let outcome = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "transcript_path": "/tmp/transcript.jsonl",
            "cwd": "/workspace",
            "hook_event_name": "PreToolUse",
            "tool_use_id": "toolu-1",
            "tool_name": "Read",
            "tool_input": { "file_path": "README.md" }
        }),
        &headers,
    );
    match &outcome.events[0] {
        NormalizedEvent::ToolStarted(event) => {
            assert_eq!(event.session_id, "claude-session");
            assert_eq!(event.tool_call_id, "toolu-1");
            assert_eq!(event.tool_name, "Read");
            assert_eq!(event.arguments, json!({ "file_path": "README.md" }));
            assert!(event.metadata.get("transcript_path").is_none());
            assert!(event.metadata.get("cwd").is_none());
            assert_eq!(
                event.payload["transcript_path"],
                json!("/tmp/transcript.jsonl")
            );
            assert_eq!(event.payload["cwd"], json!("/workspace"));
        }
        event => panic!("unexpected event: {event:?}"),
    }
    assert_eq!(outcome.response["continue"], json!(true));
    assert_eq!(
        outcome.response["hookSpecificOutput"],
        json!({
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow"
        })
    );
}

#[test]
fn preserves_supported_coding_agent_skill_load_tool_arguments() {
    let cases = [
        claude_code::adapt(
            json!({
                "session_id": "claude-session",
                "hook_event_name": "PreToolUse",
                "tool_use_id": "claude-skill",
                "tool_name": "Skill",
                "tool_input": {"skill": "review"}
            }),
            &HeaderMap::new(),
        ),
        codex::adapt(
            json!({
                "session_id": "codex-session",
                "hook_event_name": "PreToolUse",
                "tool_use_id": "codex-skill",
                "tool_name": "Bash",
                "tool_input": {"command": "cat /workspace/skills/review/SKILL.md"}
            }),
            &HeaderMap::new(),
        ),
        hermes::adapt(
            json!({
                "session_id": "hermes-session",
                "hook_event_name": "pre_tool_call",
                "tool_name": "skill_view",
                "tool_input": {"name": "review"},
                "extra": {"tool_call_id": "hermes-skill"}
            }),
            &HeaderMap::new(),
        ),
    ];

    let expected = [
        ("Skill", json!({"skill": "review"})),
        (
            "Bash",
            json!({"command": "cat /workspace/skills/review/SKILL.md"}),
        ),
        ("skill_view", json!({"name": "review"})),
    ];
    for (outcome, (tool_name, arguments)) in cases.into_iter().zip(expected) {
        match &outcome.events[0] {
            NormalizedEvent::ToolStarted(event) => {
                assert_eq!(event.tool_name, tool_name);
                assert_eq!(event.arguments, arguments);
            }
            event => panic!("unexpected event: {event:?}"),
        }
    }
}

#[test]
fn maps_slash_command_expansion_to_minimal_inferred_skill_mark() {
    let outcome = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "hook_event_name": "UserPromptExpansion",
            "expansion_type": "slash_command",
            "command_name": "review",
            "command_args": "123",
            "command_source": "plugin",
            "prompt": "/review 123"
        }),
        &HeaderMap::new(),
    );

    match &outcome.events[0] {
        NormalizedEvent::HookMark(event) => {
            assert_eq!(event.payload, json!({"skill_name": "review"}));
            assert_eq!(
                event.metadata[SKILL_LOAD_SOURCE_KEY],
                SKILL_LOAD_SOURCE_PROMPT_EXPANSION
            );
            assert_eq!(event.metadata["inferred"], true);
            assert_eq!(event.metadata["command_source"], "plugin");
            assert!(event.metadata.get("prompt").is_none());
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn does_not_infer_non_slash_or_empty_prompt_expansions() {
    for payload in [
        json!({
            "session_id": "claude-session",
            "hook_event_name": "UserPromptExpansion",
            "expansion_type": "mcp_prompt",
            "command_name": "review"
        }),
        json!({
            "session_id": "claude-session",
            "hook_event_name": "UserPromptExpansion",
            "expansion_type": "slash_command",
            "command_name": ""
        }),
    ] {
        let outcome = claude_code::adapt(payload, &HeaderMap::new());
        match &outcome.events[0] {
            NormalizedEvent::HookMark(event) => {
                assert_ne!(
                    event
                        .metadata
                        .get(SKILL_LOAD_SOURCE_KEY)
                        .and_then(Value::as_str),
                    Some(SKILL_LOAD_SOURCE_PROMPT_EXPANSION)
                );
                assert_eq!(event.payload["hook_event_name"], "UserPromptExpansion");
            }
            event => panic!("unexpected event: {event:?}"),
        }
    }
}

#[test]
fn maps_claude_post_tool_failure_with_canonical_fields() {
    let headers = HeaderMap::new();
    let outcome = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "hook_event_name": "PostToolUseFailure",
            "tool_use_id": "toolu-1",
            "tool_name": "Bash",
            "tool_input": { "command": "false" },
            "error": "failed",
            "is_interrupt": false,
            "duration_ms": 12
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::ToolEnded(event) => {
            assert_eq!(event.tool_call_id, "toolu-1");
            assert_eq!(event.tool_name, "Bash");
            assert_eq!(
                event.result,
                json!({ "error": "failed", "is_interrupt": false, "duration_ms": 12 })
            );
            assert_eq!(event.status.as_deref(), Some("error"));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_claude_permission_denied_as_tool_end() {
    let headers = HeaderMap::new();
    let outcome = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "hook_event_name": "PermissionDenied",
            "tool_use_id": "toolu-denied",
            "tool_name": "Bash",
            "tool_input": { "command": "rm -rf /tmp/project" },
            "reason": "policy"
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::ToolEnded(event) => {
            assert_eq!(event.tool_call_id, "toolu-denied");
            assert_eq!(event.status.as_deref(), Some("denied"));
            assert_eq!(event.result, json!({ "reason": "policy" }));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_claude_subagent_canonical_agent_id() {
    let headers = HeaderMap::new();
    let outcome = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "hook_event_name": "SubagentStart",
            "agent_id": "agent-worker-1",
            "agent_type": "general-purpose"
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::SubagentStarted(event) => {
            assert_eq!(event.subagent_id, "agent-worker-1");
            assert_eq!(event.metadata["agent_type"], json!("general-purpose"));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_claude_subagent_stop() {
    let outcome = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "hook_event_name": "SubagentStop",
            "agent_id": "agent-worker-1"
        }),
        &HeaderMap::new(),
    );

    match &outcome.events[0] {
        NormalizedEvent::SubagentEnded(event) => {
            assert_eq!(event.subagent_id, "agent-worker-1");
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_claude_stop_response_shape() {
    let outcome = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "hook_event_name": "Stop"
        }),
        &HeaderMap::new(),
    );

    // Claude's hook output schema rejects `null` for optional string fields like stopReason —
    // the adapter must omit them entirely (return only `{ continue: true }`).
    assert_eq!(outcome.response, json!({ "continue": true }));
    assert!(
        outcome.response.get("stopReason").is_none(),
        "stopReason must not appear in the response (Claude rejects null)"
    );
}

// Stop hooks on Claude/Codex (per-turn boundary) must yield a TurnEnded event so the session
// manager can snapshot ATIF without closing the agent scope. Codex needs this because it has no
// SessionEnd hook; Claude gets it for free for resilience.
#[test]
fn stop_hook_emits_turn_ended_for_codex() {
    let outcome = codex::adapt(
        json!({ "session_id": "codex-session", "hook_event_name": "Stop" }),
        &HeaderMap::new(),
    );
    assert!(
        outcome
            .events
            .iter()
            .any(|e| matches!(e, NormalizedEvent::TurnEnded(_))),
        "codex Stop must produce a TurnEnded event for ATIF snapshot. events: {:?}",
        outcome.events
    );
}

#[test]
fn multi_event_hooks_reuse_synthetic_session_id() {
    let outcome = codex::adapt(
        json!({ "hook_event_name": "UserPromptSubmit" }),
        &HeaderMap::new(),
    );
    assert_eq!(outcome.events.len(), 2);
    let prompt_session_id = outcome.events[0].session_id();
    assert!(prompt_session_id.starts_with("hook-"));
    assert_eq!(outcome.events[1].session_id(), prompt_session_id);

    let outcome = codex::adapt(json!({ "hook_event_name": "Stop" }), &HeaderMap::new());
    assert_eq!(outcome.events.len(), 2);
    let hint_session_id = outcome.events[0].session_id();
    assert!(hint_session_id.starts_with("hook-"));
    assert_eq!(outcome.events[1].session_id(), hint_session_id);
}

#[test]
fn stop_hook_emits_turn_ended_for_claude() {
    let outcome = claude_code::adapt(
        json!({ "session_id": "claude-session", "hook_event_name": "Stop" }),
        &HeaderMap::new(),
    );
    assert!(
        outcome
            .events
            .iter()
            .any(|e| matches!(e, NormalizedEvent::TurnEnded(_))),
        "claude Stop must produce a TurnEnded event for ATIF snapshot"
    );
}

#[test]
fn adapter_string_lookup_accepts_scalar_values_only() {
    let payload = json!({
        "number": 7,
        "boolean": false,
        "object": { "nested": true }
    });

    assert_eq!(string_at(&payload, &["number"]).as_deref(), Some("7"));
    assert_eq!(string_at(&payload, &["boolean"]).as_deref(), Some("false"));
    assert_eq!(string_at(&payload, &["object"]), None);
}

#[test]
fn agent_extractors_keep_fallbacks_at_adapter_boundary() {
    let headers = HeaderMap::new();
    let payload = json!({});

    fn assert_fallbacks(
        extractor: &dyn AgentPayloadExtractor,
        kind: AgentKind,
        payload: &serde_json::Value,
        headers: &HeaderMap,
    ) {
        assert_eq!(extractor.session_id(payload, headers), None);
        assert_eq!(extractor.event_name(payload), None);
        assert_eq!(extractor.subagent_id(payload, headers), None);
        assert_eq!(
            extractor.llm_hint(payload, headers),
            ExtractedLlmHint::default()
        );
        assert_eq!(
            extractor.tool_call(payload, headers, "PreToolUse"),
            ExtractedToolCall {
                tool_call_id: None,
                tool_name: None,
                subagent_id: None,
                arguments: None,
                result: None,
                status: None,
            }
        );

        assert!(session_id(payload, headers, extractor).starts_with("hook-"));
        assert_eq!(event_name(payload, extractor), "unknown");

        let event = common_tool_event_with_fallback(payload, headers, kind, extractor, "hook-test");
        assert!(event.tool_call_id.starts_with("tool-"));
        assert_eq!(event.tool_name, "unknown_tool");
        assert_eq!(event.arguments, json!(null));
        assert_eq!(event.result, json!(null));
    }

    assert_fallbacks(
        &CLAUDE_CODE_PAYLOAD_EXTRACTOR,
        AgentKind::ClaudeCode,
        &payload,
        &headers,
    );
    assert_fallbacks(
        &CODEX_PAYLOAD_EXTRACTOR,
        AgentKind::Codex,
        &payload,
        &headers,
    );
    assert_fallbacks(
        &HERMES_PAYLOAD_EXTRACTOR,
        AgentKind::Hermes,
        &payload,
        &headers,
    );
}

#[test]
fn codex_extractor_reads_agent_hint_and_tool_call_fields() {
    let headers = HeaderMap::new();
    let payload = json!({
        "subagent_id": "worker-1",
        "agent": {
            "id": "agent-1",
            "type": "reviewer"
        },
        "conversationId": "conversation-1",
        "generation": { "id": "generation-1" },
        "request": { "id": "request-1" },
        "modelName": "gpt-test",
        "tool_call_id": "tool-call-1",
        "tool": { "name": "search" },
        "arguments": { "query": "needle" },
        "result": { "matches": 2 },
        "status": "success"
    });

    assert_eq!(
        CODEX_PAYLOAD_EXTRACTOR.llm_hint(&payload, &headers),
        ExtractedLlmHint {
            subagent_id: Some("worker-1".into()),
            agent_id: Some("agent-1".into()),
            agent_type: Some("reviewer".into()),
            conversation_id: Some("conversation-1".into()),
            generation_id: Some("generation-1".into()),
            request_id: Some("request-1".into()),
            model: Some("gpt-test".into()),
        }
    );
    assert_eq!(
        CODEX_PAYLOAD_EXTRACTOR.tool_call(&payload, &headers, "PostToolUse"),
        ExtractedToolCall {
            tool_call_id: Some("tool-call-1".into()),
            tool_name: Some("search".into()),
            subagent_id: Some("worker-1".into()),
            arguments: Some(json!({ "query": "needle" })),
            result: Some(json!({ "matches": 2 })),
            status: Some("success".into()),
        }
    );
}

#[test]
fn agent_extractors_prefer_extra_call_ids_over_structural_ids() {
    let headers = HeaderMap::new();
    let payload = json!({
        "hook_event_name": "PostToolUse",
        "tool": { "id": "tool-structural" },
        "tool_input": { "id": "argument-id" },
        "id": "event-id",
        "extra": {
            "call_id": "extra-call"
        }
    });

    for extractor in [
        &CLAUDE_CODE_PAYLOAD_EXTRACTOR as &dyn AgentPayloadExtractor,
        &CODEX_PAYLOAD_EXTRACTOR,
        &HERMES_PAYLOAD_EXTRACTOR,
    ] {
        assert_eq!(
            extractor
                .tool_call(&payload, &headers, "PostToolUse")
                .tool_call_id
                .as_deref(),
            Some("extra-call")
        );
    }
}

#[test]
fn agent_extractors_keep_hook_event_name_precedence() {
    let payload = json!({
        "hook_event_name": "hook-winner",
        "event_name": "event-name-loser",
        "eventName": "event-name-camel-loser",
        "event": "event-loser"
    });

    for extractor in [
        &CLAUDE_CODE_PAYLOAD_EXTRACTOR as &dyn AgentPayloadExtractor,
        &CODEX_PAYLOAD_EXTRACTOR,
        &HERMES_PAYLOAD_EXTRACTOR,
    ] {
        assert_eq!(
            extractor.event_name(&payload).as_deref(),
            Some("hook-winner")
        );
    }
}

#[test]
fn claude_extractor_prefers_native_tool_use_id() {
    let headers = HeaderMap::new();
    let payload = json!({
        "hook_event_name": "PreToolUse",
        "tool_use_id": "claude-toolu",
        "tool_call_id": "generic-tool",
        "extra": {
            "call_id": "extra-call"
        }
    });

    assert_eq!(
        CLAUDE_CODE_PAYLOAD_EXTRACTOR
            .tool_call(&payload, &headers, "PreToolUse")
            .tool_call_id
            .as_deref(),
        Some("claude-toolu")
    );
}

#[test]
fn codex_extractor_prefers_codex_specific_fields() {
    let headers = HeaderMap::new();
    let payload = json!({
        "source": {
            "subagent": {
                "thread_spawn": {
                    "agent_nickname": "codex-reviewer"
                }
            }
        },
        "subagent": { "id": "nested-subagent" },
        "arguments": { "cmd": "cargo test" },
        "tool_input": { "cmd": "ignored", "id": "argument-id" },
        "extra": { "call_id": "extra-call" }
    });
    let tool_call = CODEX_PAYLOAD_EXTRACTOR.tool_call(&payload, &headers, "toolEnded");

    assert_eq!(tool_call.subagent_id.as_deref(), Some("codex-reviewer"));
    assert_eq!(tool_call.tool_call_id.as_deref(), Some("extra-call"));
    assert_eq!(tool_call.arguments, Some(json!({ "cmd": "cargo test" })));
}

#[test]
fn hermes_extractor_prefers_child_subagent_and_claude_session_header() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-claude-code-session-id",
        "claude-session".parse().unwrap(),
    );
    let payload = json!({
        "subagent_id": "generic-subagent",
        "child_subagent_id": "hermes-child"
    });

    assert_eq!(
        HERMES_PAYLOAD_EXTRACTOR
            .session_id(&payload, &headers)
            .as_deref(),
        Some("claude-session")
    );
    assert_eq!(
        HERMES_PAYLOAD_EXTRACTOR
            .subagent_id(&payload, &headers)
            .as_deref(),
        Some("hermes-child")
    );

    let nested_payload = json!({
        "subagent": { "id": "nested-subagent" },
        "extra": {
            "subagent_id": "extra-subagent"
        }
    });
    assert_eq!(
        HERMES_PAYLOAD_EXTRACTOR
            .subagent_id(&nested_payload, &headers)
            .as_deref(),
        Some("nested-subagent")
    );
}

#[test]
fn codex_extractor_ignores_claude_session_header() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-claude-code-session-id",
        "claude-session".parse().unwrap(),
    );

    // RelayOnly: unlike Claude Code and Hermes, Codex must not adopt the Claude
    // installed-mode session header. With no native session id the extractor
    // returns None, and the adapter boundary applies the synthetic fallback.
    assert_eq!(
        CODEX_PAYLOAD_EXTRACTOR.session_id(&json!({}), &headers),
        None
    );

    // The Claude header must not win over the native payload session id either.
    let payload = json!({ "session_id": "codex-native" });
    assert_eq!(
        CODEX_PAYLOAD_EXTRACTOR
            .session_id(&payload, &headers)
            .as_deref(),
        Some("codex-native")
    );

    // The NeMo Relay session header is still honored and takes precedence.
    headers.insert("x-nemo-relay-session-id", "relay-session".parse().unwrap());
    assert_eq!(
        CODEX_PAYLOAD_EXTRACTOR
            .session_id(&payload, &headers)
            .as_deref(),
        Some("relay-session")
    );
}

#[test]
fn keeps_codex_response_unwrapped() {
    let headers = HeaderMap::new();
    let outcome = codex::adapt(
        json!({
            "session_id": "codex-session",
            "hook_event_name": "sessionStart"
        }),
        &headers,
    );
    assert!(matches!(
        outcome.events[0],
        NormalizedEvent::AgentStarted(_)
    ));
    assert_eq!(outcome.response, json!({}));
}

#[test]
fn maps_hermes_shell_hook_tool_payload() {
    let headers = HeaderMap::new();
    let outcome = hermes::adapt(
        json!({
            "hook_event_name": "pre_tool_call",
            "tool_name": "terminal",
            "tool_input": { "command": "pwd" },
            "session_id": "hermes-session",
            "extra": {
                "task_id": "task-1",
                "tool_call_id": "tool-1"
            }
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::ToolStarted(event) => {
            assert_eq!(event.agent_kind, AgentKind::Hermes);
            assert_eq!(event.session_id, "hermes-session");
            assert_eq!(event.tool_call_id, "tool-1");
            assert_eq!(event.tool_name, "terminal");
            assert_eq!(event.arguments, json!({ "command": "pwd" }));
        }
        event => panic!("unexpected event: {event:?}"),
    }
    assert_eq!(outcome.response, json!({}));
}

#[test]
fn drops_uncorrelatable_hermes_pre_tool_call() {
    let headers = HeaderMap::new();
    let outcome = hermes::adapt(
        json!({
            "hook_event_name": "pre_tool_call",
            "task_id": "task-1",
            "tool_call_id": "toolcall-1",
            "tool_name": "terminal",
            "tool_input": { "command": "pwd" }
        }),
        &headers,
    );

    assert!(outcome.events.is_empty());
    assert_eq!(outcome.response, json!({}));
}

#[test]
fn maps_hermes_subagent_child_identifiers() {
    let headers = HeaderMap::new();
    let outcome = hermes::adapt(
        json!({
            "hook_event_name": "subagent_start",
            "session_id": "parent-session",
            "extra": {
                "child_session_id": "child-session",
                "child_subagent_id": "sa-1",
                "parent_turn_id": "turn-1"
            }
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::SubagentStarted(event) => {
            assert_eq!(event.agent_kind, AgentKind::Hermes);
            assert_eq!(event.session_id, "parent-session");
            assert_eq!(event.subagent_id, "sa-1");
            assert_eq!(
                event.payload["extra"]["child_session_id"],
                json!("child-session")
            );
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_hermes_camel_case_child_subagent_identifiers() {
    let headers = HeaderMap::new();

    for payload in [
        json!({
            "hook_event_name": "subagent_start",
            "session_id": "parent-session",
            "childSubagentId": "sa-camel-top"
        }),
        json!({
            "hook_event_name": "subagent_start",
            "session_id": "parent-session",
            "extra": {
                "childSubagentId": "sa-camel-extra"
            }
        }),
    ] {
        let expected = payload
            .get("childSubagentId")
            .or_else(|| payload.pointer("/extra/childSubagentId"))
            .and_then(|value| value.as_str())
            .expect("test payload should include childSubagentId")
            .to_string();
        let outcome = hermes::adapt(payload, &headers);

        match &outcome.events[0] {
            NormalizedEvent::SubagentStarted(event) => {
                assert_eq!(event.subagent_id, expected);
            }
            event => panic!("unexpected event: {event:?}"),
        }
    }
}

#[test]
fn maps_hermes_real_session_boundary_without_closing_per_turn_end() {
    let headers = HeaderMap::new();

    let per_turn = hermes::adapt(
        json!({
            "hook_event_name": "on_session_end",
            "session_id": "hermes-session"
        }),
        &headers,
    );
    // `on_session_end` is per-turn for hermes-agent, so it snapshots ATIF without becoming a
    // user-visible system trajectory step.
    assert_eq!(per_turn.events.len(), 1);
    assert!(matches!(per_turn.events[0], NormalizedEvent::TurnEnded(_)));

    let finalized = hermes::adapt(
        json!({
            "hook_event_name": "on_session_finalize",
            "session_id": "hermes-session"
        }),
        &headers,
    );
    assert_eq!(finalized.events.len(), 1);
    assert!(matches!(
        finalized.events[0],
        NormalizedEvent::AgentEnded(_)
    ));
}

#[test]
fn maps_hermes_hook_event_name_and_subagent_from_extra_payload() {
    let outcome = hermes::adapt(
        json!({
            "session_id": "hermes-session",
            "extra": {
                "hook_event_name": "subagent_stop",
                "subagent_id": "worker-1"
            }
        }),
        &HeaderMap::new(),
    );

    match &outcome.events[0] {
        NormalizedEvent::SubagentEnded(event) => {
            assert_eq!(event.event_name, "subagent_stop");
            assert_eq!(event.subagent_id, "worker-1");
            assert_eq!(event.session_id, "hermes-session");
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_hermes_api_hooks_to_llm_lifecycle() {
    let headers = HeaderMap::new();

    let started = hermes::adapt(
        json!({
            "hook_event_name": "pre_api_request",
            "session_id": "hermes-session",
            "extra": {
                "task_id": "task-1",
                "api_call_count": 2,
                "model": "qwen",
                "provider": "custom",
                "base_url": "http://localhost:11434/v1",
                "api_mode": "chat_completions",
                "message_count": 3,
                "tool_count": 1,
                "approx_input_tokens": 12,
                "request_char_count": 456,
                "max_tokens": 1024
            }
        }),
        &headers,
    );
    match &started.events[0] {
        NormalizedEvent::LlmStarted(event) => {
            assert_eq!(event.session_id, "hermes-session");
            assert_eq!(event.api_call_id, "hermes-session:task-1:2");
            assert_eq!(event.provider, "custom");
            assert_eq!(event.model_name.as_deref(), Some("qwen"));
            assert_eq!(event.request["message_count"], json!(3));
            assert_eq!(
                event.request["fidelity"]["provider_payload_exact"],
                json!(false)
            );
            assert_eq!(event.metadata["provider_payload_exact"], json!(false));
        }
        event => panic!("unexpected event: {event:?}"),
    }

    let ended = hermes::adapt(
        json!({
            "hook_event_name": "post_api_request",
            "session_id": "hermes-session",
            "extra": {
                "task_id": "task-1",
                "api_call_count": 2,
                "model": "qwen",
                "response_model": "qwen",
                "provider": "custom",
                "api_duration": 0.25,
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "prompt_tokens_details": { "cached_tokens": 3 }
                }
            }
        }),
        &headers,
    );
    match &ended.events[0] {
        NormalizedEvent::LlmEnded(event) => {
            assert_eq!(event.api_call_id, "hermes-session:task-1:2");
            assert_eq!(event.response["usage"]["prompt_tokens"], json!(10));
            assert_eq!(event.response["usage"]["completion_tokens"], json!(5));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_hermes_exact_api_hook_payloads_to_llm_lifecycle() {
    let headers = HeaderMap::new();

    let started = hermes::adapt(
        json!({
            "hook_event_name": "pre_api_request",
            "session_id": "hermes-session",
            "extra": {
                "task_id": "task-1",
                "api_request_id": "turn-1:api:2",
                "api_call_count": 2,
                "model": "qwen",
                "provider": "custom",
                "request": {
                    "method": "POST",
                    "body": {
                        "model": "qwen",
                        "messages": [
                            { "role": "user", "content": "hello" }
                        ],
                        "tools": [
                            { "type": "function", "function": { "name": "search_files" } }
                        ]
                    }
                }
            }
        }),
        &headers,
    );
    match &started.events[0] {
        NormalizedEvent::LlmStarted(event) => {
            assert_eq!(event.api_call_id, "turn-1:api:2");
            assert_eq!(event.request["messages"][0]["content"], json!("hello"));
            assert_eq!(
                event.request["tools"][0]["function"]["name"],
                json!("search_files")
            );
            assert_eq!(event.metadata["provider_payload_exact"], json!(true));
            assert_eq!(
                event.metadata["fidelity_source"],
                json!("hermes_api_hooks_sanitized")
            );
        }
        event => panic!("unexpected event: {event:?}"),
    }

    let ended = hermes::adapt(
        json!({
            "hook_event_name": "post_api_request",
            "session_id": "hermes-session",
            "extra": {
                "task_id": "task-1",
                "api_request_id": "turn-1:api:2",
                "api_call_count": 2,
                "model": "qwen",
                "response": {
                    "model": "qwen",
                    "finish_reason": "tool_calls",
                    "assistant_message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search_files",
                                    "arguments": "{\"query\":\"needle\"}"
                                }
                            }
                        ]
                    },
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5
                    }
                }
            }
        }),
        &headers,
    );
    match &ended.events[0] {
        NormalizedEvent::LlmEnded(event) => {
            assert_eq!(event.api_call_id, "turn-1:api:2");
            assert_eq!(event.response["tool_calls"][0]["id"], json!("call-1"));
            assert_eq!(event.response["usage"]["prompt_tokens"], json!(10));
            assert_eq!(event.metadata["provider_payload_exact"], json!(true));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_hermes_api_request_error_to_llm_end() {
    let outcome = hermes::adapt(
        json!({
            "hook_event_name": "api_request_error",
            "session_id": "hermes-session",
            "extra": {
                "task_id": "task-1",
                "api_request_id": "turn-1:api:3",
                "api_call_count": 3,
                "model": "qwen",
                "provider": "custom",
                "status_code": 502,
                "retry_count": 1,
                "max_retries": 2,
                "retryable": true,
                "reason": "upstream",
                "error": {
                    "type": "BadGateway",
                    "message": "gateway upstream error"
                }
            }
        }),
        &HeaderMap::new(),
    );

    match &outcome.events[0] {
        NormalizedEvent::LlmEnded(event) => {
            assert_eq!(event.api_call_id, "turn-1:api:3");
            assert_eq!(event.response["status_code"], json!(502));
            assert_eq!(
                event.response["error"]["message"],
                json!("gateway upstream error")
            );
            assert_eq!(event.metadata["provider_payload_exact"], json!(false));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_hermes_null_request_as_lossy_summary() {
    let outcome = hermes::adapt(
        json!({
            "hook_event_name": "pre_api_request",
            "session_id": "hermes-session",
            "extra": {
                "task_id": "task-1",
                "api_call_count": 4,
                "model": "qwen",
                "provider": "custom",
                "request": null,
                "message_count": 2
            }
        }),
        &HeaderMap::new(),
    );

    match &outcome.events[0] {
        NormalizedEvent::LlmStarted(event) => {
            assert_eq!(event.api_call_id, "hermes-session:task-1:4");
            assert_eq!(event.request["message_count"], json!(2));
            assert_eq!(
                event.request["fidelity"]["provider_payload_exact"],
                json!(false)
            );
            assert_eq!(event.metadata["provider_payload_exact"], json!(false));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn normalizes_mark_style_events_and_header_session_ids() {
    let mut headers = HeaderMap::new();
    headers.insert("x-nemo-relay-session-id", "header-session".parse().unwrap());
    headers.insert("x-nemo-relay-config-profile", "coverage".parse().unwrap());

    for (event_name, expected) in [
        ("UserPromptSubmit", "prompt"),
        ("afterAgentResponse", "response"),
        ("PreCompact", "compact"),
        ("PostCompact", "compact"),
        (COMPACTION_EVENT_NAME, "compact"),
        ("Notification", "notification"),
        ("Unrecognized.Event", "hook"),
    ] {
        let outcome = codex::adapt(
            json!({
                "eventName": event_name,
                "model": "model-a",
                "cwd": "/repo"
            }),
            &headers,
        );
        let (session_id, metadata, payload) = match &outcome.events[0] {
            NormalizedEvent::PromptSubmitted(event) if expected == "prompt" => {
                (event.session_id.as_str(), &event.metadata, &event.payload)
            }
            NormalizedEvent::LlmHint(event) if expected == "response" => {
                (event.session_id.as_str(), &event.metadata, &event.payload)
            }
            NormalizedEvent::Compaction(event) if expected == "compact" => {
                (event.session_id.as_str(), &event.metadata, &event.payload)
            }
            NormalizedEvent::Notification(event) if expected == "notification" => {
                (event.session_id.as_str(), &event.metadata, &event.payload)
            }
            NormalizedEvent::HookMark(event) if expected == "hook" => {
                (event.session_id.as_str(), &event.metadata, &event.payload)
            }
            event => panic!("unexpected event for {event_name}: {event:?}"),
        };
        if expected == "prompt" {
            assert!(
                matches!(outcome.events.get(1), Some(NormalizedEvent::LlmHint(_))),
                "prompt hooks should also emit a private LLM hint"
            );
        }
        assert_eq!(session_id, "header-session");
        assert_eq!(metadata["model"], json!("model-a"));
        assert!(metadata.get("cwd").is_none());
        assert_eq!(payload["cwd"], json!("/repo"));
        assert_eq!(metadata["gateway_config_profile"], json!("coverage"));
    }
}

#[test]
fn maps_hermes_llm_hooks_to_private_hints() {
    let headers = HeaderMap::new();
    let outcome = hermes::adapt(
        json!({
            "hook_event_name": "pre_llm_call",
            "session_id": "hermes-session",
            "model": "anthropic/claude-sonnet",
            "request_id": "req-1"
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::LlmHint(event) => {
            assert_eq!(event.session_id, "hermes-session");
            assert_eq!(event.event_name, "pre_llm_call");
            assert_eq!(event.model.as_deref(), Some("anthropic/claude-sonnet"));
            assert_eq!(event.request_id.as_deref(), Some("req-1"));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn extracts_tool_fields_from_fallback_payload_shapes() {
    let headers = HeaderMap::new();
    let outcome = codex::adapt(
        json!({
            "conversationId": "conversation-1",
            "event": "toolEnded",
            "tool": { "id": "tool-id", "name": "Shell" },
            "arguments": { "cmd": "pwd" },
            "result": { "stdout": "/repo" },
            "permission": "allow"
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::ToolEnded(event) => {
            assert_eq!(event.session_id, "conversation-1");
            assert_eq!(event.tool_call_id, "tool-id");
            assert_eq!(event.tool_name, "Shell");
            assert_eq!(event.arguments, json!({ "cmd": "pwd" }));
            assert_eq!(event.result, json!({ "stdout": "/repo" }));
            assert_eq!(event.status.as_deref(), Some("allow"));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn generated_ids_are_used_when_payload_omits_identifiers() {
    let headers = HeaderMap::new();
    let outcome = claude_code::adapt(
        json!({
            "hook_event_name": "PreToolUse",
            "tool_input": { "name": "Read", "file_path": "Cargo.toml" }
        }),
        &headers,
    );

    match &outcome.events[0] {
        NormalizedEvent::ToolStarted(event) => {
            assert!(event.session_id.starts_with("hook-"));
            assert!(event.tool_call_id.starts_with("tool-"));
            assert_eq!(event.tool_name, "Read");
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn stop_responses_preserve_vendor_shapes() {
    let headers = HeaderMap::new();
    let claude = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "hook_event_name": "Stop"
        }),
        &headers,
    );
    assert!(matches!(claude.events[0], NormalizedEvent::LlmHint(_)));
    assert!(
        claude.response.get("stopReason").is_none(),
        "stopReason must not be present (Claude rejects null per its hook schema)"
    );

    let codex = codex::adapt(
        json!({
            "session_id": "codex-session",
            "hook_event_name": "stop"
        }),
        &headers,
    );
    assert!(matches!(codex.events[0], NormalizedEvent::LlmHint(_)));
    assert_eq!(codex.response, json!({}));
}

#[test]
fn claude_partial_tool_payload_mixes_native_and_fallback_identifiers() {
    let outcome = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash"
        }),
        &HeaderMap::new(),
    );

    match &outcome.events[0] {
        NormalizedEvent::ToolStarted(event) => {
            assert_eq!(event.session_id, "claude-session");
            assert!(event.tool_call_id.starts_with("tool-"));
            assert_eq!(event.tool_name, "Bash");
            assert_eq!(event.arguments, json!(null));
            assert_eq!(event.result, json!(null));
            assert_eq!(event.status, None);
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn codex_partial_tool_end_keeps_missing_fields_null() {
    let outcome = codex::adapt(
        json!({
            "conversationId": "conversation-1",
            "event": "toolEnded",
            "tool_call_id": "call-1"
        }),
        &HeaderMap::new(),
    );

    match &outcome.events[0] {
        NormalizedEvent::ToolEnded(event) => {
            assert_eq!(event.session_id, "conversation-1");
            assert_eq!(event.tool_call_id, "call-1");
            assert_eq!(event.tool_name, "unknown_tool");
            assert_eq!(event.arguments, json!(null));
            assert_eq!(event.result, json!(null));
            assert_eq!(event.status, None);
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn hermes_partial_post_tool_payload_synthesizes_call_id_only() {
    let outcome = hermes::adapt(
        json!({
            "hook_event_name": "post_tool_call",
            "session_id": "hermes-session",
            "tool_name": "terminal"
        }),
        &HeaderMap::new(),
    );

    match &outcome.events[0] {
        NormalizedEvent::ToolEnded(event) => {
            assert_eq!(event.session_id, "hermes-session");
            assert!(event.tool_call_id.starts_with("tool-"));
            assert_eq!(event.tool_name, "terminal");
            assert_eq!(event.arguments, json!(null));
            assert_eq!(event.result, json!(null));
            assert_eq!(event.status, None);
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

#[test]
fn maps_hermes_post_tool_call_result_and_status_shapes() {
    let outcome = hermes::adapt(
        json!({
            "hook_event_name": "post_tool_call",
            "session_id": "hermes-session",
            "tool_call_id": "tool-1",
            "tool_name": "terminal",
            "tool_response": { "stdout": "/repo" },
            "decision": "allow"
        }),
        &HeaderMap::new(),
    );

    match &outcome.events[0] {
        NormalizedEvent::ToolEnded(event) => {
            assert_eq!(event.tool_call_id, "tool-1");
            assert_eq!(event.result, json!({ "stdout": "/repo" }));
            assert_eq!(event.status.as_deref(), Some("allow"));
        }
        event => panic!("unexpected event: {event:?}"),
    }
}

/// Walks a payload-path precedence chain: each step asserts the expected
/// winner, then removes the winning key (from the payload root, or from its
/// `extra` object when the first tuple field is true) so the next candidate
/// takes over. Once every listed key is removed, the extraction must yield
/// nothing.
fn assert_string_fallback_chain(
    payload: &mut serde_json::Value,
    chain: &[(bool, &str, &str)],
    extract: impl Fn(&serde_json::Value) -> Option<String>,
) {
    for (remove_from_extra, key, expected) in chain {
        assert_eq!(
            extract(payload).as_deref(),
            Some(*expected),
            "winner before removing `{key}`"
        );
        let object = if *remove_from_extra {
            payload["extra"].as_object_mut().unwrap()
        } else {
            payload.as_object_mut().unwrap()
        };
        object.remove(*key);
    }
    assert_eq!(extract(payload), None, "chain should be exhausted");
}

#[test]
fn hermes_tool_result_path_precedence_walks_fallback_chain() {
    let headers = HeaderMap::new();
    let mut payload = json!({
        "tool_output": "from-tool-output",
        "tool_response": "from-tool-response",
        "output": "from-output",
        "result": "from-result",
        "extra": {
            "tool_output": "from-extra-tool-output",
            "result": "from-extra-result"
        }
    });

    assert_string_fallback_chain(
        &mut payload,
        &[
            (false, "tool_output", "from-tool-output"),
            (false, "tool_response", "from-tool-response"),
            (false, "output", "from-output"),
            (false, "result", "from-result"),
            (true, "tool_output", "from-extra-tool-output"),
            (true, "result", "from-extra-result"),
        ],
        |payload| {
            HERMES_PAYLOAD_EXTRACTOR
                .tool_call(payload, &headers, "post_tool_call")
                .result
                .map(|result| result.as_str().expect("string tool result").to_string())
        },
    );
}

#[test]
fn hermes_tool_status_prefers_explicit_fields_over_derived_status() {
    let headers = HeaderMap::new();
    let mut payload = json!({
        "status": "success",
        "decision": "block",
        "permission": "deny"
    });

    assert_string_fallback_chain(
        &mut payload,
        &[
            (false, "status", "success"),
            (false, "decision", "block"),
            (false, "permission", "deny"),
        ],
        |payload| {
            HERMES_PAYLOAD_EXTRACTOR
                .tool_call(payload, &headers, "post_tool_call")
                .status
        },
    );

    // Explicit status fields win over event-name-derived status; without them the conservative
    // failure spellings still map to `error`.
    assert_eq!(
        HERMES_PAYLOAD_EXTRACTOR
            .tool_call(
                &json!({ "status": "success" }),
                &headers,
                "post_tool_call_failed"
            )
            .status
            .as_deref(),
        Some("success")
    );
    assert_eq!(
        HERMES_PAYLOAD_EXTRACTOR
            .tool_call(&json!({}), &headers, "post_tool_call_failed")
            .status
            .as_deref(),
        Some("error")
    );
}

#[test]
fn claude_extractor_reads_llm_hint_fields() {
    let headers = HeaderMap::new();
    let payload = json!({
        "hook_event_name": "Stop",
        "session_id": "claude-session",
        "subagent_id": "worker-1",
        "agent_id": "agent-1",
        "agent_type": "general-purpose",
        "conversation_id": "conversation-1",
        "generation_id": "generation-1",
        "extra": { "request_id": "request-1" },
        "model": "claude-sonnet-4"
    });

    assert_eq!(
        CLAUDE_CODE_PAYLOAD_EXTRACTOR.llm_hint(&payload, &headers),
        ExtractedLlmHint {
            subagent_id: Some("worker-1".into()),
            agent_id: Some("agent-1".into()),
            agent_type: Some("general-purpose".into()),
            conversation_id: Some("conversation-1".into()),
            generation_id: Some("generation-1".into()),
            request_id: Some("request-1".into()),
            model: Some("claude-sonnet-4".into()),
        }
    );
}

#[test]
fn claude_stop_with_partial_hint_fields_keeps_missing_fields_none() {
    let outcome = claude_code::adapt(
        json!({
            "session_id": "claude-session",
            "hook_event_name": "Stop",
            "model": "claude-sonnet-4"
        }),
        &HeaderMap::new(),
    );

    match &outcome.events[0] {
        NormalizedEvent::LlmHint(event) => {
            assert_eq!(event.session_id, "claude-session");
            assert_eq!(event.model.as_deref(), Some("claude-sonnet-4"));
            assert_eq!(event.subagent_id, None);
            assert_eq!(event.agent_id, None);
            assert_eq!(event.agent_type, None);
            assert_eq!(event.conversation_id, None);
            assert_eq!(event.generation_id, None);
            assert_eq!(event.request_id, None);
            assert_eq!(event.metadata["model"], json!("claude-sonnet-4"));
        }
        event => panic!("unexpected event: {event:?}"),
    }
    assert!(matches!(outcome.events[1], NormalizedEvent::TurnEnded(_)));
}

#[test]
fn llm_hint_model_and_request_id_precedence_chains() {
    let headers = HeaderMap::new();

    // Hint extraction is shared across harnesses; Claude Code stands in for all of them.
    let mut payload = json!({
        "model": "flat-model",
        "model_name": "snake-model",
        "modelName": "camel-model"
    });
    assert_string_fallback_chain(
        &mut payload,
        &[
            (false, "model", "flat-model"),
            (false, "model_name", "snake-model"),
            (false, "modelName", "camel-model"),
        ],
        |payload| {
            CLAUDE_CODE_PAYLOAD_EXTRACTOR
                .llm_hint(payload, &headers)
                .model
        },
    );

    let mut payload = json!({
        "request_id": "flat-snake",
        "requestId": "flat-camel",
        "request": { "id": "nested" },
        "extra": { "request_id": "extra" }
    });
    assert_string_fallback_chain(
        &mut payload,
        &[
            (false, "request_id", "flat-snake"),
            (false, "requestId", "flat-camel"),
            (false, "request", "nested"),
            (false, "extra", "extra"),
        ],
        |payload| {
            CLAUDE_CODE_PAYLOAD_EXTRACTOR
                .llm_hint(payload, &headers)
                .request_id
        },
    );
}

#[test]
fn session_id_path_precedence_walks_fallback_chain() {
    let headers = HeaderMap::new();
    // The canonical session-id chain is shared by every harness; Claude Code stands in for all.
    let mut payload = json!({
        "session_id": "flat-snake",
        "sessionId": "flat-camel",
        "session": { "id": "nested" },
        "conversation_id": "conversation-snake",
        "conversationId": "conversation-camel",
        "parent_session_id": "parent",
        "task_id": "task",
        "extra": { "session_id": "extra-session", "task_id": "extra-task" }
    });

    assert_string_fallback_chain(
        &mut payload,
        &[
            (false, "session_id", "flat-snake"),
            (false, "sessionId", "flat-camel"),
            (false, "session", "nested"),
            (false, "conversation_id", "conversation-snake"),
            (false, "conversationId", "conversation-camel"),
            (false, "parent_session_id", "parent"),
            (false, "task_id", "task"),
            (true, "session_id", "extra-session"),
            (true, "task_id", "extra-task"),
        ],
        |payload| CLAUDE_CODE_PAYLOAD_EXTRACTOR.session_id(payload, &headers),
    );
}

#[test]
fn event_name_path_precedence_walks_fallback_chain() {
    let mut payload = json!({
        "hook_event_name": "hook-name",
        "eventName": "camel-name",
        "type": "type-name",
        "name": "name-name",
        "extra": {
            "hook_event_name": "extra-hook-name",
            "name": "extra-name"
        }
    });

    assert_string_fallback_chain(
        &mut payload,
        &[
            (false, "hook_event_name", "hook-name"),
            (false, "eventName", "camel-name"),
            (false, "type", "type-name"),
            (false, "name", "name-name"),
            (true, "hook_event_name", "extra-hook-name"),
            (true, "name", "extra-name"),
        ],
        |payload| CLAUDE_CODE_PAYLOAD_EXTRACTOR.event_name(payload),
    );
}

#[test]
fn subagent_id_path_precedence_falls_back_to_nested_shapes_then_header() {
    let headers = HeaderMap::new();
    let mut payload = json!({
        "subagent_id": "flat-subagent",
        "child_subagent_id": "flat-child",
        "agent_id": "flat-agent",
        "subagent": { "id": "nested-subagent" },
        "agent": { "id": "nested-agent" },
        "extra": { "agent_id": "extra-agent" }
    });

    assert_string_fallback_chain(
        &mut payload,
        &[
            (false, "subagent_id", "flat-subagent"),
            (false, "child_subagent_id", "flat-child"),
            (false, "agent_id", "flat-agent"),
            (false, "subagent", "nested-subagent"),
            (false, "agent", "nested-agent"),
            (true, "agent_id", "extra-agent"),
        ],
        |payload| CLAUDE_CODE_PAYLOAD_EXTRACTOR.subagent_id(payload, &headers),
    );

    // With every payload path exhausted, the explicit subagent header is the last fallback.
    let mut header_map = HeaderMap::new();
    header_map.insert("x-nemo-relay-subagent-id", "header-worker".parse().unwrap());
    assert_eq!(
        CLAUDE_CODE_PAYLOAD_EXTRACTOR
            .subagent_id(&payload, &header_map)
            .as_deref(),
        Some("header-worker")
    );
}

#[test]
fn codex_subagent_id_prefers_flat_ids_over_thread_spawn_nickname() {
    let headers = HeaderMap::new();
    let mut payload = json!({
        "subagent_id": "flat-subagent",
        "source": {
            "subagent": { "thread_spawn": { "agent_nickname": "codex-nickname" } }
        },
        "subagent": { "id": "nested-subagent" }
    });

    assert_string_fallback_chain(
        &mut payload,
        &[
            (false, "subagent_id", "flat-subagent"),
            (false, "source", "codex-nickname"),
            (false, "subagent", "nested-subagent"),
        ],
        |payload| CODEX_PAYLOAD_EXTRACTOR.subagent_id(payload, &headers),
    );
}

#[test]
fn null_intermediate_objects_do_not_mask_string_fallback_paths() {
    let headers = HeaderMap::new();
    let payload = json!({
        "subagent": null,
        "agent": { "id": "nested-agent" }
    });

    // A null intermediate stops the `subagent.id` lookup without error, so the later
    // `agent.id` candidate still supplies the identifier.
    assert_eq!(
        CLAUDE_CODE_PAYLOAD_EXTRACTOR
            .subagent_id(&payload, &headers)
            .as_deref(),
        Some("nested-agent")
    );
}

#[test]
fn json_path_lookups_handle_empty_strings_arrays_and_deep_nesting() {
    let payload = json!({
        "empty": "",
        "list": [{ "id": "inside-array" }],
        "deep": { "level_two": { "level_three": { "value": "deep-value" } } }
    });

    // Empty strings exist as values but are filtered from string lookups.
    assert_eq!(value_at(&payload, &["empty"]), Some(json!("")));
    assert_eq!(string_at(&payload, &["empty"]), None);

    // Arrays are not traversed by object-key paths.
    assert_eq!(value_at(&payload, &["list", "id"]), None);

    assert_eq!(
        string_at(&payload, &["deep", "level_two", "level_three", "value"]).as_deref(),
        Some("deep-value")
    );
    assert_eq!(
        first_string_at(
            &payload,
            &[
                &["missing"][..],
                &["empty"][..],
                &["list", "id"][..],
                &["deep", "level_two", "level_three", "value"][..],
            ],
        )
        .as_deref(),
        Some("deep-value")
    );
    assert_eq!(
        first_value_at(&payload, &[&["missing"][..], &["also", "missing"][..]]),
        None
    );
}
