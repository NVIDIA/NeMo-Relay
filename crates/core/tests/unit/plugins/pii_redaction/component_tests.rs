// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the PII redaction plugin component contract.
#![allow(clippy::await_holding_lock)]

use super::*;
use crate::api::event::Event;
use crate::api::llm::{
    LlmCallExecuteParams, LlmCallParams, LlmRequest, llm_call, llm_call_execute,
};
use crate::api::runtime::{
    LlmExecutionNextFn, NemoRelayContextState, create_scope_stack, global_context,
    set_thread_scope_stack,
};
use crate::api::subscriber::{deregister_subscriber, register_subscriber};
use crate::api::tool::{ToolCallEndParams, ToolCallParams, tool_call, tool_call_end};
use crate::codec::openai_chat::OpenAIChatCodec;
use crate::codec::openai_responses::OpenAIResponsesCodec;
use crate::codec::traits::LlmResponseCodec;
use crate::plugin::{
    PluginComponentSpec, PluginConfig, PluginRegistrationContext, clear_plugin_configuration,
    ensure_builtin_plugins_registered, initialize_plugins, list_plugin_kinds,
    validate_plugin_config,
};
use serde_json::json;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

fn component(config: Json) -> PluginComponentSpec {
    let Json::Object(config) = config else {
        panic!("component config must be an object");
    };
    PluginComponentSpec {
        kind: PII_REDACTION_PLUGIN_KIND.to_string(),
        enabled: true,
        config,
    }
}

fn plugin_config(config: Json) -> PluginConfig {
    PluginConfig {
        version: 1,
        components: vec![component(config)],
        policy: Default::default(),
    }
}

fn reset_runtime() {
    let _ = clear_plugin_configuration();
    crate::plugins::pii_redaction::component::clear_local_backend_provider().unwrap();
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

fn setup_isolated_thread() {
    let stack = create_scope_stack();
    set_thread_scope_stack(stack);
}

fn capture_events(name: &str) -> Arc<Mutex<Vec<Event>>> {
    let events = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    register_subscriber(
        name,
        Arc::new(move |event| sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    events
}

fn captured_events_snapshot(events: &Arc<Mutex<Vec<Event>>>) -> Vec<Event> {
    crate::api::subscriber::flush_subscribers().unwrap();
    events.lock().unwrap().clone()
}

fn noop_openai_chat_exec_fn(response: Json) -> LlmExecutionNextFn {
    Arc::new(move |_req| {
        let response = response.clone();
        Box::pin(async move { Ok(response) })
    })
}

#[test]
fn builtin_registry_includes_pii_redaction_component() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    ensure_builtin_plugins_registered().unwrap();

    let plugin_kinds = list_plugin_kinds();
    assert!(
        plugin_kinds
            .iter()
            .any(|kind| kind == PII_REDACTION_PLUGIN_KIND)
    );
}

#[test]
fn validate_rejects_config_with_no_enabled_surfaces() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "builtin": {
            "action": "remove"
        },
        "input": false,
        "output": false,
        "tool_input": false,
        "tool_output": false,
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.code == "pii_redaction.unsupported_value"
            && diag
                .message
                .contains("at least one redaction surface must be enabled")
    }));
}

#[test]
fn validate_rejects_local_section_outside_local_mode() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "builtin": {
            "action": "remove"
        },
        "local": {
            "backend": "future-local-model"
        }
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("local") && diag.message.contains("mode = 'local_model'")
    }));
}

#[test]
fn validate_rejects_builtin_mode_without_builtin_section() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin"
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("builtin")
            && diag.message.contains("required when mode = 'builtin'")
    }));
}

#[test]
fn validate_rejects_llm_surfaces_without_codec() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "builtin": {
            "action": "remove"
        },
        "input": true,
        "output": false,
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("codec")
            && diag
                .message
                .contains("codec is required when any LLM surface is enabled")
    }));
}

#[test]
fn validate_rejects_regex_replace_without_pattern() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "builtin": {
            "action": "regex_replace"
        }
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("builtin.pattern")
            && diag
                .message
                .contains("required when builtin.action = 'regex_replace'")
    }));
}

#[test]
fn validate_rejects_mask_with_empty_mask_char() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "builtin": {
            "action": "mask",
            "mask_char": ""
        }
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("builtin.mask_char")
            && diag.message.contains("must not be empty")
    }));
}

#[test]
fn validate_rejects_builtin_detector_and_pattern_together() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "builtin": {
            "action": "mask",
            "pattern": "secret",
            "detector": "email"
        }
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("builtin.detector")
            && diag.message.contains("cannot both be set")
    }));
}

#[test]
fn validate_rejects_unknown_builtin_detector() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let report = validate_plugin_config(&plugin_config(json!({
        "mode": "builtin",
        "builtin": {
            "action": "mask",
            "detector": "ssn-ish"
        }
    })));

    assert!(report.diagnostics.iter().any(|diag| {
        diag.field.as_deref() == Some("builtin.detector")
            && diag.message.contains("supported built-in detector presets")
    }));
}

#[test]
fn local_backend_provider_is_invoked_for_local_model_mode() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();

    let called = Arc::new(AtomicBool::new(false));
    let called_inner = Arc::clone(&called);
    register_local_backend_provider(Arc::new(
        move |config, _ctx: &mut PluginRegistrationContext| {
            called_inner.store(true, Ordering::SeqCst);
            assert_eq!(config.mode, "local_model");
            Ok(())
        },
    ))
    .unwrap();

    let plugin = PiiRedactionPlugin;
    let mut ctx = PluginRegistrationContext::with_namespace("test::");
    let config = json!({
        "mode": "local_model",
        "tool_input": true,
    });
    let Json::Object(config) = config else {
        panic!("component config must be object");
    };

    futures::executor::block_on(plugin.register(&config, &mut ctx)).unwrap();

    assert!(called.load(Ordering::SeqCst));
}

#[test]
fn builtin_backend_sanitizes_tool_start_and_end_payloads_with_preorder_targets() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": true,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/api_key", "/nested/token", "/result/secret"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-tool-events");
    let handle = tool_call(
        ToolCallParams::builder()
            .name("search")
            .args(json!({
                "api_key": "sk-abc123",
                "nested": {
                    "token": "sk-secret",
                    "note": "leave me"
                }
            }))
            .build(),
    )
    .unwrap();
    tool_call_end(
        ToolCallEndParams::builder()
            .handle(&handle)
            .result(json!({
                "result": {
                    "secret": "sk-final",
                    "public": "ok"
                }
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "api_key": "[REDACTED]",
            "nested": {
                "token": "[REDACTED]",
                "note": "leave me"
            }
        }))
    );
    assert_eq!(
        captured_events[1].output(),
        Some(&json!({
            "result": {
                "secret": "[REDACTED]",
                "public": "ok"
            }
        }))
    );

    deregister_subscriber("pii-redaction-tool-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_remove_deletes_object_fields_and_nulls_array_or_root_targets() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": true,
        "builtin": {
            "action": "remove",
            "target_paths": ["/secret", "/nested/remove_me", "/items/1", "/result/token"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-remove-events");
    let handle = tool_call(
        ToolCallParams::builder()
            .name("search")
            .args(json!({
                "secret": "abc",
                "nested": {
                    "keep": "yes",
                    "remove_me": "gone"
                },
                "items": ["a", "b", "c"]
            }))
            .build(),
    )
    .unwrap();
    tool_call_end(
        ToolCallEndParams::builder()
            .handle(&handle)
            .result(json!({
                "result": {
                    "token": "drop-me",
                    "public": "ok"
                }
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "nested": {
                "keep": "yes"
            },
            "items": ["a", null, "c"]
        }))
    );
    assert_eq!(
        captured_events[1].output(),
        Some(&json!({
            "result": {
                "public": "ok"
            }
        }))
    );

    deregister_subscriber("pii-redaction-remove-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_redact_replaces_matching_tool_payload_substrings_with_default_token() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "tool_input": true,
        "tool_output": true,
        "input": false,
        "output": false,
        "builtin": {
            "action": "redact",
            "detector": "bearer_token",
            "target_paths": []
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-redact-tool-events");
    let secret = "Bearer sk-demo-secret-123456";
    let handle = tool_call(
        ToolCallParams::builder()
            .name("redact_tool")
            .args(json!({
                "auth": secret,
                "message": format!("primary auth={secret}")
            }))
            .build(),
    )
    .unwrap();
    tool_call_end(
        ToolCallEndParams::builder()
            .handle(&handle)
            .result(json!({
                "result": secret,
                "nested": {"token": secret}
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(
        captured_events[0].input().unwrap()["auth"],
        json!("[REDACTED]")
    );
    assert_eq!(
        captured_events[0].input().unwrap()["message"],
        json!("primary auth=[REDACTED]")
    );
    assert_eq!(
        captured_events[1].output().unwrap()["result"],
        json!("[REDACTED]")
    );
    assert_eq!(
        captured_events[1].output().unwrap()["nested"]["token"],
        json!("[REDACTED]")
    );

    deregister_subscriber("pii-redaction-redact-tool-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_preserves_configured_prefix_and_suffix() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": true,
        "builtin": {
            "action": "mask",
            "mask_char": "*",
            "unmasked_prefix": 2,
            "unmasked_suffix": 2,
            "target_paths": ["/account", "/result/token"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-mask-events");
    let handle = tool_call(
        ToolCallParams::builder()
            .name("lookup")
            .args(json!({
                "account": "abcdef1234",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();
    tool_call_end(
        ToolCallEndParams::builder()
            .handle(&handle)
            .result(json!({
                "result": {
                    "token": "9876543210",
                    "public": "ok"
                }
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "account": "ab******34",
            "keep": "unchanged"
        }))
    );
    assert_eq!(
        captured_events[1].output(),
        Some(&json!({
            "result": {
                "token": "98******10",
                "public": "ok"
            }
        }))
    );

    deregister_subscriber("pii-redaction-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_detector_masks_only_matching_substrings() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "email",
            "mask_char": "*",
            "target_paths": ["/message"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-detector-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "message": "Email alice@example.com or bob@example.com",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "message": "Email a****@example.com or b**@example.com",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-detector-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_email_detector_preserves_domain_by_default() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "email",
            "target_paths": ["/contact"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-email-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "contact": "alice@example.com",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "contact": "a****@example.com",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-email-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_phone_detector_preserves_last_four_digits_by_default() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "phone",
            "target_paths": ["/phone"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-phone-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "phone": "+1 (555) 123-4567",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "phone": "+* (***) ***-4567",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-phone-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_api_key_detector_preserves_prefix_and_last_four_by_default() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "api_key",
            "target_paths": ["/api_key"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-api-key-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "api_key": "sk-abcdef123456",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "api_key": "sk-********3456",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-api-key-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_detector_uses_explicit_prefix_suffix_over_defaults() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "email",
            "unmasked_prefix": 2,
            "unmasked_suffix": 2,
            "target_paths": ["/contact"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-detector-explicit-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "contact": "alice@example.com",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "contact": "al*************om",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-detector-explicit-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_ip_address_detector_preserves_last_octet_by_default() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "ip_address",
            "target_paths": ["/ip"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-ip-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "ip": "192.168.10.42",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "ip": "***.***.***.42",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-ip-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_url_detector_preserves_scheme_and_host_by_default() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "url",
            "target_paths": ["/url"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-url-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "url": "https://example.com/path?q=1",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "url": "https://example.com/*",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-url-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_ipv6_detector_preserves_last_segment_by_default() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "ipv6",
            "target_paths": ["/ip"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-ipv6-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "ip": "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "ip": "****:****:****:****:****:****:****:7334",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-ipv6-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_bearer_token_detector_preserves_scheme_and_last_four() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "bearer_token",
            "target_paths": ["/auth"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-bearer-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "auth": "Bearer token-value-1234",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "auth": "Bearer ************1234",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-bearer-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_credit_card_detector_preserves_last_four_digits() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "credit_card",
            "target_paths": ["/card"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-credit-card-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "card": "4111 1111 1111 1234",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "card": "**** **** **** 1234",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-credit-card-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_jwt_detector_preserves_header_and_signature_tail() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    let jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.cGF5bG9hZA.signaturetail";
    let expected_jwt = {
        let parts = jwt.split('.').collect::<Vec<_>>();
        format!(
            "{}.{}.{}",
            parts[0],
            mask_text(parts[1], "*", 0, 0),
            mask_text(parts[2], "*", 0, 6)
        )
    };
    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "jwt",
            "target_paths": ["/token"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-jwt-default-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "token": jwt,
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "token": expected_jwt,
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-jwt-default-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_cloud_key_detectors_preserves_expected_segments() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "aws_access_key_id",
            "target_paths": ["/key"]
        }
    }))))
    .unwrap();
    let events = capture_events("pii-redaction-aws-access-key-mask-events");
    let aws_access_key = "AKIAIOSFODNN7EXAMPLE";
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({"key": aws_access_key}))
            .build(),
    )
    .unwrap();
    assert_eq!(
        captured_events_snapshot(&events)[0].input(),
        Some(&json!({"key": mask_text(aws_access_key, "*", 4, 4)}))
    );
    deregister_subscriber("pii-redaction-aws-access-key-mask-events").unwrap();
    clear_plugin_configuration().unwrap();

    reset_runtime();
    setup_isolated_thread();
    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "gcp_api_key",
            "target_paths": ["/key"]
        }
    }))))
    .unwrap();
    let events = capture_events("pii-redaction-gcp-key-mask-events");
    let gcp_key = format!("AIza{}", "A".repeat(35));
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({"key": gcp_key}))
            .build(),
    )
    .unwrap();
    assert_eq!(
        captured_events_snapshot(&events)[0].input(),
        Some(&json!({"key": mask_text(&gcp_key, "*", 6, 4)}))
    );
    deregister_subscriber("pii-redaction-gcp-key-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_hash_with_detector_hashes_only_matching_substrings() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "hash",
            "detector": "email",
            "target_paths": ["/message"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-detector-hash-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "message": "Email alice@example.com please",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "message": format!(
                "Email {} please",
                hex_sha256("alice@example.com")
            ),
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-detector-hash-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_short_detector_match_leaves_value_unchanged() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "email",
            "target_paths": ["/contact"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-short-detector-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "contact": "a@example.com",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "contact": "a@example.com",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-short-detector-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_empty_target_paths_sanitizes_all_matching_string_leaves() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "email"
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-empty-target-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "primary": "alice@example.com",
                "nested": {
                    "secondary": "bob@example.com",
                    "note": "no pii here"
                },
                "items": ["carol@example.com", "safe text"]
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "primary": "a****@example.com",
            "nested": {
                "secondary": "b**@example.com",
                "note": "no pii here"
            },
            "items": ["c****@example.com", "safe text"]
        }))
    );

    deregister_subscriber("pii-redaction-empty-target-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_malformed_ip_or_url_detector_input_leaves_value_unchanged() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "ip_address",
            "target_paths": ["/ip", "/url"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-malformed-detector-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "ip": "not-an-ip",
                "url": "mailto:alice@example.com",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "ip": "not-an-ip",
            "url": "mailto:alice@example.com",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-malformed-detector-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_mask_with_detector_sanitizes_llm_response_from_normalized_message_path() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "email",
            "target_paths": ["/message"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-detector-llm-response-events");
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let _ = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}),
            })
            .func(noop_openai_chat_exec_fn(json!({
                "id": "chatcmpl-123",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Reach me at alice@example.com"},
                        "finish_reason": "stop"
                    }
                ]
            })))
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(
        captured_events[1].output().unwrap()["choices"][0]["message"]["content"],
        json!("Reach me at a****@example.com")
    );
    assert_eq!(
        captured_events[1]
            .annotated_response()
            .and_then(|response| response.response_text()),
        Some("Reach me at a****@example.com")
    );

    deregister_subscriber("pii-redaction-detector-llm-response-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_hash_with_detector_hashes_multiple_matches_in_one_string() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "hash",
            "detector": "email",
            "target_paths": ["/message"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-multi-detector-hash-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "message": "alice@example.com and bob@example.com",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "message": format!(
                "{} and {}",
                hex_sha256("alice@example.com"),
                hex_sha256("bob@example.com")
            ),
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-multi-detector-hash-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_empty_target_paths_handles_arrays_and_multiple_detector_types() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "url"
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-array-mask-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "items": [
                    "https://example.com/a",
                    "safe text",
                    {"nested": "http://nvidia.com/private/path"},
                    42
                ],
                "keep": "mailto:alice@example.com"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "items": [
                "https://example.com/*",
                "safe text",
                {"nested": "http://nvidia.com/*"},
                42
            ],
            "keep": "mailto:alice@example.com"
        }))
    );

    deregister_subscriber("pii-redaction-array-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_detector_sanitizes_tool_output_payloads() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": false,
        "tool_output": true,
        "builtin": {
            "action": "mask",
            "detector": "email",
            "target_paths": ["/result/contact"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-tool-output-mask-events");
    let handle = tool_call(
        ToolCallParams::builder()
            .name("lookup")
            .args(json!({"query": "alice"}))
            .build(),
    )
    .unwrap();
    tool_call_end(
        ToolCallEndParams::builder()
            .handle(&handle)
            .result(json!({
                "result": {
                    "contact": "alice@example.com",
                    "public": "ok"
                }
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert_eq!(
        captured_events[1].output(),
        Some(&json!({
            "result": {
                "contact": "a****@example.com",
                "public": "ok"
            }
        }))
    );

    deregister_subscriber("pii-redaction-tool-output-mask-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_mask_with_phone_detector_ignores_non_matching_digit_shapes() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": false,
        "builtin": {
            "action": "mask",
            "detector": "phone",
            "target_paths": ["/value"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-phone-false-positive-events");
    let _handle = tool_call(
        ToolCallParams::builder()
            .name("notify")
            .args(json!({
                "value": "Order 12345 is ready",
                "keep": "unchanged"
            }))
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "value": "Order 12345 is ready",
            "keep": "unchanged"
        }))
    );

    deregister_subscriber("pii-redaction-phone-false-positive-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[test]
fn builtin_backend_sanitizes_llm_start_payload_via_codec_and_reencodes_provider_shape() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": true,
        "output": false,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/messages/0/content", "/messages/1/content"]
        }
    }))))
    .unwrap();

    let events = capture_events("pii-redaction-llm-events");
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "sk-system-secret"},
                {"role": "user", "content": "sk-user-secret"}
            ],
            "temperature": 0.2
        }),
    };

    let _handle = llm_call(
        LlmCallParams::builder()
            .name("openai")
            .request(&request)
            .build(),
    )
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 1);
    assert_eq!(
        captured_events[0].input(),
        Some(&json!({
            "headers": {},
            "content": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "[REDACTED]"},
                    {"role": "user", "content": "[REDACTED]"}
                ],
                "temperature": 0.2
            }
        }))
    );

    deregister_subscriber("pii-redaction-llm-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_llm_end_payload_and_response_codec_decodes_sanitized_output() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/choices/0/message/content"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-llm-end-events");
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "hello"}
            ]
        }),
    };
    let response = json!({
        "id": "chatcmpl-123",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "sk-response-secret"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5
        }
    });
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(request)
            .func(noop_openai_chat_exec_fn(response.clone()))
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(result, response);

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(captured_events.len(), 2);
    assert_eq!(
        captured_events[1].output(),
        Some(&json!({
            "id": "chatcmpl-123",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "[REDACTED]"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5
            }
        }))
    );

    let annotated = captured_events[1]
        .annotated_response()
        .expect("annotated_response should be present");
    assert_eq!(annotated.response_text(), Some("[REDACTED]"));
    assert_eq!(annotated.model.as_deref(), Some("gpt-4o-mini"));

    deregister_subscriber("pii-redaction-llm-end-events").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_openai_chat_response_from_normalized_message_path() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/message"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-openai-chat-normalized-response");
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let _ = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}),
            })
            .func(noop_openai_chat_exec_fn(json!({
                "id": "chatcmpl-123",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "sk-chat-secret"},
                        "finish_reason": "stop"
                    }
                ]
            })))
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(
        captured_events[1].output().unwrap()["choices"][0]["message"]["content"],
        json!("[REDACTED]")
    );
    assert_eq!(
        captured_events[1]
            .annotated_response()
            .and_then(|response| response.response_text()),
        Some("[REDACTED]")
    );

    deregister_subscriber("pii-redaction-openai-chat-normalized-response").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_redact_sanitizes_openai_chat_response_from_detector_path() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_chat",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "redact",
            "detector": "email",
            "target_paths": ["/message"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-openai-chat-redact-response");
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let _ = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}),
            })
            .func(noop_openai_chat_exec_fn(json!({
                "id": "chatcmpl-redact-123",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "alice@example.com"},
                        "finish_reason": "stop"
                    }
                ]
            })))
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(
        captured_events[1].output().unwrap()["choices"][0]["message"]["content"],
        json!("[REDACTED]")
    );
    assert_eq!(
        captured_events[1]
            .annotated_response()
            .and_then(|response| response.response_text()),
        Some("[REDACTED]")
    );

    deregister_subscriber("pii-redaction-openai-chat-redact-response").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_anthropic_response_from_normalized_message_path() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "anthropic_messages",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/message"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-anthropic-normalized-response");
    let response_codec: Arc<dyn LlmResponseCodec> =
        Arc::new(crate::codec::anthropic::AnthropicMessagesCodec);

    let _ = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("anthropic")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "hello"}]}),
            })
            .func(noop_openai_chat_exec_fn(json!({
                "id": "msg_123",
                "model": "claude-sonnet-4-20250514",
                "role": "assistant",
                "type": "message",
                "content": [{"type": "text", "text": "sk-anthropic-secret"}],
                "stop_reason": "end_turn"
            })))
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(
        captured_events[1].output().unwrap()["content"][0]["text"],
        json!("[REDACTED]")
    );
    assert_eq!(
        captured_events[1]
            .annotated_response()
            .and_then(|response| response.response_text()),
        Some("[REDACTED]")
    );

    deregister_subscriber("pii-redaction-anthropic-normalized-response").unwrap();
    clear_plugin_configuration().unwrap();
}

#[tokio::test]
async fn builtin_backend_sanitizes_openai_responses_response_from_normalized_message_path() {
    let _guard = crate::plugins::pii_redaction::test_mutex().lock().unwrap();
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "builtin",
        "codec": "openai_responses",
        "input": false,
        "output": true,
        "tool_input": false,
        "tool_output": false,
        "builtin": {
            "action": "regex_replace",
            "pattern": "sk-[A-Za-z0-9_-]+",
            "replacement": "[REDACTED]",
            "target_paths": ["/message"]
        }
    })))
    .await
    .unwrap();

    let events = capture_events("pii-redaction-openai-responses-normalized-response");
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIResponsesCodec);

    let _ = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({"model": "gpt-4.1-mini", "input": "hello"}),
            })
            .func(noop_openai_chat_exec_fn(json!({
                "id": "resp_123",
                "model": "gpt-4.1-mini",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "sk-responses-secret"}
                        ]
                    }
                ]
            })))
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let captured_events = captured_events_snapshot(&events);
    assert_eq!(
        captured_events[1].output().unwrap()["output"][0]["content"][0]["text"],
        json!("[REDACTED]")
    );
    assert_eq!(
        captured_events[1]
            .annotated_response()
            .and_then(|response| response.response_text()),
        Some("[REDACTED]")
    );

    deregister_subscriber("pii-redaction-openai-responses-normalized-response").unwrap();
    clear_plugin_configuration().unwrap();
}
