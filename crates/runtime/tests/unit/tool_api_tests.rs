// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for tool API lifecycle behavior.

#![allow(clippy::await_holding_lock)]

use std::sync::{Arc, Mutex};

use serde_json::json;

use super::{ToolCallExecuteParams, tool_call_execute};
use crate::api::event::ScopeCategory;
use crate::api::runtime::{NemoRelayContextState, global_context};
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use crate::error::FlowError;
use crate::json::Json;

fn reset_global() {
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

fn lock_global_runtime() -> std::sync::MutexGuard<'static, ()> {
    crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
}

#[test]
fn execution_context_is_preserved_across_intercepts_but_not_tool_arguments_or_events() {
    use crate::api::registry::{
        register_tool_execution_intercept, register_tool_request_intercept,
    };
    let _guard = lock_global_runtime();
    reset_global();
    let expected = json!({"example.approval": {"receipt": "private-receipt"}});
    let seen = Arc::new(Mutex::new(Vec::new()));
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = events.clone();
    register_subscriber(
        "context-events",
        Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push(serde_json::to_string(event).unwrap());
        }),
    )
    .unwrap();
    register_tool_request_intercept(
        "context-request",
        0,
        false,
        Arc::new(|_, args| {
            Box::pin(async move {
                assert_eq!(args, json!({"value": 1}));
                Ok(json!({"value": 2}))
            })
        }),
    )
    .unwrap();
    for priority in [0, 1] {
        let seen = seen.clone();
        register_tool_execution_intercept(
            &format!("context-{priority}"),
            priority,
            Arc::new(move |context, next| {
                let seen = seen.clone();
                Box::pin(async move {
                    seen.lock()
                        .unwrap()
                        .push(context.execution_context().cloned());
                    let mut local_copy = context.execution_context().cloned().unwrap();
                    local_copy["example.approval"]["receipt"] = json!("changed-copy");
                    Ok(next(context.args().clone()).await?.into())
                })
            }),
        )
        .unwrap();
    }
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let result = tool_call_execute(
            ToolCallExecuteParams::builder()
                .name("context-tool")
                .args(json!({"value": 1}))
                .execution_context(expected.clone())
                .func(Arc::new(|args| {
                    Box::pin(async move {
                        assert_eq!(args, json!({"value": 2}));
                        Ok(args.into())
                    })
                }))
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(result.result, json!({"value": 2}));
    });
    flush_subscribers().unwrap();
    assert_eq!(
        *seen.lock().unwrap(),
        vec![Some(expected.clone()), Some(expected)]
    );
    assert!(!events.lock().unwrap().is_empty());
    assert!(
        events
            .lock()
            .unwrap()
            .iter()
            .all(|event| !event.contains("private-receipt"))
    );
}

#[test]
fn execution_context_is_invocation_local_and_nested_calls_do_not_inherit_it() {
    use crate::api::registry::register_tool_execution_intercept;
    let _guard = lock_global_runtime();
    reset_global();
    register_tool_execution_intercept(
        "context-isolation",
        0,
        Arc::new(|context, next| {
            Box::pin(async move {
                tokio::task::yield_now().await;
                match context.tool_name() {
                    "left" => {
                        assert_eq!(context.execution_context(), Some(&json!({"owner": "left"})))
                    }
                    "right" => assert_eq!(
                        context.execution_context(),
                        Some(&json!({"owner": "right"}))
                    ),
                    "nested" => assert!(context.execution_context().is_none()),
                    _ => panic!("unexpected tool"),
                }
                Ok(next(context.args().clone()).await?.into())
            })
        }),
    )
    .unwrap();
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let make_call = |name: &'static str| {
            tool_call_execute(
                ToolCallExecuteParams::builder()
                    .name(name)
                    .args(json!({}))
                    .execution_context(json!({"owner": name}))
                    .func(Arc::new(|_| {
                        Box::pin(async {
                            tool_call_execute(
                                ToolCallExecuteParams::builder()
                                    .name("nested")
                                    .args(json!({}))
                                    .func(Arc::new(|args| Box::pin(async move { Ok(args.into()) })))
                                    .build(),
                            )
                            .await
                        })
                    }))
                    .build(),
            )
        };
        let (left, right) = tokio::join!(make_call("left"), make_call("right"));
        left.unwrap();
        right.unwrap();
    });
}

#[test]
fn invalid_execution_context_is_rejected_before_any_middleware_or_tool() {
    use crate::api::registry::register_tool_conditional_execution_guardrail;
    let _guard = lock_global_runtime();
    reset_global();
    register_tool_conditional_execution_guardrail(
        "context-validation",
        0,
        Arc::new(|_, _| panic!("invalid context reached middleware")),
    )
    .unwrap();
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        for value in [
            Json::Null,
            json!([]),
            json!(true),
            json!({"data": "x".repeat(16_384)}),
        ] {
            let error = tool_call_execute(
                ToolCallExecuteParams::builder()
                    .name("invalid-context")
                    .args(json!({}))
                    .execution_context(value)
                    .func(Arc::new(|_| panic!("invalid context reached tool")))
                    .build(),
            )
            .await
            .unwrap_err();
            assert!(matches!(error, FlowError::InvalidArgument(_)));
        }
    });
}

#[test]
fn execution_context_nesting_is_bounded_before_middleware_or_tool() {
    use crate::api::registry::register_tool_conditional_execution_guardrail;
    let _guard = lock_global_runtime();
    reset_global();
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = calls.clone();
    register_tool_conditional_execution_guardrail(
        "context-depth",
        0,
        Arc::new(move |_, _| {
            observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async { Ok(None) })
        }),
    )
    .unwrap();
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        for arrays in [false, true] {
            for levels in [64, 65, 161] {
                let mut value = Json::Null;
                for _ in 1..levels {
                    value = if arrays {
                        json!([value])
                    } else {
                        json!({"nested": value})
                    };
                }
                let value = json!({"nested": value});
                assert!(serde_json::to_vec(&value).unwrap().len() < 16_384);
                let before = calls.load(std::sync::atomic::Ordering::SeqCst);
                let result = tool_call_execute(
                    ToolCallExecuteParams::builder()
                        .name("context-depth")
                        .args(json!({}))
                        .execution_context(value)
                        .func(Arc::new(move |args| {
                            assert_eq!(levels, 64, "over-deep context reached tool");
                            Box::pin(async move { Ok(args.into()) })
                        }))
                        .build(),
                )
                .await;
                if levels == 64 {
                    result.unwrap();
                    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), before + 1);
                } else {
                    let error = result.unwrap_err();
                    assert!(matches!(error, FlowError::InvalidArgument(_)));
                    assert!(error.to_string().contains("64 container levels"));
                    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), before);
                }
            }
        }
    });
}

#[test]
fn application_review_claim_does_not_bypass_conditional_guardrails() {
    use crate::api::registry::register_tool_conditional_execution_guardrail;
    let _guard = lock_global_runtime();
    reset_global();
    register_tool_conditional_execution_guardrail(
        "context-is-not-authority",
        0,
        Arc::new(|_, _| Box::pin(async { Ok(Some("blocked by application policy".into())) })),
    )
    .unwrap();
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let error = tool_call_execute(
            ToolCallExecuteParams::builder()
                .name("reviewed-tool")
                .args(json!({}))
                .execution_context(json!({"review": {"approved": true}}))
                .func(Arc::new(|_| panic!("review claim bypassed a guardrail")))
                .build(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("blocked by application policy"));
    });
}

#[test]
fn tool_call_execute_adds_otel_status_metadata_to_end_events() {
    let _guard = lock_global_runtime();
    reset_global();

    let captured_events = Arc::new(Mutex::new(Vec::<(String, Option<Json>)>::new()));
    let subscriber_events = captured_events.clone();
    register_subscriber(
        "tool-status-metadata",
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::End) {
                subscriber_events
                    .lock()
                    .unwrap()
                    .push((event.name().to_string(), event.metadata().cloned()));
            }
        }),
    )
    .unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let result = tool_call_execute(
            ToolCallExecuteParams::builder()
                .name("tool-ok")
                .args(json!({"value": 1}))
                .func(Arc::new(|_args| {
                    Box::pin(async { Ok(json!({"ok": true}).into()) })
                }))
                .metadata(json!({"caller": "tool-ok", "otel.status_code": "USER"}))
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(result.result, json!({"ok": true}));
        assert!(result.annotation.is_none());

        // A handled tool failure is returned normally, but its span is an error.
        let handled = tool_call_execute(
            ToolCallExecuteParams::builder()
                .name("tool-handled-error")
                .args(json!({}))
                .func(Arc::new(|_args| {
                    Box::pin(async {
                        Ok(json!({
                            "schema": "nemo.relay.tool_result@1",
                            "content": "handled failure",
                            "is_error": true,
                            "result": {"native": "preserved"}
                        })
                        .into())
                    })
                }))
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(handled.result["result"]["native"], json!("preserved"));

        let error = tool_call_execute(
            ToolCallExecuteParams::builder()
                .name("tool-error")
                .args(json!({"value": 2}))
                .func(Arc::new(|_args| {
                    Box::pin(async { Err(FlowError::Internal("tool boom".to_string())) })
                }))
                .metadata(json!({"caller": "tool-error"}))
                .build(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("tool boom"));
    });

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("tool-status-metadata").unwrap());

    let events = captured_events.lock().unwrap();
    let metadata_for = |name: &str| {
        events
            .iter()
            .find(|event| event.0 == name)
            .and_then(|event| event.1.as_ref())
            .unwrap_or_else(|| panic!("missing end event metadata for {name}"))
    };

    let success_metadata = metadata_for("tool-ok");
    assert_eq!(success_metadata["caller"], json!("tool-ok"));
    assert_eq!(success_metadata["otel.status_code"], json!("OK"));
    assert!(success_metadata.get("otel.status_description").is_none());

    let handled_metadata = metadata_for("tool-handled-error");
    assert_eq!(handled_metadata["otel.status_code"], json!("ERROR"));
    assert!(handled_metadata.get("error.type").is_none());

    let error_metadata = metadata_for("tool-error");
    assert_eq!(error_metadata["caller"], json!("tool-error"));
    assert_eq!(error_metadata["otel.status_code"], json!("ERROR"));
    assert_eq!(error_metadata["error.type"], json!("internal_error"));
    assert!(
        error_metadata["otel.status_description"]
            .as_str()
            .unwrap()
            .contains("tool boom")
    );
}

/// Check stage order, result annotations, and lifecycle events for all tool option combinations.
#[test]
fn execution_options_select_stages_and_preserve_results_and_lifecycle() {
    use crate::api::execution::ExecutionOptions;
    use crate::api::registry::{
        register_tool_conditional_execution_guardrail, register_tool_execution_intercept,
        register_tool_request_intercept,
    };
    use crate::api::tool::ToolExecutionResult;

    let _guard = lock_global_runtime();
    reset_global();
    let stages = Arc::new(Mutex::new(Vec::new()));
    let seen = stages.clone();
    register_tool_conditional_execution_guardrail(
        "options-check",
        0,
        Arc::new(move |_, args| {
            let seen = seen.clone();
            Box::pin(async move {
                assert_eq!(args, json!({"original": true}));
                seen.lock().unwrap().push("conditional");
                Ok(None)
            })
        }),
    )
    .unwrap();
    let seen = stages.clone();
    register_tool_request_intercept(
        "options-request",
        0,
        false,
        Arc::new(move |_, _| {
            let seen = seen.clone();
            Box::pin(async move {
                seen.lock().unwrap().push("request");
                Ok(json!({"rewritten": true}))
            })
        }),
    )
    .unwrap();
    let seen = stages.clone();
    register_tool_execution_intercept(
        "options-execution",
        0,
        Arc::new(move |context, next| {
            let seen = seen.clone();
            Box::pin(async move {
                seen.lock().unwrap().push("execution");
                Ok(next(context.args().clone()).await?.into())
            })
        }),
    )
    .unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = events.clone();
    register_subscriber(
        "options-events",
        Arc::new(move |event| {
            if event.name() == "options-tool" {
                captured.lock().unwrap().push(event.scope_category());
            }
        }),
    )
    .unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        for mask in 0..8 {
            let options = ExecutionOptions {
                conditional_execution: mask & 1 == 0,
                request_intercepts: mask & 2 == 0,
                execution_intercepts: mask & 4 == 0,
            };
            stages.lock().unwrap().clear();
            events.lock().unwrap().clear();
            let seen = stages.clone();
            let result = tool_call_execute(
                ToolCallExecuteParams::builder()
                    .name("options-tool")
                    .args(json!({"original": true}))
                    .execution_options(options)
                    .func(Arc::new(move |args| {
                        let seen = seen.clone();
                        Box::pin(async move {
                            tokio::task::yield_now().await;
                            seen.lock().unwrap().push("callback");
                            Ok(ToolExecutionResult::new(args)
                                .with_annotation(json!({"opaque": true})))
                        })
                    }))
                    .build(),
            )
            .await
            .unwrap();
            assert_eq!(
                result.result,
                if options.request_intercepts {
                    json!({"rewritten": true})
                } else {
                    json!({"original": true})
                }
            );
            assert_eq!(result.annotation, Some(json!({"opaque": true})));
            let mut expected = vec![];
            if options.conditional_execution {
                expected.push("conditional");
            }
            if options.request_intercepts {
                expected.push("request");
            }
            if options.execution_intercepts {
                expected.push("execution");
            }
            expected.push("callback");
            assert_eq!(*stages.lock().unwrap(), expected);
            flush_subscribers().unwrap();
            assert_eq!(
                *events.lock().unwrap(),
                vec![Some(ScopeCategory::Start), Some(ScopeCategory::End)]
            );
        }
    });
    reset_global();
}
