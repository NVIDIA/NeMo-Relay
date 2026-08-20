// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn native_executor_config_accepts_only_positive_integer_overrides() {
    assert_eq!(NativeExecutorConfig::default().worker_threads, 2);
    assert_eq!(
        NativeExecutorConfig::default()
            .with_component_config(&Map::new())
            .unwrap()
            .worker_threads,
        2
    );
    assert_eq!(
        NativeExecutorConfig::default()
            .with_component_config(&Map::from_iter([(
                "executor".into(),
                serde_json::json!({"worker_threads": 3}),
            )]))
            .unwrap()
            .worker_threads,
        3
    );
    for config in [
        serde_json::json!(true),
        serde_json::json!({"worker_threads": 0}),
        serde_json::json!({"worker_threads": -1}),
        serde_json::json!({"worker_threads": true}),
        serde_json::json!({"worker_threads": 1.5}),
        serde_json::json!({"worker_threads": "two"}),
    ] {
        assert!(
            NativeExecutorConfig::default()
                .with_component_config(&Map::from_iter([("executor".into(), config)]))
                .is_err()
        );
    }
}

#[test]
fn native_executor_starts_once_and_drains_accepted_tasks_on_drop() {
    let executor = NativeExecutor::new(NativeExecutorConfig { worker_threads: 1 }, "coverage-test");
    let first = executor.ensure_started().unwrap();
    let second = executor.ensure_started().unwrap();
    assert_eq!(first.id(), second.id());

    let completed = Arc::new(AtomicBool::new(false));
    let observed = Arc::clone(&completed);
    executor
        .spawn(async move {
            observed.store(true, Ordering::Release);
        })
        .unwrap();
    drop(executor);
    assert!(completed.load(Ordering::Acquire));
}

#[test]
fn async_sdk_status_helpers_keep_operation_labels_stable() {
    assert!(status_result(NemoRelayStatus::Ok, "register").is_ok());
    assert_eq!(
        status_result(NemoRelayStatus::InvalidArg, "register").unwrap_err(),
        "register failed: InvalidArg"
    );
    let labels = [
        (
            NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeRequest,
            "tool request sanitizer",
        ),
        (
            NemoRelayNativeAsyncMiddlewareKind::ToolSanitizeResponse,
            "tool response sanitizer",
        ),
        (
            NemoRelayNativeAsyncMiddlewareKind::ToolConditionalExecution,
            "tool conditional guardrail",
        ),
        (
            NemoRelayNativeAsyncMiddlewareKind::ToolRequestIntercept,
            "tool request intercept",
        ),
        (
            NemoRelayNativeAsyncMiddlewareKind::ToolExecutionIntercept,
            "tool execution intercept",
        ),
        (
            NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeRequest,
            "LLM request sanitizer",
        ),
        (
            NemoRelayNativeAsyncMiddlewareKind::LlmSanitizeResponse,
            "LLM response sanitizer",
        ),
        (
            NemoRelayNativeAsyncMiddlewareKind::LlmConditionalExecution,
            "LLM conditional guardrail",
        ),
        (
            NemoRelayNativeAsyncMiddlewareKind::LlmRequestIntercept,
            "LLM request intercept",
        ),
        (
            NemoRelayNativeAsyncMiddlewareKind::LlmExecutionIntercept,
            "LLM execution intercept",
        ),
        (
            NemoRelayNativeAsyncMiddlewareKind::LlmStreamExecutionIntercept,
            "LLM stream execution intercept",
        ),
        (
            NemoRelayNativeAsyncMiddlewareKind::MarkSanitize,
            "mark sanitizer",
        ),
        (
            NemoRelayNativeAsyncMiddlewareKind::ScopeSanitizeStart,
            "scope start sanitizer",
        ),
        (
            NemoRelayNativeAsyncMiddlewareKind::ScopeSanitizeEnd,
            "scope end sanitizer",
        ),
        (
            NemoRelayNativeAsyncMiddlewareKind::EventMetadataInjector,
            "Event metadata injector",
        ),
    ];
    for (kind, expected) in labels {
        assert_eq!(registration_operation(kind), expected);
    }
}
