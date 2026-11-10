// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile and exercise the exact Rust source displayed by the concept guides.

mod fork_task {
    include!("../../../docs/about-nemo-relay/concepts/_code/fork_task.rs");

    #[tokio::test]
    async fn spawned_worker_uses_a_forked_stack() {
        nemo_relay::context::with_task_scope_stack(
            nemo_relay::context::create_scope_stack(),
            async { spawn_worker().await.unwrap().await.unwrap() },
        )
        .await;
    }
}

mod run_agent {
    include!("../../../docs/about-nemo-relay/concepts/_code/run_agent.rs");

    #[tokio::test]
    async fn agent_closes_its_scope_after_success_and_provider_failure() {
        nemo_relay::context::with_task_scope_stack(
            nemo_relay::context::create_scope_stack(),
            async {
                let baseline = nemo_relay::context::current_scope_stack_snapshot()
                    .unwrap()
                    .unwrap();
                for fail in [false, true] {
                    let request = LlmRequest {
                        headers: Default::default(),
                        content: serde_json::json!({"model": "example-model"}),
                    };
                    let provider: LlmExecutionNextFn = std::sync::Arc::new(move |_| {
                        Box::pin(async move {
                            if fail {
                                Err(nemo_relay::FlowError::InvalidArgument(
                                    "provider failed".into(),
                                ))
                            } else {
                                Ok(serde_json::json!({"answer": "hello"}))
                            }
                        })
                    });
                    let outcome = run_agent(request, provider).await;
                    if fail {
                        assert!(outcome.is_err());
                    } else {
                        let (response, traceparent) = outcome.unwrap();
                        assert_eq!(response["answer"], "hello");
                        assert!(traceparent.starts_with("00-"));
                    }
                    let after = nemo_relay::context::current_scope_stack_snapshot()
                        .unwrap()
                        .unwrap();
                    assert_eq!(after.scopes.len(), baseline.scopes.len());
                    assert_eq!(after.scopes[0].scope_id, baseline.scopes[0].scope_id);
                }
            },
        )
        .await;
    }
}

mod correlate_agent {
    include!("../../../docs/about-nemo-relay/concepts/_code/correlate_agent.rs");

    #[tokio::test]
    async fn correlation_finds_the_active_agent() {
        use nemo_relay::scope::{self, PopScopeParams, PushScopeParams};
        nemo_relay::context::with_task_scope_stack(
            nemo_relay::context::create_scope_stack(),
            async {
                let baseline = agent_scope_uuid().unwrap();
                let agent = scope::push_scope(
                    PushScopeParams::builder()
                        .name("agent")
                        .scope_type(ScopeType::Agent)
                        .build(),
                )
                .unwrap();
                assert_eq!(agent_scope_uuid().unwrap(), Some(agent.uuid.to_string()));
                scope::pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build())
                    .unwrap();
                assert_eq!(agent_scope_uuid().unwrap(), baseline);
            },
        )
        .await;
    }
}

mod redact_openai {
    include!("../../../docs/about-nemo-relay/concepts/_code/redact_openai.rs");

    #[test]
    fn middleware_example_registers_through_the_sdk() {
        register_redaction().unwrap();
        nemo_relay::registry::deregister_llm_sanitize_request_guardrail("redact-openai-chat")
            .unwrap();
    }
}

mod instrument_applications {
    use nemo_relay::context::{
        create_scope_stack, current_scope_stack_snapshot, set_thread_scope_stack,
    };
    use nemo_relay::registry::{
        deregister_llm_conditional_execution_guardrail,
        deregister_tool_conditional_execution_guardrail, deregister_tool_execution_intercept,
        deregister_tool_sanitize_request_guardrail, list_runtime_registrations,
        register_llm_conditional_execution_guardrail,
        register_tool_conditional_execution_guardrail,
    };
    use nemo_relay::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
    use nemo_relay::tool::{ToolCallExecuteParams, ToolExecutionResult, tool_call_execute};
    use serde_json::{Value, json};
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    mod tool_program {
        include!("../../../docs/instrument-applications/_code/rust/tool_call.rs");
        pub(super) fn run() -> Result<(), Box<dyn std::error::Error>> {
            main()
        }
    }
    mod model_program {
        include!("../../../docs/instrument-applications/_code/rust/llm_call.rs");
        pub(super) fn run() -> Result<(), Box<dyn std::error::Error>> {
            main()
        }
    }
    mod query_policy {
        include!("../../../docs/instrument-applications/_code/rust/query_policy.rs");
    }
    mod tool_policy {
        include!("../../../docs/instrument-applications/_code/rust/tool_policy.rs");
    }
    mod printer {
        include!("../../../docs/instrument-applications/_code/rust/print_event.rs");
    }

    fn paired_call(records: &[Value], category: &str, name: &str) -> Value {
        let starts: Vec<_> = records
            .iter()
            .filter(|event| {
                event["category"] == category
                    && event["name"] == name
                    && event["scope_category"] == "start"
            })
            .collect();
        assert_eq!(starts.len(), 1, "{category} {name}: {records:?}");
        let start = starts[0];
        let ends: Vec<_> = records
            .iter()
            .filter(|event| event["uuid"] == start["uuid"] && event["scope_category"] == "end")
            .collect();
        assert_eq!(ends.len(), 1);
        assert_eq!(start["parent_uuid"], ends[0]["parent_uuid"]);
        start.clone()
    }

    // One test keeps the examples' process-wide registration names sequential.
    #[test]
    fn documented_programs_policies_and_cleanup() {
        set_thread_scope_stack(create_scope_stack());
        let records = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&records);
        register_subscriber(
            "documentation-capture",
            Arc::new(move |event| {
                captured.lock().unwrap().push(event.to_json_value());
            }),
        )
        .unwrap();

        for (program, category, name) in [
            (
                tool_program::run as fn() -> Result<(), Box<dyn std::error::Error>>,
                "tool",
                "search",
            ),
            (model_program::run, "llm", "demo-provider"),
        ] {
            records.lock().unwrap().clear();
            program().unwrap();
            flush_subscribers().unwrap();
            let events = records.lock().unwrap().clone();
            let call = paired_call(&events, category, name);
            let end = events
                .iter()
                .find(|event| event["uuid"] == call["uuid"] && event["scope_category"] == "end")
                .unwrap();
            let expected = if category == "tool" {
                json!({"query": "runtime instrumentation", "hits": [{"title": "NeMo Relay"}]})
            } else {
                json!({"text": "hello", "messages": [{"role": "user", "content": "hello"}]})
            };
            assert_eq!(end["data"], expected);
            let parent = events
                .iter()
                .find(|event| {
                    event["uuid"] == call["parent_uuid"] && event["scope_category"] == "start"
                })
                .unwrap();
            assert_eq!(parent["name"], "agent-run");
            assert!(
                events.iter().any(
                    |event| event["uuid"] == parent["uuid"] && event["scope_category"] == "end"
                )
            );
        }

        let baseline = current_scope_stack_snapshot()
            .unwrap()
            .unwrap()
            .scopes
            .len();
        register_tool_conditional_execution_guardrail(
            "documentation.reject",
            1,
            Arc::new(|_, _| Box::pin(async { Ok(Some("documentation rejection".into())) })),
        )
        .unwrap();
        assert!(
            tool_program::run()
                .unwrap_err()
                .to_string()
                .contains("documentation rejection")
        );
        deregister_tool_conditional_execution_guardrail("documentation.reject").unwrap();
        assert_eq!(
            current_scope_stack_snapshot()
                .unwrap()
                .unwrap()
                .scopes
                .len(),
            baseline
        );

        register_llm_conditional_execution_guardrail(
            "documentation.reject-model",
            1,
            Arc::new(|_request| Box::pin(async { Ok(Some("documentation rejection".into())) })),
        )
        .unwrap();
        assert!(
            model_program::run()
                .unwrap_err()
                .to_string()
                .contains("documentation rejection")
        );
        deregister_llm_conditional_execution_guardrail("documentation.reject-model").unwrap();
        assert_eq!(
            current_scope_stack_snapshot()
                .unwrap()
                .unwrap()
                .scopes
                .len(),
            baseline
        );
        let registrations = list_runtime_registrations(None).unwrap();
        assert!(!registrations.iter().any(|registration| matches!(
            registration.local_name.as_str(),
            "instrumentation-check" | "llm-check"
        )));

        tokio::runtime::Runtime::new().unwrap().block_on(async {
            for combined in [false, true] {
                if combined { tool_policy::register_tool_policy().unwrap(); }
                else { query_policy::register_query_policy().unwrap(); }
                for query in ["weather", ""] {
                    records.lock().unwrap().clear();
                    let calls = Arc::new(AtomicUsize::new(0));
                    let callback_calls = Arc::clone(&calls);
                    let result = tool_call_execute(ToolCallExecuteParams::builder()
                        .name("search")
                        .args(json!({"query": query, "api_key": "demo-value"}))
                        .func(Arc::new(move |arguments| {
                            let callback_calls = Arc::clone(&callback_calls);
                            Box::pin(async move {
                                callback_calls.fetch_add(1, Ordering::SeqCst);
                                assert_eq!(arguments["api_key"], "demo-value");
                                Ok(ToolExecutionResult::new(json!({"query": arguments["query"]})))
                            })
                        })).build()).await;
                    flush_subscribers().unwrap();
                    if query.is_empty() {
                        assert!(result.unwrap_err().to_string().contains("search.query is required"));
                        assert_eq!(calls.load(Ordering::SeqCst), 0);
                    } else {
                        assert_eq!(result.unwrap().result, json!({"query": query}));
                        assert_eq!(calls.load(Ordering::SeqCst), 1);
                        if combined { assert_eq!(paired_call(&records.lock().unwrap(), "tool", "search")["data"]["api_key"], "<redacted>"); }
                    }
                }
                if combined {
                    let execution = tool_call_execute(ToolCallExecuteParams::builder()
                        .name("search")
                        .args(json!({"query": "weather"}))
                        .func(Arc::new(|_arguments| Box::pin(async {
                            Ok(ToolExecutionResult {
                                result: json!({"ok": true}),
                                annotation: Some(json!({"verification": "preserved"})),
                            })
                        })))
                        .build()).await.unwrap();
                    assert_eq!(execution.annotation, Some(json!({"verification": "preserved"})));
                }
                deregister_tool_conditional_execution_guardrail("search.require_query").unwrap();
                if combined {
                    deregister_tool_sanitize_request_guardrail("search.redact_api_key").unwrap();
                    deregister_tool_execution_intercept("search.measure").unwrap();
                }
            }
        });
        register_subscriber("documentation-json", Arc::new(printer::log_event)).unwrap();
        nemo_relay::scope::event(
            nemo_relay::scope::EmitMarkEventParams::builder()
                .name("documentation-check")
                .data(json!({"sample": "synthetic"}))
                .build(),
        )
        .unwrap();
        flush_subscribers().unwrap();
        deregister_subscriber("documentation-json").unwrap();
        deregister_subscriber("documentation-capture").unwrap();
    }
}
