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
