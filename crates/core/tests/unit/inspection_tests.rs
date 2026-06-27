// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::api::llm::LlmRequest;
use crate::error::FlowError;
use crate::inspection::{
    Finding, InspectionContext, InspectionDecision, InspectionTarget, Inspector,
    RelayInspectionAdapter,
};
use crate::json::Json;
use serde_json::{Map, json};

struct TestInspector;

impl Inspector for TestInspector {
    fn inspect(
        &self,
        target: InspectionTarget,
        ctx: &InspectionContext,
    ) -> crate::error::Result<InspectionDecision> {
        match target {
            InspectionTarget::LlmRequest {
                provider,
                mut request,
            } => {
                if ctx.provider.as_deref() == Some("deny-llm") {
                    return Ok(InspectionDecision::Deny {
                        reason: format!("blocked provider {provider}"),
                        findings: vec![Finding {
                            code: "llm_denied".to_string(),
                            message: "provider rejected".to_string(),
                        }],
                    });
                }

                request["headers"]["x-inspected"] = json!("true");
                Ok(InspectionDecision::Mutate {
                    target: InspectionTarget::LlmRequest { provider, request },
                    findings: vec![Finding {
                        code: "llm_mutated".to_string(),
                        message: "request annotated".to_string(),
                    }],
                })
            }
            InspectionTarget::ToolRequest {
                tool_name,
                mut input,
            } => {
                if ctx.scope_id.as_deref() == Some("deny-tool") {
                    return Ok(InspectionDecision::Deny {
                        reason: format!("blocked tool {tool_name}"),
                        findings: vec![Finding {
                            code: "tool_denied".to_string(),
                            message: "tool rejected".to_string(),
                        }],
                    });
                }

                input["relay_inspected"] = json!(true);
                Ok(InspectionDecision::Mutate {
                    target: InspectionTarget::ToolRequest { tool_name, input },
                    findings: vec![Finding {
                        code: "tool_mutated".to_string(),
                        message: "tool args annotated".to_string(),
                    }],
                })
            }
            InspectionTarget::HttpRequest { .. } => Ok(InspectionDecision::Allow),
        }
    }
}

fn make_request() -> LlmRequest {
    LlmRequest {
        headers: Map::new(),
        content: json!({
            "messages": [{"role": "user", "content": "hello"}]
        }),
    }
}

#[test]
fn relay_inspection_adapter_mutates_llm_requests() {
    let adapter = RelayInspectionAdapter::new(TestInspector);
    let request = make_request();
    let ctx = InspectionContext {
        provider: Some("allow-llm".to_string()),
        ..InspectionContext::default()
    };

    let mutated = adapter
        .inspect_llm_request("openai", request, &ctx)
        .expect("inspection should succeed");

    assert_eq!(
        mutated.headers.get("x-inspected"),
        Some(&Json::String("true".to_string()))
    );
}

#[test]
fn relay_inspection_adapter_denies_llm_requests() {
    let adapter = RelayInspectionAdapter::new(TestInspector);
    let error = adapter
        .inspect_llm_request(
            "openai",
            make_request(),
            &InspectionContext {
                provider: Some("deny-llm".to_string()),
                ..InspectionContext::default()
            },
        )
        .expect_err("inspection should deny");

    match error {
        FlowError::GuardrailRejected(reason) => {
            assert!(reason.contains("blocked provider openai"));
        }
        other => panic!("expected guardrail rejection, got {other}"),
    }
}

#[test]
fn relay_inspection_adapter_mutates_tool_requests() {
    let adapter = RelayInspectionAdapter::new(TestInspector);
    let mutated = adapter
        .inspect_tool_request(
            "search",
            json!({"query": "books"}),
            &InspectionContext::default(),
        )
        .expect("inspection should succeed");

    assert_eq!(mutated["relay_inspected"], json!(true));
}

#[test]
fn relay_inspection_adapter_denies_tool_requests() {
    let adapter = RelayInspectionAdapter::new(TestInspector);
    let error = adapter
        .inspect_tool_request(
            "search",
            json!({"query": "books"}),
            &InspectionContext {
                scope_id: Some("deny-tool".to_string()),
                ..InspectionContext::default()
            },
        )
        .expect_err("inspection should deny");

    match error {
        FlowError::GuardrailRejected(reason) => {
            assert!(reason.contains("blocked tool search"));
        }
        other => panic!("expected guardrail rejection, got {other}"),
    }
}
