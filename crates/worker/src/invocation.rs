// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Normalize the surface-specific wire alternatives for the private dispatcher.
use super::*;

pub(super) struct CallbackInvocation {
    pub invocation_id: String,
    pub registration_name: String,
    pub scope: Option<ScopeContext>,
    pub surface: i32,
    pub continuation_id: String,
    pub payload: Option<CallbackPayload>,
}

pub(super) enum CallbackPayload {
    Event(Box<wire::Event>),
    Tool(ToolPayload),
    Llm(Box<LlmPayload>),
    ConditionalMiddleware(wire::ConditionalMiddlewareInvocation),
}

impl TryFrom<wire::InvokeRequest> for CallbackInvocation {
    type Error = WorkerSdkError;
    fn try_from(value: wire::InvokeRequest) -> Result<Self> {
        use RegistrationSurface as S;
        use wire::invoke_request::Invocation as I;
        let mut continuation_id = String::new();
        let (surface, payload) = match value
            .invocation
            .ok_or_else(|| WorkerSdkError::InvalidInput("missing invocation alternative".into()))?
        {
            I::Subscriber(v) => (S::Subscriber, CallbackPayload::Event(Box::new(v))),
            I::EventMetadataInjector(v) => (
                S::EventMetadataInjector,
                CallbackPayload::Event(Box::new(v)),
            ),
            I::MarkSanitizeGuardrail(v) => (
                S::MarkSanitizeGuardrail,
                CallbackPayload::Event(Box::new(v)),
            ),
            I::ScopeSanitizeStartGuardrail(v) => (
                S::ScopeSanitizeStartGuardrail,
                CallbackPayload::Event(Box::new(v)),
            ),
            I::ScopeSanitizeEndGuardrail(v) => (
                S::ScopeSanitizeEndGuardrail,
                CallbackPayload::Event(Box::new(v)),
            ),
            I::ToolSanitizeRequestGuardrail(v) => (S::ToolSanitizeRequestGuardrail, tool(v)?),
            I::ToolSanitizeResponseGuardrail(v) => (S::ToolSanitizeResponseGuardrail, tool(v)?),
            I::ToolConditionalExecutionGuardrail(v) => {
                (S::ToolConditionalExecutionGuardrail, tool(v)?)
            }
            I::ToolRequestIntercept(v) => (S::ToolRequestIntercept, tool(v)?),
            I::ToolExecutionIntercept(v) => {
                continuation_id = v.continuation_id;
                (
                    S::ToolExecutionIntercept,
                    CallbackPayload::Tool(ToolPayload {
                        tool_name: v.tool_name,
                        value: required_json::<Json>(v.args, "tool args")?,
                        tool_call_id: v.tool_call_id,
                    }),
                )
            }
            I::LlmSanitizeRequestGuardrail(v) => (
                S::LlmSanitizeRequestGuardrail,
                CallbackPayload::Llm(Box::new(LlmPayload {
                    request: v.request,
                    request_codec: v.context,
                    ..Default::default()
                })),
            ),
            I::LlmSanitizeResponseGuardrail(v) => (
                S::LlmSanitizeResponseGuardrail,
                CallbackPayload::Llm(Box::new(LlmPayload {
                    response: v.response,
                    response_codec: v.context,
                    ..Default::default()
                })),
            ),
            I::LlmConditionalExecutionGuardrail(v) => (
                S::LlmConditionalExecutionGuardrail,
                CallbackPayload::Llm(Box::new(LlmPayload {
                    request: v.request,
                    ..Default::default()
                })),
            ),
            I::LlmRequestIntercept(v) => (
                S::LlmRequestIntercept,
                CallbackPayload::Llm(Box::new(LlmPayload {
                    model_name: v.model_name,
                    request: v.request,
                    annotated_request: v.annotated_request.map(|v| *v),
                    ..Default::default()
                })),
            ),
            I::LlmExecutionIntercept(v) => {
                continuation_id = v.continuation_id;
                (
                    S::LlmExecutionIntercept,
                    CallbackPayload::Llm(Box::new(LlmPayload {
                        model_name: v.model_name,
                        request: v.request,
                        request_codec: v.request_codec,
                        response_codec: v.response_codec,
                        ..Default::default()
                    })),
                )
            }
            I::ConditionalMiddlewareGuardrail(v) => (
                S::ConditionalMiddlewareGuardrail,
                CallbackPayload::ConditionalMiddleware(v),
            ),
        };
        Ok(Self {
            invocation_id: value.invocation_id,
            registration_name: value.registration_name,
            scope: value.scope,
            surface: surface as i32,
            continuation_id,
            payload: Some(payload),
        })
    }
}

impl From<wire::InvokeStreamRequest> for CallbackInvocation {
    fn from(v: wire::InvokeStreamRequest) -> Self {
        Self {
            invocation_id: v.invocation_id,
            registration_name: v.registration_name,
            scope: v.scope,
            surface: RegistrationSurface::LlmStreamExecutionIntercept as i32,
            continuation_id: v.continuation_id,
            payload: Some(CallbackPayload::Llm(Box::new(LlmPayload {
                model_name: v.model_name,
                request: v.request,
                request_codec: v.request_codec,
                ..Default::default()
            }))),
        }
    }
}

fn tool(v: wire::ToolValueInvocation) -> Result<CallbackPayload> {
    Ok(CallbackPayload::Tool(ToolPayload {
        tool_name: v.tool_name,
        value: required_json::<Json>(v.value, "tool value")?,
        tool_call_id: None,
    }))
}
