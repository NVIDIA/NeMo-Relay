// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private invocation preparation; live RPCs always use surface-specific wire types.
use super::*;

#[derive(Clone)]
pub(crate) struct PreparedInvocation {
    pub activation_id: String,
    pub auth_token: String,
    pub invocation_id: String,
    pub registration_name: String,
    pub surface: i32,
    pub continuation_id: String,
    pub scope: Option<ScopeContext>,
    pub payload: Option<Payload>,
}
#[derive(Clone)]
pub(crate) enum Payload {
    Event(Box<wire::Event>),
    Tool(ToolInvocation),
    Llm(Box<LlmInvocation>),
    ConditionalMiddleware(wire::ConditionalMiddlewareInvocation),
}
#[derive(Clone)]
pub(crate) struct ToolInvocation {
    pub tool_name: String,
    pub value: Option<wire::JsonValue>,
    pub tool_call_id: Option<String>,
}
#[derive(Clone)]
pub(crate) struct LlmInvocation {
    pub model_name: String,
    pub request: Option<wire::LlmRequest>,
    pub annotated_request: Option<wire::AnnotatedLlmRequest>,
    pub response: Option<wire::JsonValue>,
    pub sanitize_context: Option<SanitizeContext>,
    pub execution_codec_context: Option<Box<ExecutionCodecContext>>,
}
#[derive(Clone)]
pub(crate) enum SanitizeContext {
    RequestSanitizeContext(wire::LlmCodecContext),
    ResponseSanitizeContext(wire::LlmCodecContext),
}
#[derive(Clone)]
pub(crate) struct ExecutionCodecContext {
    pub request: Option<wire::LlmCodecContext>,
    pub response: Option<wire::LlmCodecContext>,
}

fn missing(field: &str) -> FlowError {
    FlowError::InvalidArgument(format!("missing {field}"))
}
impl PreparedInvocation {
    pub fn into_wire(self) -> FlowResult<wire::InvokeRequest> {
        use RegistrationSurface as S;
        use wire::invoke_request::Invocation as I;
        let surface = S::try_from(self.surface).map_err(|_| missing("valid surface"))?;
        let invocation = match self.payload.ok_or_else(|| missing("payload"))? {
            Payload::Event(v) => match surface {
                S::Subscriber => I::Subscriber(*v),
                S::EventMetadataInjector => I::EventMetadataInjector(*v),
                S::MarkSanitizeGuardrail => I::MarkSanitizeGuardrail(*v),
                S::ScopeSanitizeStartGuardrail => I::ScopeSanitizeStartGuardrail(*v),
                S::ScopeSanitizeEndGuardrail => I::ScopeSanitizeEndGuardrail(*v),
                _ => return Err(missing("matching event surface")),
            },
            Payload::Tool(v) => {
                if surface == S::ToolExecutionIntercept {
                    I::ToolExecutionIntercept(wire::ToolExecutionInvocation {
                        tool_name: v.tool_name,
                        args: v.value,
                        tool_call_id: v.tool_call_id,
                        continuation_id: self.continuation_id,
                    })
                } else {
                    let value = wire::ToolValueInvocation {
                        tool_name: v.tool_name,
                        value: v.value,
                    };
                    match surface {
                        S::ToolSanitizeRequestGuardrail => I::ToolSanitizeRequestGuardrail(value),
                        S::ToolSanitizeResponseGuardrail => I::ToolSanitizeResponseGuardrail(value),
                        S::ToolConditionalExecutionGuardrail => {
                            I::ToolConditionalExecutionGuardrail(value)
                        }
                        S::ToolRequestIntercept => I::ToolRequestIntercept(value),
                        _ => return Err(missing("matching tool surface")),
                    }
                }
            }
            Payload::Llm(v) => match surface {
                S::LlmSanitizeRequestGuardrail => {
                    I::LlmSanitizeRequestGuardrail(wire::LlmSanitizeRequestInvocation {
                        request: v.request,
                        context: match v.sanitize_context {
                            Some(SanitizeContext::RequestSanitizeContext(v)) => Some(v),
                            _ => return Err(missing("request sanitizer context")),
                        },
                    })
                }
                S::LlmSanitizeResponseGuardrail => {
                    I::LlmSanitizeResponseGuardrail(wire::LlmSanitizeResponseInvocation {
                        response: v.response,
                        context: match v.sanitize_context {
                            Some(SanitizeContext::ResponseSanitizeContext(v)) => Some(v),
                            _ => return Err(missing("response sanitizer context")),
                        },
                    })
                }
                S::LlmConditionalExecutionGuardrail => {
                    I::LlmConditionalExecutionGuardrail(wire::LlmConditionalInvocation {
                        request: v.request,
                    })
                }
                S::LlmRequestIntercept => I::LlmRequestIntercept(wire::LlmRequestInvocation {
                    model_name: v.model_name,
                    request: v.request,
                    annotated_request: v.annotated_request.map(Box::new),
                }),
                S::LlmExecutionIntercept => {
                    let contexts = v
                        .execution_codec_context
                        .ok_or_else(|| missing("execution codec context"))?;
                    I::LlmExecutionIntercept(wire::LlmExecutionInvocation {
                        model_name: v.model_name,
                        request: v.request,
                        continuation_id: self.continuation_id,
                        request_codec: contexts.request,
                        response_codec: contexts.response,
                    })
                }
                _ => return Err(missing("matching LLM surface")),
            },
            Payload::ConditionalMiddleware(v) if surface == S::ConditionalMiddlewareGuardrail => {
                I::ConditionalMiddlewareGuardrail(v)
            }
            _ => return Err(missing("matching callback surface")),
        };
        Ok(wire::InvokeRequest {
            credentials: Some(wire::Credentials {
                activation_id: self.activation_id,
                auth_token: self.auth_token,
            }),
            invocation_id: self.invocation_id,
            registration_name: self.registration_name,
            scope: self.scope,
            invocation: Some(invocation),
        })
    }
    pub fn into_stream_wire(self) -> FlowResult<wire::InvokeStreamRequest> {
        let Some(Payload::Llm(v)) = self.payload else {
            return Err(missing("stream LLM payload"));
        };
        if self.surface != RegistrationSurface::LlmStreamExecutionIntercept as i32 {
            return Err(missing("stream surface"));
        }
        let contexts = v
            .execution_codec_context
            .ok_or_else(|| missing("stream request codec"))?;
        Ok(wire::InvokeStreamRequest {
            credentials: Some(wire::Credentials {
                activation_id: self.activation_id,
                auth_token: self.auth_token,
            }),
            invocation_id: self.invocation_id,
            registration_name: self.registration_name,
            scope: self.scope,
            model_name: v.model_name,
            request: v.request,
            continuation_id: self.continuation_id,
            request_codec: contexts.request,
        })
    }
}
