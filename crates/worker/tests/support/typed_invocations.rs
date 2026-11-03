// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Domain fixture builders for the active typed worker services.
use super::*;

#[derive(Clone, Default)]
pub(crate) struct InvocationFixture {
    pub credentials: Option<pb::Credentials>,
    pub invocation_id: String,
    pub registration_name: String,
    pub surface: i32,
    pub continuation_id: String,
    pub scope: Option<ScopeContext>,
    pub payload: Option<Payload>,
}
#[derive(Clone)]
pub(crate) enum Payload {
    Event(Box<pb::Event>),
    Tool(ToolInvocation),
    Llm(Box<LlmInvocation>),
    ConditionalMiddleware(pb::ConditionalMiddlewareInvocation),
}
#[derive(Clone)]
pub(crate) struct ToolInvocation {
    pub tool_name: String,
    pub value: Option<pb::JsonValue>,
    pub tool_call_id: Option<String>,
}
#[derive(Clone)]
pub(crate) struct LlmInvocation {
    pub model_name: String,
    pub request: Option<pb::LlmRequest>,
    pub annotated_request: Option<pb::AnnotatedLlmRequest>,
    pub response: Option<pb::JsonValue>,
    pub sanitize_context: Option<SanitizeContext>,
    pub execution_codec_context: Option<Box<ExecutionCodecContext>>,
}
#[derive(Clone)]
pub(crate) enum SanitizeContext {
    RequestSanitizeContext(pb::LlmCodecContext),
    ResponseSanitizeContext(pb::LlmCodecContext),
}
#[derive(Clone)]
pub(crate) struct ExecutionCodecContext {
    pub request: Option<pb::LlmCodecContext>,
    pub response: Option<pb::LlmCodecContext>,
}

fn missing(field: &str) -> Status {
    Status::invalid_argument(format!("missing {field}"))
}
impl InvocationFixture {
    pub fn into_wire(self) -> std::result::Result<pb::InvokeRequest, Status> {
        use RegistrationSurface as S;
        use pb::invoke_request::Invocation as I;
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
                    I::ToolExecutionIntercept(pb::ToolExecutionInvocation {
                        tool_name: v.tool_name,
                        args: v.value,
                        tool_call_id: v.tool_call_id,
                        continuation_id: self.continuation_id,
                    })
                } else {
                    let value = pb::ToolValueInvocation {
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
                    I::LlmSanitizeRequestGuardrail(pb::LlmSanitizeRequestInvocation {
                        request: v.request,
                        context: match v.sanitize_context {
                            Some(SanitizeContext::RequestSanitizeContext(v)) => Some(v),
                            _ => Some(empty_codec()),
                        },
                    })
                }
                S::LlmSanitizeResponseGuardrail => {
                    I::LlmSanitizeResponseGuardrail(pb::LlmSanitizeResponseInvocation {
                        response: v.response,
                        context: match v.sanitize_context {
                            Some(SanitizeContext::ResponseSanitizeContext(v)) => Some(v),
                            _ => Some(empty_codec()),
                        },
                    })
                }
                S::LlmConditionalExecutionGuardrail => {
                    I::LlmConditionalExecutionGuardrail(pb::LlmConditionalInvocation {
                        request: v.request,
                    })
                }
                S::LlmRequestIntercept => I::LlmRequestIntercept(pb::LlmRequestInvocation {
                    model_name: v.model_name,
                    request: v.request,
                    annotated_request: v.annotated_request.map(Box::new),
                }),
                S::LlmExecutionIntercept => {
                    let contexts = v
                        .execution_codec_context
                        .ok_or_else(|| missing("execution codec context"))?;
                    I::LlmExecutionIntercept(pb::LlmExecutionInvocation {
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
        Ok(pb::InvokeRequest {
            credentials: self.credentials,
            invocation_id: self.invocation_id,
            registration_name: self.registration_name,
            scope: self.scope,
            invocation: Some(invocation),
        })
    }
    pub fn into_stream_wire(self) -> std::result::Result<pb::InvokeStreamRequest, Status> {
        let Some(Payload::Llm(v)) = self.payload else {
            return Err(missing("stream LLM payload"));
        };
        if self.surface != RegistrationSurface::LlmStreamExecutionIntercept as i32 {
            return Err(missing("stream surface"));
        }
        let contexts = v
            .execution_codec_context
            .ok_or_else(|| missing("stream request codec"))?;
        Ok(pb::InvokeStreamRequest {
            credentials: self.credentials,
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

fn empty_codec() -> pb::LlmCodecContext {
    pb::LlmCodecContext {
        codec: Some(pb::LlmCodecReference {
            kind: "none".into(),
            id: None,
        }),
        capability_id: None,
    }
}

#[derive(Clone)]
pub(crate) struct PluginWorkerClient<T>(
    pb::plugin_worker_service_client::PluginWorkerServiceClient<T>,
);
impl<T> std::ops::Deref for PluginWorkerClient<T> {
    type Target = pb::plugin_worker_service_client::PluginWorkerServiceClient<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> std::ops::DerefMut for PluginWorkerClient<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl PluginWorkerClient<Channel> {
    pub fn wire(
        &mut self,
    ) -> &mut pb::plugin_worker_service_client::PluginWorkerServiceClient<Channel> {
        &mut self.0
    }
    pub fn new(channel: Channel) -> Self {
        Self(pb::plugin_worker_service_client::PluginWorkerServiceClient::new(channel))
    }
    pub async fn connect(endpoint: String) -> std::result::Result<Self, tonic::transport::Error> {
        Ok(Self(
            pb::plugin_worker_service_client::PluginWorkerServiceClient::connect(endpoint).await?,
        ))
    }
    pub async fn invoke(
        &mut self,
        request: Request<InvocationFixture>,
    ) -> std::result::Result<Response<pb::InvokeResponse>, Status> {
        let (metadata, extensions, message) = request.into_parts();
        self.0
            .invoke(Request::from_parts(
                metadata,
                extensions,
                message.into_wire()?,
            ))
            .await
    }
    pub async fn invoke_stream(
        &mut self,
        request: Request<InvocationFixture>,
    ) -> std::result::Result<Response<tonic::Streaming<pb::InvokeStreamResponse>>, Status> {
        let (metadata, extensions, message) = request.into_parts();
        self.0
            .invoke_stream(Request::from_parts(
                metadata,
                extensions,
                message.into_stream_wire()?,
            ))
            .await
    }
}
