// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Invocation-scoped codec context for LLM execution intercepts.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::callbacks::{LlmSanitizeRequestContext, LlmSanitizeResponseContext};
use crate::api::llm::LlmRequest;
use crate::codec::request::AnnotatedLlmRequest;
use crate::codec::response::AnnotatedLlmResponse;
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use crate::error::{FlowError, Result};
use crate::json::Json;

const INACTIVE_EXECUTION_CODEC_ERROR: &str = "LLM execution codec capability is no longer active";

#[derive(Debug)]
struct ExecutionCodecGate {
    active: AtomicBool,
}

impl ExecutionCodecGate {
    fn new() -> Self {
        Self {
            active: AtomicBool::new(true),
        }
    }

    fn ensure_active(&self) -> Result<()> {
        if self.active.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(FlowError::InvalidArgument(
                INACTIVE_EXECUTION_CODEC_ERROR.into(),
            ))
        }
    }

    fn revoke(&self) {
        self.active.store(false, Ordering::Release);
    }
}

/// Revokes codec capabilities issued to one execution-intercept invocation.
pub(crate) struct LlmExecutionCodecLeaseGuard {
    gate: Arc<ExecutionCodecGate>,
}

impl Drop for LlmExecutionCodecLeaseGuard {
    fn drop(&mut self) {
        self.gate.revoke();
    }
}

struct RevocableRequestCodec {
    codec: Arc<dyn LlmCodec>,
    gate: Arc<ExecutionCodecGate>,
}

impl LlmCodec for RevocableRequestCodec {
    fn codec_identity(&self) -> super::LlmCodecIdentity {
        self.codec.codec_identity()
    }

    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        self.gate.ensure_active()?;
        self.codec.decode(request)
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        self.gate.ensure_active()?;
        self.codec.encode(annotated, original)
    }
}

struct RevocableResponseCodec {
    codec: Arc<dyn LlmResponseCodec>,
    gate: Arc<ExecutionCodecGate>,
}

impl LlmResponseCodec for RevocableResponseCodec {
    fn codec_identity(&self) -> super::LlmCodecIdentity {
        self.codec.codec_identity()
    }

    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        self.gate.ensure_active()?;
        self.codec.decode_response(response)
    }
}

/// Active request and response codec context for one managed LLM execution.
///
/// The request direction is always present and distinguishes an invocation
/// with no request codec from an invocation with a built-in, runtime, or opaque
/// codec. Unary execution also carries a response direction. Streaming
/// execution deliberately leaves [`Self::response_codec`] unavailable because
/// Relay's response codecs operate on complete provider responses rather than
/// individual stream chunks.
///
/// The codecs are fixed when the managed invocation is created. Rewriting a
/// payload does not select another codec; decoding or encoding an incompatible
/// wire representation fails rather than inferring a different format.
/// Resolved codec capabilities are valid only for the callback that received
/// this context. Unary capabilities expire when that callback settles;
/// streaming request capabilities remain valid until its returned stream ends
/// or closes. Retained capabilities return [`FlowError::InvalidArgument`]
/// after expiry.
#[derive(Clone, Debug, Default)]
pub struct LlmExecutionContext {
    request_codec: LlmSanitizeRequestContext,
    response_codec: Option<LlmSanitizeResponseContext>,
}

impl LlmExecutionContext {
    /// Construct an execution context from its directional codec contexts.
    #[must_use]
    pub fn new(
        request_codec: LlmSanitizeRequestContext,
        response_codec: Option<LlmSanitizeResponseContext>,
    ) -> Self {
        Self {
            request_codec,
            response_codec,
        }
    }

    /// Construct the context for a unary managed execution.
    pub(crate) fn for_unary_codecs(
        request_codec: Option<Arc<dyn LlmCodec>>,
        response_codec: &Option<Arc<dyn LlmResponseCodec>>,
    ) -> Self {
        Self::new(
            LlmSanitizeRequestContext::for_request_codec(request_codec),
            Some(LlmSanitizeResponseContext::for_response_codec(
                response_codec.clone(),
            )),
        )
    }

    /// Construct the context for a streaming managed execution.
    pub(crate) fn for_streaming_codec(request_codec: Option<Arc<dyn LlmCodec>>) -> Self {
        Self::new(
            LlmSanitizeRequestContext::for_request_codec(request_codec),
            None,
        )
    }

    /// Issue revocable codec facades for one execution-intercept invocation.
    ///
    /// The source context retains Relay's selected codecs, but callbacks only
    /// receive the facades created here. Dropping the returned guard makes all
    /// retained facade clones fail without exposing the underlying codec.
    pub(crate) fn lease(&self) -> (Self, LlmExecutionCodecLeaseGuard) {
        let gate = Arc::new(ExecutionCodecGate::new());
        let request_codec = match self.request_codec.resolve_codec() {
            Some(codec) => LlmSanitizeRequestContext::for_request_codec(Some(Arc::new(
                RevocableRequestCodec {
                    codec,
                    gate: Arc::clone(&gate),
                },
            ))),
            None => LlmSanitizeRequestContext::with_identity(self.request_codec.codec().clone()),
        };
        let response_codec =
            self.response_codec
                .as_ref()
                .map(|context| match context.resolve_codec() {
                    Some(codec) => LlmSanitizeResponseContext::for_response_codec(Some(Arc::new(
                        RevocableResponseCodec {
                            codec,
                            gate: Arc::clone(&gate),
                        },
                    ))),
                    None => LlmSanitizeResponseContext::with_identity(context.codec().clone()),
                });

        (
            Self::new(request_codec, response_codec),
            LlmExecutionCodecLeaseGuard { gate },
        )
    }

    /// Return the request-direction codec identity and revocable capability.
    #[must_use]
    pub fn request_codec(&self) -> &LlmSanitizeRequestContext {
        &self.request_codec
    }

    /// Return the unary response-direction codec identity and revocable capability.
    ///
    /// Streaming execution returns `None` because Relay does not expose a
    /// completed-response codec for individual stream chunks.
    #[must_use]
    pub fn response_codec(&self) -> Option<&LlmSanitizeResponseContext> {
        self.response_codec.as_ref()
    }
}
