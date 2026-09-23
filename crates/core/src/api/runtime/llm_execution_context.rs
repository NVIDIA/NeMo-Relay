// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Invocation-scoped codec context for LLM execution intercepts.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use super::callbacks::{LlmRequestCodecContext, LlmResponseCodecContext};
use crate::api::llm::LlmRequest;
use crate::codec::request::AnnotatedLlmRequest;
use crate::codec::response::AnnotatedLlmResponse;
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use crate::error::{FlowError, Result};
use crate::json::Json;

const INACTIVE_EXECUTION_CODEC_ERROR: &str = "LLM execution codec capability is no longer active";

fn inactive_execution_codec_error() -> FlowError {
    FlowError::InvalidArgument(INACTIVE_EXECUTION_CODEC_ERROR.into())
}

fn upgrade_active_codec<T: ?Sized>(codec: &Weak<T>, gate: &ExecutionCodecGate) -> Result<Arc<T>> {
    // Upgrade first so a call admitted by the gate keeps the codec alive until
    // its synchronous operation completes, even if the lease then expires.
    let codec = codec.upgrade().ok_or_else(inactive_execution_codec_error)?;
    gate.ensure_active()?;
    Ok(codec)
}

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
            Err(inactive_execution_codec_error())
        }
    }

    fn revoke(&self) {
        self.active.store(false, Ordering::Release);
    }
}

/// Revokes codec capabilities issued to one execution-intercept invocation.
pub(crate) struct LlmExecutionCodecLeaseGuard {
    gate: Arc<ExecutionCodecGate>,
    request_codec: Option<Arc<dyn LlmCodec>>,
    response_codec: Option<Arc<dyn LlmResponseCodec>>,
}

impl Drop for LlmExecutionCodecLeaseGuard {
    fn drop(&mut self) {
        self.gate.revoke();
        drop(self.request_codec.take());
        drop(self.response_codec.take());
    }
}

struct RevocableRequestCodec {
    codec: Weak<dyn LlmCodec>,
    identity: super::LlmCodecIdentity,
    gate: Arc<ExecutionCodecGate>,
}

impl LlmCodec for RevocableRequestCodec {
    fn codec_identity(&self) -> super::LlmCodecIdentity {
        self.identity.clone()
    }

    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        upgrade_active_codec(&self.codec, &self.gate)?.decode(request)
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        upgrade_active_codec(&self.codec, &self.gate)?.encode(annotated, original)
    }
}

struct RevocableResponseCodec {
    codec: Weak<dyn LlmResponseCodec>,
    identity: super::LlmCodecIdentity,
    gate: Arc<ExecutionCodecGate>,
}

impl LlmResponseCodec for RevocableResponseCodec {
    fn codec_identity(&self) -> super::LlmCodecIdentity {
        self.identity.clone()
    }

    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        upgrade_active_codec(&self.codec, &self.gate)?.decode_response(response)
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
    request_codec: LlmRequestCodecContext,
    response_codec: Option<LlmResponseCodecContext>,
}

impl LlmExecutionContext {
    /// Construct an execution context from its directional codec contexts.
    #[must_use]
    pub fn new(
        request_codec: LlmRequestCodecContext,
        response_codec: Option<LlmResponseCodecContext>,
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
            LlmRequestCodecContext::for_request_codec(request_codec),
            Some(LlmResponseCodecContext::for_response_codec(
                response_codec.clone(),
            )),
        )
    }

    /// Construct the context for a streaming managed execution.
    pub(crate) fn for_streaming_codec(request_codec: Option<Arc<dyn LlmCodec>>) -> Self {
        Self::new(
            LlmRequestCodecContext::for_request_codec(request_codec),
            None,
        )
    }

    /// Issue revocable codec facades for one execution-intercept invocation.
    ///
    /// The source context retains Relay's selected codecs, but callbacks only
    /// receive the facades created here. Dropping the returned guard makes all
    /// retained facade clones fail and releases the lease's strong codec
    /// references.
    pub(crate) fn lease(&self) -> (Self, LlmExecutionCodecLeaseGuard) {
        let gate = Arc::new(ExecutionCodecGate::new());
        let leased_request_codec = self.request_codec.resolve_codec();
        let request_codec = match leased_request_codec.as_ref() {
            Some(codec) => {
                LlmRequestCodecContext::for_request_codec(Some(Arc::new(RevocableRequestCodec {
                    codec: Arc::downgrade(codec),
                    identity: self.request_codec.codec().clone(),
                    gate: Arc::clone(&gate),
                })))
            }
            None => LlmRequestCodecContext::with_identity(self.request_codec.codec().clone()),
        };
        let leased_response_codec = self
            .response_codec
            .as_ref()
            .and_then(LlmResponseCodecContext::resolve_codec);
        let response_codec =
            self.response_codec
                .as_ref()
                .map(|context| match leased_response_codec.as_ref() {
                    Some(codec) => LlmResponseCodecContext::for_response_codec(Some(Arc::new(
                        RevocableResponseCodec {
                            codec: Arc::downgrade(codec),
                            identity: context.codec().clone(),
                            gate: Arc::clone(&gate),
                        },
                    ))),
                    None => LlmResponseCodecContext::with_identity(context.codec().clone()),
                });

        (
            Self::new(request_codec, response_codec),
            LlmExecutionCodecLeaseGuard {
                gate,
                request_codec: leased_request_codec,
                response_codec: leased_response_codec,
            },
        )
    }

    /// Return the request-direction codec identity and revocable capability.
    #[must_use]
    pub fn request_codec(&self) -> &LlmRequestCodecContext {
        &self.request_codec
    }

    /// Return the unary response-direction codec identity and revocable capability.
    ///
    /// Streaming execution returns `None` because Relay does not expose a
    /// completed-response codec for individual stream chunks.
    #[must_use]
    pub fn response_codec(&self) -> Option<&LlmResponseCodecContext> {
        self.response_codec.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::runtime::{BuiltinLlmCodec, LlmCodecIdentity};

    struct DropProbeCodec;

    impl LlmCodec for DropProbeCodec {
        fn codec_identity(&self) -> LlmCodecIdentity {
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
        }

        fn decode(&self, _request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
            unreachable!("the facade must reject access after lease expiry")
        }

        fn encode(
            &self,
            _annotated: &AnnotatedLlmRequest,
            _original: &LlmRequest,
        ) -> Result<LlmRequest> {
            unreachable!("the facade must reject access after lease expiry")
        }
    }

    impl LlmResponseCodec for DropProbeCodec {
        fn codec_identity(&self) -> LlmCodecIdentity {
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
        }

        fn decode_response(&self, _response: &Json) -> Result<AnnotatedLlmResponse> {
            unreachable!("the facade must reject access after lease expiry")
        }
    }

    #[test]
    fn retained_facades_do_not_keep_backing_codec_alive_after_lease_expiry() {
        let backing = Arc::new(DropProbeCodec);
        let backing_probe = Arc::downgrade(&backing);
        let request_codec: Arc<dyn LlmCodec> = backing.clone();
        let response_codec: Arc<dyn LlmResponseCodec> = backing.clone();
        let context =
            LlmExecutionContext::for_unary_codecs(Some(request_codec), &Some(response_codec));
        drop(backing);

        let (leased_context, guard) = context.lease();
        let retained_request = leased_context.request_codec().resolve_codec().unwrap();
        let retained_response = leased_context
            .response_codec()
            .and_then(LlmResponseCodecContext::resolve_codec)
            .unwrap();
        drop(leased_context);
        drop(context);

        assert!(backing_probe.upgrade().is_some());

        drop(guard);

        assert!(backing_probe.upgrade().is_none());
        assert_eq!(
            retained_request.codec_identity(),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
        );
        assert_eq!(
            retained_response.codec_identity(),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
        );
        assert!(matches!(
            retained_request.decode(&LlmRequest {
                headers: serde_json::Map::new(),
                content: Json::Null,
            }),
            Err(FlowError::InvalidArgument(message))
                if message == INACTIVE_EXECUTION_CODEC_ERROR
        ));
        assert!(matches!(
            retained_response.decode_response(&Json::Null),
            Err(FlowError::InvalidArgument(message))
                if message == INACTIVE_EXECUTION_CODEC_ERROR
        ));
    }
}
