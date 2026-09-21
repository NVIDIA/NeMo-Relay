// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Invocation-scoped codec context for internal LLM execution adapters.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::api::llm::LlmRequest;
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use crate::error::Result;
use crate::json::Json;

use super::callbacks::{
    LlmExecutionFn, LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionFn,
    LlmStreamExecutionNextFn,
};
#[cfg(feature = "worker-grpc")]
use super::callbacks::{LlmSanitizeRequestContext, LlmSanitizeResponseContext};

/// Active request and response codecs for one managed LLM execution.
///
/// Public execution-interceptor callbacks remain unchanged. Relay adapts them
/// into one private context-aware callback shape so language and process
/// bridges receive the active codecs explicitly.
///
/// The context describes the codecs selected when the managed invocation was
/// created. Execution interceptors may rewrite payloads within that codec's
/// contract, but changing the provider wire format does not select a new codec.
/// A subsequent decode or encode will reject an incompatible payload rather
/// than silently infer another codec.
#[derive(Clone)]
pub(crate) struct LlmExecutionCodecContext {
    #[cfg(feature = "worker-grpc")]
    request: LlmSanitizeRequestContext,
    #[cfg(feature = "worker-grpc")]
    response: LlmSanitizeResponseContext,
}

impl LlmExecutionCodecContext {
    #[cfg(feature = "worker-grpc")]
    pub(crate) fn new(
        request: LlmSanitizeRequestContext,
        response: LlmSanitizeResponseContext,
    ) -> Self {
        Self { request, response }
    }

    pub(crate) fn for_codecs(
        request_codec: Option<Arc<dyn LlmCodec>>,
        response_codec: &Option<Arc<dyn LlmResponseCodec>>,
    ) -> Self {
        #[cfg(feature = "worker-grpc")]
        {
            Self::new(
                LlmSanitizeRequestContext::for_request_codec(request_codec),
                LlmSanitizeResponseContext::for_response_codec(response_codec.clone()),
            )
        }
        #[cfg(not(feature = "worker-grpc"))]
        {
            let _ = (request_codec, response_codec);
            Self {}
        }
    }

    #[cfg(feature = "worker-grpc")]
    pub(crate) fn request(&self) -> &LlmSanitizeRequestContext {
        &self.request
    }

    #[cfg(feature = "worker-grpc")]
    pub(crate) fn response(&self) -> &LlmSanitizeResponseContext {
        &self.response
    }
}

/// Private non-streaming execution callback used by Relay's registries.
pub(crate) type ContextualLlmExecutionFn = Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmExecutionCodecContext,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
>;

/// Private streaming execution callback used by Relay's registries.
pub(crate) type ContextualLlmStreamExecutionFn = Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmExecutionCodecContext,
            LlmStreamExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<LlmJsonStream>> + Send>>
        + Send
        + Sync,
>;

/// Adapt the stable public callback into Relay's private context-aware shape.
pub(crate) fn adapt_llm_execution_fn(callback: LlmExecutionFn) -> ContextualLlmExecutionFn {
    Arc::new(move |name, request, _context, next| callback(name, request, next))
}

/// Adapt the stable public stream callback into Relay's private context-aware shape.
pub(crate) fn adapt_llm_stream_execution_fn(
    callback: LlmStreamExecutionFn,
) -> ContextualLlmStreamExecutionFn {
    Arc::new(move |name, request, _context, next| callback(name, request, next))
}
