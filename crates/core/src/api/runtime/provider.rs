// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-local provider execution supplied by an embedding host.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub use nemo_relay_types::api::provider::{LlmProviderFormat, LlmProviderRequest};

use crate::api::runtime::LlmJsonStream;
use crate::error::Result;
use crate::json::Json;

/// Host callback for a buffered call to an authorized provider target.
pub type LlmProviderCallFn = Arc<
    dyn Fn(LlmProviderRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync,
>;

/// Host callback for a streamed call to an authorized provider target.
pub type LlmProviderStreamFn = Arc<
    dyn Fn(LlmProviderRequest) -> Pin<Box<dyn Future<Output = Result<LlmJsonStream>> + Send>>
        + Send
        + Sync,
>;

/// Private request-scoped provider execution supplied by the host.
///
/// Callbacks must authorize every target, keep credentials out of returned
/// values and errors, disable credential-bearing redirects, and propagate
/// cancellation by dropping pending I/O. The runtime never serializes these
/// callbacks or places them in events. Native plugins access them only through
/// their live execution continuation.
#[derive(Clone)]
pub struct LlmProviderDispatcher {
    pub(crate) call: LlmProviderCallFn,
    pub(crate) stream: LlmProviderStreamFn,
}

impl LlmProviderDispatcher {
    /// Create a dispatcher bound to one inbound request's credentials and policy.
    pub fn new(call: LlmProviderCallFn, stream: LlmProviderStreamFn) -> Self {
        Self { call, stream }
    }
}

tokio::task_local! {
    static PROVIDER_DISPATCHER: Option<LlmProviderDispatcher>;
}

/// Run a managed LLM invocation with a private host provider dispatcher.
///
/// Each concurrent request must supply its own dispatcher. Ordinary LLM
/// execution remains unchanged when no dispatcher is installed.
pub async fn with_llm_provider_dispatcher<F: Future>(
    dispatcher: LlmProviderDispatcher,
    future: F,
) -> F::Output {
    scope_provider_dispatcher(Some(dispatcher), future).await
}

pub(crate) fn current_provider_dispatcher() -> Option<LlmProviderDispatcher> {
    PROVIDER_DISPATCHER.try_with(Clone::clone).ok().flatten()
}

pub(crate) async fn scope_provider_dispatcher<F: Future>(
    dispatcher: Option<LlmProviderDispatcher>,
    future: F,
) -> F::Output {
    PROVIDER_DISPATCHER.scope(dispatcher, future).await
}
