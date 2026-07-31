// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Invocation-scoped transport target for managed LLM continuations.

use std::collections::BTreeMap;
use std::future::Future;

tokio::task_local! {
    static TASK_LLM_DISPATCH_TARGET: LlmDispatchTargetContext;
}

/// Validated provider transport target bound to one LLM continuation invocation.
///
/// This internal bridge keeps transport data out of [`crate::api::llm::LlmRequest`]
/// while allowing the terminal gateway callback to dispatch a plugin-selected
/// provider request.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmDispatchTargetContext {
    method: String,
    url: String,
    route: String,
    headers: BTreeMap<String, String>,
}

impl LlmDispatchTargetContext {
    /// Construct a validated target from the native-plugin adapter.
    #[doc(hidden)]
    #[must_use]
    pub fn new(
        method: String,
        url: String,
        route: String,
        headers: BTreeMap<String, String>,
    ) -> Self {
        Self {
            method,
            url,
            route,
            headers,
        }
    }

    /// HTTP method selected for this invocation.
    #[doc(hidden)]
    #[must_use]
    pub fn method(&self) -> &str {
        &self.method
    }

    /// Absolute provider URL selected for this invocation.
    #[doc(hidden)]
    #[must_use]
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Relay provider-route identifier selected for this invocation.
    #[doc(hidden)]
    #[must_use]
    pub fn route(&self) -> &str {
        &self.route
    }

    /// Explicit provider headers selected for this invocation.
    #[doc(hidden)]
    #[must_use]
    pub fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }
}

/// Return the typed target bound to the current continuation invocation.
#[doc(hidden)]
#[must_use]
pub fn current_llm_dispatch_target() -> Option<LlmDispatchTargetContext> {
    TASK_LLM_DISPATCH_TARGET.try_with(Clone::clone).ok()
}

/// Poll a future with one typed target bound to its continuation invocation.
pub(crate) async fn scope_llm_dispatch_target<F: Future>(
    target: LlmDispatchTargetContext,
    future: F,
) -> F::Output {
    TASK_LLM_DISPATCH_TARGET.scope(target, future).await
}
