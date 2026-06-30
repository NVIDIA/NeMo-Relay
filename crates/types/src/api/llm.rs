// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared LLM data types.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use crate::Json;
use crate::api::event::PendingMarkSpec;
use crate::codec::request::AnnotatedLlmRequest;

bitflags! {
    /// Bitflags that modify LLM-call behavior and observability.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct LlmAttributes: u32 {
        /// Marks the request as stateful from the runtime's perspective.
        const STATEFUL = 0b01;
        /// Marks the request as streaming.
        const STREAMING = 0b10;
    }
}

/// JSON-shaped LLM request payload passed through the runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmRequest {
    /// Provider-specific request headers.
    pub headers: serde_json::Map<String, Json>,
    /// Provider-specific request body.
    pub content: Json,
}

/// Result of an LLM request intercept that can schedule lifecycle marks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmRequestInterceptOutcome {
    /// Rewritten provider request when no request codec is active.
    ///
    /// With a request codec, callbacks may rewrite `headers`, but `content`
    /// is read-only and provider-body changes must be made through
    /// [`Self::annotated_request`].
    pub request: LlmRequest,
    /// Optional normalized request annotation to carry forward.
    ///
    /// This is required and authoritative for provider content when a request
    /// codec is active. It remains optional when no request codec is active.
    #[serde(default)]
    pub annotated_request: Option<AnnotatedLlmRequest>,
    /// Ordered marks to emit after Relay creates and starts the LLM scope.
    #[serde(default)]
    pub pending_marks: Vec<PendingMarkSpec>,
}

impl LlmRequestInterceptOutcome {
    /// Create an outcome without pending marks.
    pub fn new(request: LlmRequest, annotated_request: Option<AnnotatedLlmRequest>) -> Self {
        Self {
            request,
            annotated_request,
            pending_marks: Vec::new(),
        }
    }

    /// Append one pending mark while preserving interceptor order.
    #[must_use]
    pub fn with_pending_mark(mut self, mark: PendingMarkSpec) -> Self {
        self.pending_marks.push(mark);
        self
    }
}
