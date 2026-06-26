// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared LLM data types.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use crate::Json;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    /// Provider-specific request headers.
    pub headers: serde_json::Map<String, Json>,
    /// Provider-specific request body.
    pub content: Json,
}
