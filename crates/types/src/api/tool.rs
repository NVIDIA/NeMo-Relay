// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared tool data types.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use crate::Json;
use crate::api::event::PendingMarkSpec;

bitflags! {
    /// Bitflags that modify tool-call behavior and observability.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ToolAttributes: u32 {
        /// Marks the tool as executing out-of-process.
        const REMOTE = 0b01;
    }
}

/// Result of a tool execution intercept that can schedule lifecycle marks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolExecutionInterceptOutcome {
    /// Tool result returned to the remaining middleware and application.
    pub result: Json,
    /// Ordered marks for the managed tool lifecycle owner to emit.
    #[serde(default)]
    pub pending_marks: Vec<PendingMarkSpec>,
}

impl ToolExecutionInterceptOutcome {
    /// Create an outcome without pending marks.
    pub fn new(result: Json) -> Self {
        Self {
            result,
            pending_marks: Vec::new(),
        }
    }

    /// Append one pending mark while preserving callback order.
    #[must_use]
    pub fn with_pending_mark(mut self, mark: PendingMarkSpec) -> Self {
        self.pending_marks.push(mark);
        self
    }
}

impl From<Json> for ToolExecutionInterceptOutcome {
    fn from(result: Json) -> Self {
        Self::new(result)
    }
}
