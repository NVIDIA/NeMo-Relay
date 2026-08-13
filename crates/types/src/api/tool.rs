// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared tool data types.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use crate::Json;
use crate::api::event::PendingMarkSpec;

/// Versioned native-plugin JSON envelope schema for [`ToolExecutionResult`].
pub const TOOL_EXECUTION_RESULT_SCHEMA: &str = "nemo.relay.ToolExecutionResult@1";

/// Versioned native-plugin JSON envelope schema for [`ToolExecutionInterceptOutcome`].
pub const TOOL_EXECUTION_INTERCEPT_OUTCOME_SCHEMA: &str =
    "nemo.relay.ToolExecutionInterceptOutcome@2";

fn normalize_annotation(annotation: Option<Json>) -> Option<Json> {
    annotation.filter(|value| !value.is_null())
}

fn normalized_annotation(annotation: &Option<Json>) -> Option<&Json> {
    annotation.as_ref().filter(|value| !value.is_null())
}

fn annotation_is_absent(annotation: &Option<Json>) -> bool {
    normalized_annotation(annotation).is_none()
}

bitflags! {
    /// Bitflags that modify tool-call behavior and observability.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ToolAttributes: u32 {
        /// Marks the tool as executing out-of-process.
        const REMOTE = 0b01;
    }
}

/// Canonical application-visible result of tool execution.
///
/// Relay transports `result` and `annotation` without interpreting either
/// value. The annotation is an adjacent metadata channel and does not impose a
/// schema on the application-owned result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionResult {
    /// Application-owned tool result.
    pub result: Json,
    /// Optional opaque metadata associated with the result.
    #[serde(default, skip_serializing_if = "annotation_is_absent")]
    pub annotation: Option<Json>,
}

impl ToolExecutionResult {
    /// Create a result without an annotation.
    pub fn new(result: Json) -> Self {
        Self {
            result,
            annotation: None,
        }
    }

    /// Create a result with an opaque annotation.
    pub fn annotated(result: Json, annotation: Json) -> Self {
        Self {
            result,
            annotation: normalize_annotation(Some(annotation)),
        }
    }

    /// Replace the opaque annotation while preserving the application result.
    #[must_use]
    pub fn with_annotation(mut self, annotation: Json) -> Self {
        self.annotation = normalize_annotation(Some(annotation));
        self
    }

    /// Remove the annotation while preserving the application result.
    #[must_use]
    pub fn without_annotation(mut self) -> Self {
        self.annotation = None;
        self
    }
}

impl PartialEq for ToolExecutionResult {
    fn eq(&self, other: &Self) -> bool {
        self.result == other.result
            && normalized_annotation(&self.annotation) == normalized_annotation(&other.annotation)
    }
}

impl From<Json> for ToolExecutionResult {
    fn from(result: Json) -> Self {
        Self::new(result)
    }
}

/// Canonical result returned by a tool execution intercept.
///
/// `result` and `annotation` are returned to the remaining middleware and
/// application. `pending_marks` are Relay-owned lifecycle metadata retained
/// separately and emitted after the tool-end event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionInterceptOutcome {
    /// Tool result returned to the remaining middleware and application.
    pub result: Json,
    /// Optional opaque metadata associated with the result.
    #[serde(default, skip_serializing_if = "annotation_is_absent")]
    pub annotation: Option<Json>,
    /// Ordered marks for the managed tool lifecycle owner to emit.
    #[serde(default)]
    pub pending_marks: Vec<PendingMarkSpec>,
}

impl ToolExecutionInterceptOutcome {
    /// Create an outcome without pending marks.
    pub fn new(result: Json) -> Self {
        Self {
            result,
            annotation: None,
            pending_marks: Vec::new(),
        }
    }

    /// Create an outcome with an opaque annotation and no pending marks.
    pub fn annotated(result: Json, annotation: Json) -> Self {
        Self {
            result,
            annotation: normalize_annotation(Some(annotation)),
            pending_marks: Vec::new(),
        }
    }

    /// Replace the opaque annotation while preserving the application result
    /// and pending marks.
    #[must_use]
    pub fn with_annotation(mut self, annotation: Json) -> Self {
        self.annotation = normalize_annotation(Some(annotation));
        self
    }

    /// Remove the annotation while preserving the application result and
    /// pending marks.
    #[must_use]
    pub fn without_annotation(mut self) -> Self {
        self.annotation = None;
        self
    }

    /// Convert the intercept outcome to its application-visible result.
    pub fn into_execution_result(self) -> ToolExecutionResult {
        ToolExecutionResult {
            result: self.result,
            annotation: normalize_annotation(self.annotation),
        }
    }

    /// Append one pending mark while preserving callback order.
    #[must_use]
    pub fn with_pending_mark(mut self, mark: PendingMarkSpec) -> Self {
        self.pending_marks.push(mark);
        self
    }
}

impl PartialEq for ToolExecutionInterceptOutcome {
    fn eq(&self, other: &Self) -> bool {
        self.result == other.result
            && normalized_annotation(&self.annotation) == normalized_annotation(&other.annotation)
            && self.pending_marks == other.pending_marks
    }
}

impl From<Json> for ToolExecutionInterceptOutcome {
    fn from(result: Json) -> Self {
        Self::new(result)
    }
}

impl From<ToolExecutionResult> for ToolExecutionInterceptOutcome {
    fn from(result: ToolExecutionResult) -> Self {
        Self {
            result: result.result,
            annotation: normalize_annotation(result.annotation),
            pending_marks: Vec::new(),
        }
    }
}
