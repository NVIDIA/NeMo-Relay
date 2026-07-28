// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared tool data types.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use crate::Json;
use crate::api::event::PendingMarkSpec;

/// Category-profile key used to expose a tool-result annotation on lifecycle events.
pub const TOOL_RESULT_ANNOTATION_PROFILE_KEY: &str = "tool_result_annotation";
/// JSON-envelope schema for an opaque tool execution frame.
pub const TOOL_EXECUTION_FRAME_SCHEMA: &str = "nemo.relay.ToolExecutionFrame@1";
/// JSON-envelope schema for a frame-aware tool execution intercept outcome.
pub const TOOL_EXECUTION_FRAME_OUTCOME_SCHEMA: &str = "nemo.relay.ToolExecutionFrameOutcome@1";

bitflags! {
    /// Bitflags that modify tool-call behavior and observability.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ToolAttributes: u32 {
        /// Marks the tool as executing out-of-process.
        const REMOTE = 0b01;
    }
}

/// Relay-owned wrapper returned by a raw-result tool execution intercept.
///
/// `result` is passed to the remaining middleware and application. `pending_marks`
/// are Relay-owned lifecycle metadata retained separately and emitted after the
/// tool-end event; they are not included in the application-visible result.
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

/// Tool result plus an optional opaque annotation for Relay interception.
///
/// The raw [`Self::result`] remains the application-visible value. The
/// annotation is carried only as adjacent middleware and lifecycle context;
/// Relay does not define or interpret the schema of either value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolExecutionFrame {
    /// Raw application-owned tool result.
    pub result: Json,
    /// Optional application-supplied result annotation, opaque to Relay.
    #[serde(
        default,
        deserialize_with = "deserialize_annotation",
        skip_serializing_if = "annotation_is_absent"
    )]
    pub annotation: Option<Json>,
}

fn deserialize_annotation<'de, D>(deserializer: D) -> Result<Option<Json>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<Json>::deserialize(deserializer).map(normalize_annotation)
}

fn annotation_is_absent(annotation: &Option<Json>) -> bool {
    annotation.as_ref().is_none_or(Json::is_null)
}

fn normalize_annotation(annotation: Option<Json>) -> Option<Json> {
    annotation.filter(|value| !value.is_null())
}

impl ToolExecutionFrame {
    /// Create a frame without an annotation.
    #[must_use]
    pub fn new(result: Json) -> Self {
        Self {
            result,
            annotation: None,
        }
    }

    /// Create a frame carrying an opaque annotation.
    #[must_use]
    pub fn annotated(result: Json, annotation: Json) -> Self {
        Self {
            result,
            annotation: normalize_annotation(Some(annotation)),
        }
    }

    /// Attach or replace the opaque annotation.
    #[must_use]
    pub fn with_annotation(mut self, annotation: Json) -> Self {
        self.annotation = normalize_annotation(Some(annotation));
        self
    }

    /// Remove any annotation.
    #[must_use]
    pub fn without_annotation(mut self) -> Self {
        self.annotation = None;
        self
    }

    /// Normalize JSON `null` to the frame's absent-annotation representation.
    ///
    /// Direct struct construction can produce `Some(Json::Null)` even though
    /// JSON deserialization maps both a missing field and `null` to `None`.
    /// Relay calls this at middleware boundaries to keep in-memory and wire
    /// semantics stable.
    #[must_use]
    pub fn normalized(mut self) -> Self {
        self.annotation = normalize_annotation(self.annotation);
        self
    }
}

impl From<Json> for ToolExecutionFrame {
    fn from(result: Json) -> Self {
        Self::new(result)
    }
}

/// Result returned by an annotation-aware tool execution intercept.
///
/// Pending marks remain Relay-owned lifecycle metadata and are not exposed
/// through the annotation-aware continuation, matching the existing v1
/// execution-intercept behavior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolExecutionFrameOutcome {
    /// Raw result and optional opaque annotation returned to upstream middleware.
    pub frame: ToolExecutionFrame,
    /// Ordered marks for the managed tool lifecycle owner to emit.
    #[serde(default)]
    pub pending_marks: Vec<PendingMarkSpec>,
}

impl ToolExecutionFrameOutcome {
    /// Create an outcome without pending marks.
    #[must_use]
    pub fn new(frame: ToolExecutionFrame) -> Self {
        Self {
            frame: frame.normalized(),
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

impl From<ToolExecutionFrame> for ToolExecutionFrameOutcome {
    fn from(frame: ToolExecutionFrame) -> Self {
        Self::new(frame)
    }
}
