// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::codec::request::AnnotatedLLMRequest;
use crate::codec::response::AnnotatedLLMResponse;
use crate::json::Json;
use crate::types::llm::LLMAttributes;
use crate::types::scope::{HandleAttributes, ScopeAttributes, ScopeType};
use crate::types::tool::ToolAttributes;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScopeStartEvent {
    pub parent_uuid: Option<Uuid>,
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub data: Option<Json>,
    pub metadata: Option<Json>,
    pub attributes: ScopeAttributes,
    pub scope_type: ScopeType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScopeEndEvent {
    pub parent_uuid: Option<Uuid>,
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub data: Option<Json>,
    pub metadata: Option<Json>,
    pub attributes: ScopeAttributes,
    pub scope_type: ScopeType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolStartEvent {
    pub parent_uuid: Option<Uuid>,
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub data: Option<Json>,
    pub metadata: Option<Json>,
    pub attributes: ToolAttributes,
    pub input: Option<Json>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolEndEvent {
    pub parent_uuid: Option<Uuid>,
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub data: Option<Json>,
    pub metadata: Option<Json>,
    pub attributes: ToolAttributes,
    pub output: Option<Json>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LLMStartEvent {
    pub parent_uuid: Option<Uuid>,
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub data: Option<Json>,
    pub metadata: Option<Json>,
    pub attributes: LLMAttributes,
    pub input: Option<Json>,
    pub model_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub annotated_request: Option<Arc<AnnotatedLLMRequest>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LLMEndEvent {
    pub parent_uuid: Option<Uuid>,
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub data: Option<Json>,
    pub metadata: Option<Json>,
    pub attributes: LLMAttributes,
    pub output: Option<Json>,
    pub model_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub annotated_response: Option<Arc<AnnotatedLLMResponse>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarkEvent {
    pub parent_uuid: Option<Uuid>,
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub name: String,
    pub data: Option<Json>,
    pub metadata: Option<Json>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Event {
    ScopeStart(ScopeStartEvent),
    ScopeEnd(ScopeEndEvent),
    ToolStart(ToolStartEvent),
    ToolEnd(ToolEndEvent),
    LLMStart(LLMStartEvent),
    LLMEnd(LLMEndEvent),
    Mark(MarkEvent),
}

impl Event {
    fn now() -> DateTime<Utc> {
        Utc::now()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn scope_start(
        parent_uuid: Option<Uuid>,
        uuid: Uuid,
        name: impl Into<String>,
        data: Option<Json>,
        metadata: Option<Json>,
        attributes: ScopeAttributes,
        scope_type: ScopeType,
    ) -> Self {
        Self::ScopeStart(ScopeStartEvent {
            parent_uuid,
            uuid,
            timestamp: Self::now(),
            name: name.into(),
            data,
            metadata,
            attributes,
            scope_type,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn scope_end(
        parent_uuid: Option<Uuid>,
        uuid: Uuid,
        name: impl Into<String>,
        data: Option<Json>,
        metadata: Option<Json>,
        attributes: ScopeAttributes,
        scope_type: ScopeType,
    ) -> Self {
        Self::ScopeEnd(ScopeEndEvent {
            parent_uuid,
            uuid,
            timestamp: Self::now(),
            name: name.into(),
            data,
            metadata,
            attributes,
            scope_type,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tool_start(
        parent_uuid: Option<Uuid>,
        uuid: Uuid,
        name: impl Into<String>,
        data: Option<Json>,
        metadata: Option<Json>,
        attributes: ToolAttributes,
        input: Option<Json>,
        tool_call_id: Option<String>,
    ) -> Self {
        Self::ToolStart(ToolStartEvent {
            parent_uuid,
            uuid,
            timestamp: Self::now(),
            name: name.into(),
            data,
            metadata,
            attributes,
            input,
            tool_call_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tool_end(
        parent_uuid: Option<Uuid>,
        uuid: Uuid,
        name: impl Into<String>,
        data: Option<Json>,
        metadata: Option<Json>,
        attributes: ToolAttributes,
        output: Option<Json>,
        tool_call_id: Option<String>,
    ) -> Self {
        Self::ToolEnd(ToolEndEvent {
            parent_uuid,
            uuid,
            timestamp: Self::now(),
            name: name.into(),
            data,
            metadata,
            attributes,
            output,
            tool_call_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn llm_start(
        parent_uuid: Option<Uuid>,
        uuid: Uuid,
        name: impl Into<String>,
        data: Option<Json>,
        metadata: Option<Json>,
        attributes: LLMAttributes,
        input: Option<Json>,
        model_name: Option<String>,
        annotated_request: Option<Arc<AnnotatedLLMRequest>>,
    ) -> Self {
        Self::LLMStart(LLMStartEvent {
            parent_uuid,
            uuid,
            timestamp: Self::now(),
            name: name.into(),
            data,
            metadata,
            attributes,
            input,
            model_name,
            annotated_request,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn llm_end(
        parent_uuid: Option<Uuid>,
        uuid: Uuid,
        name: impl Into<String>,
        data: Option<Json>,
        metadata: Option<Json>,
        attributes: LLMAttributes,
        output: Option<Json>,
        model_name: Option<String>,
        annotated_response: Option<Arc<AnnotatedLLMResponse>>,
    ) -> Self {
        Self::LLMEnd(LLMEndEvent {
            parent_uuid,
            uuid,
            timestamp: Self::now(),
            name: name.into(),
            data,
            metadata,
            attributes,
            output,
            model_name,
            annotated_response,
        })
    }

    pub fn mark(
        parent_uuid: Option<Uuid>,
        uuid: Uuid,
        name: impl Into<String>,
        data: Option<Json>,
        metadata: Option<Json>,
    ) -> Self {
        Self::Mark(MarkEvent {
            parent_uuid,
            uuid,
            timestamp: Self::now(),
            name: name.into(),
            data,
            metadata,
        })
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::ScopeStart(_) => "ScopeStart",
            Self::ScopeEnd(_) => "ScopeEnd",
            Self::ToolStart(_) => "ToolStart",
            Self::ToolEnd(_) => "ToolEnd",
            Self::LLMStart(_) => "LLMStart",
            Self::LLMEnd(_) => "LLMEnd",
            Self::Mark(_) => "Mark",
        }
    }

    pub fn parent_uuid(&self) -> Option<Uuid> {
        match self {
            Self::ScopeStart(event) => event.parent_uuid,
            Self::ScopeEnd(event) => event.parent_uuid,
            Self::ToolStart(event) => event.parent_uuid,
            Self::ToolEnd(event) => event.parent_uuid,
            Self::LLMStart(event) => event.parent_uuid,
            Self::LLMEnd(event) => event.parent_uuid,
            Self::Mark(event) => event.parent_uuid,
        }
    }

    pub fn uuid(&self) -> Uuid {
        match self {
            Self::ScopeStart(event) => event.uuid,
            Self::ScopeEnd(event) => event.uuid,
            Self::ToolStart(event) => event.uuid,
            Self::ToolEnd(event) => event.uuid,
            Self::LLMStart(event) => event.uuid,
            Self::LLMEnd(event) => event.uuid,
            Self::Mark(event) => event.uuid,
        }
    }

    pub fn timestamp(&self) -> &DateTime<Utc> {
        match self {
            Self::ScopeStart(event) => &event.timestamp,
            Self::ScopeEnd(event) => &event.timestamp,
            Self::ToolStart(event) => &event.timestamp,
            Self::ToolEnd(event) => &event.timestamp,
            Self::LLMStart(event) => &event.timestamp,
            Self::LLMEnd(event) => &event.timestamp,
            Self::Mark(event) => &event.timestamp,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::ScopeStart(event) => &event.name,
            Self::ScopeEnd(event) => &event.name,
            Self::ToolStart(event) => &event.name,
            Self::ToolEnd(event) => &event.name,
            Self::LLMStart(event) => &event.name,
            Self::LLMEnd(event) => &event.name,
            Self::Mark(event) => &event.name,
        }
    }

    pub fn data(&self) -> Option<&Json> {
        match self {
            Self::ScopeStart(event) => event.data.as_ref(),
            Self::ScopeEnd(event) => event.data.as_ref(),
            Self::ToolStart(event) => event.data.as_ref(),
            Self::ToolEnd(event) => event.data.as_ref(),
            Self::LLMStart(event) => event.data.as_ref(),
            Self::LLMEnd(event) => event.data.as_ref(),
            Self::Mark(event) => event.data.as_ref(),
        }
    }

    pub fn metadata(&self) -> Option<&Json> {
        match self {
            Self::ScopeStart(event) => event.metadata.as_ref(),
            Self::ScopeEnd(event) => event.metadata.as_ref(),
            Self::ToolStart(event) => event.metadata.as_ref(),
            Self::ToolEnd(event) => event.metadata.as_ref(),
            Self::LLMStart(event) => event.metadata.as_ref(),
            Self::LLMEnd(event) => event.metadata.as_ref(),
            Self::Mark(event) => event.metadata.as_ref(),
        }
    }

    pub fn attributes(&self) -> Option<HandleAttributes> {
        match self {
            Self::ScopeStart(event) => Some(HandleAttributes::Scope(event.attributes)),
            Self::ScopeEnd(event) => Some(HandleAttributes::Scope(event.attributes)),
            Self::ToolStart(event) => Some(HandleAttributes::Tool(event.attributes)),
            Self::ToolEnd(event) => Some(HandleAttributes::Tool(event.attributes)),
            Self::LLMStart(event) => Some(HandleAttributes::Llm(event.attributes)),
            Self::LLMEnd(event) => Some(HandleAttributes::Llm(event.attributes)),
            Self::Mark(_) => None,
        }
    }

    pub fn scope_type(&self) -> Option<ScopeType> {
        match self {
            Self::ScopeStart(event) => Some(event.scope_type),
            Self::ScopeEnd(event) => Some(event.scope_type),
            Self::ToolStart(_)
            | Self::ToolEnd(_)
            | Self::LLMStart(_)
            | Self::LLMEnd(_)
            | Self::Mark(_) => None,
        }
    }

    pub fn input(&self) -> Option<&Json> {
        match self {
            Self::ToolStart(event) => event.input.as_ref(),
            Self::LLMStart(event) => event.input.as_ref(),
            _ => None,
        }
    }

    pub fn output(&self) -> Option<&Json> {
        match self {
            Self::ToolEnd(event) => event.output.as_ref(),
            Self::LLMEnd(event) => event.output.as_ref(),
            _ => None,
        }
    }

    pub fn model_name(&self) -> Option<&str> {
        match self {
            Self::LLMStart(event) => event.model_name.as_deref(),
            Self::LLMEnd(event) => event.model_name.as_deref(),
            _ => None,
        }
    }

    pub fn tool_call_id(&self) -> Option<&str> {
        match self {
            Self::ToolStart(event) => event.tool_call_id.as_deref(),
            Self::ToolEnd(event) => event.tool_call_id.as_deref(),
            _ => None,
        }
    }
}
