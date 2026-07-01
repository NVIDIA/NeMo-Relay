// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal neutral inspection contract primitives and Relay adapters for the
//! cross-repo POC.

#![cfg_attr(not(test), allow(dead_code))]

use crate::api::llm::LlmRequest;
use crate::error::{FlowError, Result};
use crate::json::Json;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) enum InspectionTarget {
    LlmRequest {
        provider: String,
        request: Json,
    },
    ToolRequest {
        tool_name: String,
        input: Json,
    },
    HttpRequest {
        method: String,
        path: String,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct InspectionContext {
    pub sandbox_id: Option<String>,
    pub scope_id: Option<String>,
    pub provider: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Finding {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) enum InspectionDecision {
    Allow,
    Deny {
        reason: String,
        findings: Vec<Finding>,
    },
    Mutate {
        target: InspectionTarget,
        findings: Vec<Finding>,
    },
}

pub(crate) trait Inspector: Send + Sync {
    fn inspect(
        &self,
        target: InspectionTarget,
        ctx: &InspectionContext,
    ) -> Result<InspectionDecision>;
}

pub(crate) struct RelayInspectionAdapter<I> {
    inspector: I,
}

impl<I> RelayInspectionAdapter<I>
where
    I: Inspector,
{
    pub(crate) fn new(inspector: I) -> Self {
        Self { inspector }
    }

    pub(crate) fn inspect_llm_request(
        &self,
        provider: &str,
        request: LlmRequest,
        ctx: &InspectionContext,
    ) -> Result<LlmRequest> {
        let request_value = serde_json::to_value(&request)
            .map_err(|error| FlowError::InvalidArgument(error.to_string()))?;
        let decision = self.inspector.inspect(
            InspectionTarget::LlmRequest {
                provider: provider.to_string(),
                request: request_value,
            },
            ctx,
        )?;

        match decision {
            InspectionDecision::Allow => Ok(request),
            InspectionDecision::Deny { reason, .. } => Err(FlowError::GuardrailRejected(reason)),
            InspectionDecision::Mutate { target, .. } => match target {
                InspectionTarget::LlmRequest { request, .. } => serde_json::from_value(request)
                    .map_err(|error| FlowError::InvalidArgument(error.to_string())),
                other => Err(FlowError::InvalidArgument(format!(
                    "expected mutated LlmRequest target, got {other:?}"
                ))),
            },
        }
    }

    pub(crate) fn inspect_tool_request(
        &self,
        tool_name: &str,
        input: Json,
        ctx: &InspectionContext,
    ) -> Result<Json> {
        let decision = self.inspector.inspect(
            InspectionTarget::ToolRequest {
                tool_name: tool_name.to_string(),
                input: input.clone(),
            },
            ctx,
        )?;

        match decision {
            InspectionDecision::Allow => Ok(input),
            InspectionDecision::Deny { reason, .. } => Err(FlowError::GuardrailRejected(reason)),
            InspectionDecision::Mutate { target, .. } => match target {
                InspectionTarget::ToolRequest { input, .. } => Ok(input),
                other => Err(FlowError::InvalidArgument(format!(
                    "expected mutated ToolRequest target, got {other:?}"
                ))),
            },
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/inspection_tests.rs"]
mod tests;
