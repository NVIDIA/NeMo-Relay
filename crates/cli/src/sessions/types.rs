// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session gateway inputs and prepared-call outputs.

#[cfg(test)]
use nemo_relay::api::llm::LlmHandle;
use nemo_relay::api::llm::{LlmAttributes, LlmRequest};
use nemo_relay::api::runtime::ScopeStackHandle;
use nemo_relay::api::scope::ScopeHandle;
use serde_json::Value;

#[derive(Debug, Clone)]
pub(crate) struct LlmGatewayStart {
    pub(crate) session_id: Option<String>,
    pub(crate) provider: String,
    pub(crate) model_name: Option<String>,
    pub(crate) subagent_id: Option<String>,
    pub(crate) conversation_id: Option<String>,
    pub(crate) generation_id: Option<String>,
    pub(crate) request_id: Option<String>,
    pub(crate) request: LlmRequest,
    pub(crate) streaming: bool,
    pub(crate) metadata: Value,
}

/// Legacy active-LLM record retained for manual-correlation tests.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct ActiveLlm {
    pub(super) stack: ScopeStackHandle,
    pub(super) handle: LlmHandle,
    pub(super) session_id: String,
    pub(super) owner_subagent_id: Option<String>,
}

/// Inputs for invoking managed LLM execution after releasing the session lock.
pub(crate) struct GatewayCallPrep {
    pub(crate) scope_stack: ScopeStackHandle,
    pub(crate) session_id: String,
    pub(crate) provider_name: String,
    pub(crate) request: LlmRequest,
    pub(crate) parent: Option<ScopeHandle>,
    pub(crate) attributes: LlmAttributes,
    pub(crate) metadata: Value,
    pub(crate) model_name: Option<String>,
    pub(crate) owner_subagent_id: Option<String>,
    pub(crate) bypass_managed_pipeline: bool,
    pub(crate) session_finish: GatewaySessionFinish,
}

/// Cleanup policy for the session selected by one gateway request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GatewaySessionFinish {
    Retain,
    PruneIfEmpty,
    Close,
}
