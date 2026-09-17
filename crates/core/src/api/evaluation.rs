// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral evaluation lifecycle and managed execution API.
//!
//! Evaluation is modeled as an evaluator scope, not as chat completion. Provider adapters can
//! translate these DTOs to their wire format while subscribers and event sanitizers observe a
//! stable operation independent of any one provider.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use crate::api::scope::{
    PopScopeParams, PushScopeParams, ScopeAttributes, ScopeHandle, ScopeType, pop_scope, push_scope,
};
use crate::api::shared::{metadata_with_otel_error, metadata_with_otel_status};
use crate::error::{FlowError, Result};
use crate::evaluation::{EvaluationRequest, EvaluationResponse};
use crate::json::Json;

/// Provider callback used by [`evaluation_execute`].
pub type EvaluationExecutionNextFn = Arc<
    dyn Fn(
            EvaluationRequest,
        ) -> Pin<Box<dyn Future<Output = Result<EvaluationResponse>> + Send + 'static>>
        + Send
        + Sync,
>;

/// Runtime-owned handle identifying an active evaluator scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationHandle {
    scope: ScopeHandle,
}

impl EvaluationHandle {
    /// Return the underlying scope handle for parenting and advanced runtime integration.
    pub fn scope(&self) -> &ScopeHandle {
        &self.scope
    }
}

/// Builder parameters for [`evaluation_call`].
#[derive(TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct EvaluationCallParams<'a> {
    /// Logical provider or evaluator name recorded on lifecycle events.
    pub name: &'a str,
    /// Provider-neutral request recorded as evaluator input.
    pub request: &'a EvaluationRequest,
    /// Optional explicit parent scope.
    #[builder(default)]
    pub parent: Option<&'a ScopeHandle>,
    /// Scope behavior flags.
    #[builder(default = ScopeAttributes::empty())]
    pub attributes: ScopeAttributes,
    /// Optional application payload stored on the handle.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional metadata recorded on the start event.
    #[builder(default)]
    pub metadata: Option<Json>,
}

/// Builder parameters for [`evaluation_call_end`].
#[derive(TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct EvaluationCallEndParams<'a> {
    /// Active evaluation handle to close.
    pub handle: &'a EvaluationHandle,
    /// Provider-neutral response recorded as evaluator output.
    pub response: &'a EvaluationResponse,
    /// Optional metadata merged onto the end event.
    #[builder(default)]
    pub metadata: Option<Json>,
}

/// Builder parameters for [`evaluation_execute`].
#[derive(TypedBuilder)]
#[builder(field_defaults(setter(strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct EvaluationExecuteParams {
    /// Logical provider or evaluator name recorded on lifecycle events.
    #[builder(setter(into))]
    pub name: String,
    /// Provider-neutral request supplied to the provider callback.
    pub request: EvaluationRequest,
    /// Provider callback or execution continuation.
    pub func: EvaluationExecutionNextFn,
    /// Optional explicit parent scope.
    #[builder(default)]
    pub parent: Option<ScopeHandle>,
    /// Scope behavior flags.
    #[builder(default = ScopeAttributes::empty())]
    pub attributes: ScopeAttributes,
    /// Optional application payload stored on the handle.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional metadata recorded on lifecycle events.
    #[builder(default)]
    pub metadata: Option<Json>,
}

/// Start a manual provider-neutral evaluation lifecycle.
pub fn evaluation_call(params: EvaluationCallParams<'_>) -> Result<EvaluationHandle> {
    let input = serde_json::to_value(params.request)
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let scope = push_scope(
        PushScopeParams::builder()
            .name(params.name)
            .scope_type(ScopeType::Evaluator)
            .parent_opt(params.parent)
            .attributes(params.attributes)
            .data_opt(params.data)
            .metadata_opt(params.metadata)
            .input(input)
            .build(),
    )?;
    Ok(EvaluationHandle { scope })
}

/// Finish a manual provider-neutral evaluation lifecycle.
pub fn evaluation_call_end(params: EvaluationCallEndParams<'_>) -> Result<()> {
    let output = serde_json::to_value(params.response)
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&params.handle.scope.uuid)
            .output(output)
            .metadata_opt(params.metadata)
            .build(),
    )
}

/// Execute a provider-neutral evaluation callback inside a complete evaluator lifecycle.
///
/// Event sanitizers run on both lifecycle events. Success and failure are represented with the
/// same OpenTelemetry status metadata used by the LLM and tool managed-execution APIs.
pub async fn evaluation_execute(params: EvaluationExecuteParams) -> Result<EvaluationResponse> {
    let handle = evaluation_call(
        EvaluationCallParams::builder()
            .name(&params.name)
            .request(&params.request)
            .parent_opt(params.parent.as_ref())
            .attributes(params.attributes)
            .data_opt(params.data)
            .metadata_opt(params.metadata.clone())
            .build(),
    )?;
    match (params.func)(params.request).await {
        Ok(response) => {
            evaluation_call_end(
                EvaluationCallEndParams::builder()
                    .handle(&handle)
                    .response(&response)
                    .metadata_opt(metadata_with_otel_status(params.metadata, "OK", None))
                    .build(),
            )?;
            Ok(response)
        }
        Err(error) => {
            pop_scope(
                PopScopeParams::builder()
                    .handle_uuid(&handle.scope.uuid)
                    .metadata_opt(metadata_with_otel_error(params.metadata, &error))
                    .build(),
            )?;
            Err(error)
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/evaluation_api_tests.rs"]
mod tests;
