// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use uuid::Uuid;

use crate::codec::request::AnnotatedLLMRequest;
use crate::codec::traits::LlmCodec;
use crate::context::callbacks::EventSubscriberFn;
use crate::context::global::global_context;
use crate::context::scope_stack::{current_scope_stack, task_scope_top};
use crate::error::{FlowError, Result};
use crate::shared_runtime::ensure_process_runtime_owner;
use crate::types::llm::LLMRequest;
use crate::types::scope::ScopeHandle;

pub(crate) fn resolve_parent_uuid(parent: Option<&ScopeHandle>) -> Option<Uuid> {
    Some(
        parent
            .map(|handle| handle.uuid)
            .unwrap_or_else(|| task_scope_top().uuid),
    )
}

pub(crate) fn snapshot_event_subscribers(
    scope_local_subscribers: Vec<EventSubscriberFn>,
) -> Result<Vec<EventSubscriberFn>> {
    let context = global_context();
    let state = context
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    Ok(state.collect_event_subscribers(&scope_local_subscribers))
}

pub(crate) fn ensure_runtime_owner() -> Result<()> {
    ensure_process_runtime_owner()
}

pub(crate) fn run_request_intercepts_with_codec(
    name: &str,
    request: LLMRequest,
    codec: Option<Arc<dyn LlmCodec>>,
) -> Result<(LLMRequest, Option<Arc<AnnotatedLLMRequest>>)> {
    let scope_stack = current_scope_stack();
    let scope_guard = scope_stack.read().expect("scope stack lock poisoned");
    let scope_locals =
        scope_guard.collect_scope_local_registries(|registries| &registries.llm_request_intercepts);

    let context = global_context();
    let state = context
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;

    let original = request.clone();
    let annotated = match &codec {
        Some(codec) => Some(codec.decode(&request)?),
        None => None,
    };

    let (intercepted_request, intercepted_annotated) =
        state.llm_request_intercepts_chain(name, request, annotated, &scope_locals)?;

    match (codec, intercepted_annotated) {
        (Some(codec), Some(annotated)) => {
            let mut encoded = codec.encode(&annotated, &original)?;
            encoded.headers = intercepted_request.headers;
            Ok((encoded, Some(Arc::new(annotated))))
        }
        _ => Ok((intercepted_request, None)),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/shared_tests.rs"]
mod tests;
