// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared middleware callback contracts and core registry aliases.

use crate::{api::tool::ToolExecutionInterceptOutcome, error::Result, json::Json};
pub use nemo_relay_types::api::runtime::callbacks::*;
use std::{future::Future, pin::Pin, sync::Arc};

/// Scope-local registry references passed into streaming execution-chain builders.
///
/// # Returns
/// A shared reference to a scope-local streaming execution registry.
pub(crate) type LlmStreamExecutionRegistryRef<'a> = &'a crate::registry::SortedRegistry<
    crate::api::registry::ExecutionIntercept<LlmStreamExecutionFn>,
>;
/// Slice of scope-local streaming execution registries.
///
/// # Returns
/// A borrowed slice of scope-local streaming execution registry references.
pub(crate) type LlmStreamExecutionRegistryRefs<'a> = &'a [LlmStreamExecutionRegistryRef<'a>];

pub(crate) type ToolExecutionOutcomeNextFn = Arc<
    dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<ToolExecutionInterceptOutcome>> + Send>>
        + Send
        + Sync,
>;
pub(crate) type LlmStreamResponseCompletionFn = Box<
    dyn for<'a> FnOnce(&'a Json) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> + Send,
>;
