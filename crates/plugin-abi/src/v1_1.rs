// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native API v1.1: an additive tool-context lane over the frozen v1.0 baseline.
//!
//! All baseline ownership and continuation types retain their original meanings.
//! Application execution context is a separate value, never business arguments.

pub use crate::handles_v1_1::*;
use crate::{Json, future::FfiFuture, v1_0 as a};

/// One tool intercept invocation, including optional application assertions.
#[repr(C)]
pub struct ToolInvocation {
    /// Unchanged baseline invocation, including business arguments and scope.
    pub invocation: a::Invocation,
    /// Invocation-local application context; structural validity grants no authority.
    pub execution_context: Option<Json>,
}

/// Tool execution callback supporting the structured application context.
pub trait ToolCallback: Send + Sync {
    /// Invoke with isolated context and the unchanged baseline continuation.
    fn invoke_tool(
        &self,
        invocation: ToolInvocation,
        next: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::ToolExecutionOutcome>>;
}

/// Component-owned tool execution registration applied by the activation transaction.
#[repr(C)]
pub struct ToolRegistration {
    /// Component-local registration name.
    pub name: String,
    /// Middleware ordering priority.
    pub priority: i32,
    /// Callback owner released with the component.
    pub callback: ToolCallbackBox,
}

/// Component-scoped runtime authority with the additive tool registration lane.
pub trait Runtime: Send + Sync {
    /// Obtain the same baseline authority, without widening its lifetime or permissions.
    fn baseline(&self) -> a::Result<a::RuntimeBox>;
    /// Register context-aware tool middleware through the existing host transaction.
    fn register_tool_execution(&self, registration: ToolRegistration) -> a::Result<()>;
}

/// Plugin definition built against native API v1.1.
pub trait Plugin: Send + Sync {
    /// Whether the plugin permits multiple configured components.
    fn allows_multiple_components(&self) -> bool;
    /// Validate one component without installing callbacks.
    fn validate(&self, config: Json) -> Vec<a::ConfigDiagnostic>;
    /// Register with the additive runtime; cleanup ownership remains baseline v1.0.
    fn register(
        &self,
        config: Json,
        runtime: RuntimeBox,
    ) -> FfiFuture<a::Result<a::RegisteredComponent>>;
}

/// Exact module name for native v1.1 factory discovery.
pub const FACTORY_MODULE: &str = "nemo.relay.native.factory.v1.1";

/// Construct a definition only after the complete declared contract is checked.
pub trait Factory: Send + Sync {
    /// Create one plugin definition before admitting component work.
    fn create(&self) -> a::Result<PluginBox>;
}

/// Preserve a baseline author's source API on a v1.1 build SDK.
///
/// Baseline callbacks retain baseline behavior, including rejecting supplied
/// execution context. Authors use the v1.1 SDK tool lane to receive that context.
pub struct LegacyPlugin(pub a::PluginBox);
impl Plugin for LegacyPlugin {
    fn allows_multiple_components(&self) -> bool {
        self.0.allows_multiple_components()
    }
    fn validate(&self, config: Json) -> Vec<a::ConfigDiagnostic> {
        self.0.validate(config)
    }
    fn register(
        &self,
        config: Json,
        runtime: RuntimeBox,
    ) -> FfiFuture<a::Result<a::RegisteredComponent>> {
        use crate::future::FutureExt as _;
        match runtime.baseline() {
            Ok(baseline) => self.0.register(config, baseline),
            Err(error) => async move { Err(error) }.into_ffi(),
        }
    }
}
