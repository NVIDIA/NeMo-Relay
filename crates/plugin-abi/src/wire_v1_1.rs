// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Additive checked interfaces. The v1.0 wire module is unchanged.
#![allow(missing_docs, non_camel_case_types)] // xabi generates public plumbing.
#![allow(clippy::type_complexity)] // The wire contract spells out all owned replies.
use crate::{
    Json,
    future::FfiFuture,
    transfer::{Unit, Value},
    v1_0 as a,
    v1_1::*,
};

#[xabi::xabi(id = "nemo.relay.native.v1.1.ToolCallback", version = 1)]
pub trait WireToolCallback {
    fn invoke_tool(
        &self,
        invocation: Value<ToolInvocation>,
        next: Value<a::ContinuationBox>,
    ) -> std::result::Result<Value<FfiFuture<a::Result<a::ToolExecutionOutcome>>>, Value<a::Error>>;
}
#[xabi::xabi(id = "nemo.relay.native.v1.1.Runtime", version = 1)]
pub trait WireRuntime {
    fn baseline(&self) -> std::result::Result<Value<a::RuntimeBox>, Value<a::Error>>;
    fn register_tool_execution(
        &self,
        registration: Value<ToolRegistration>,
    ) -> std::result::Result<Value<Unit>, Value<a::Error>>;
}
#[xabi::xabi(id = "nemo.relay.native.v1.1.Plugin", version = 1)]
pub trait WirePlugin {
    fn allows_multiple_components(&self) -> std::result::Result<Value<bool>, Value<a::Error>>;
    fn validate(
        &self,
        config: Value<Json>,
    ) -> std::result::Result<Value<Vec<a::ConfigDiagnostic>>, Value<a::Error>>;
    fn register(
        &self,
        config: Value<Json>,
        runtime: Value<RuntimeBox>,
    ) -> std::result::Result<Value<FfiFuture<a::Result<a::RegisteredComponent>>>, Value<a::Error>>;
}
#[xabi::xabi(id = "nemo.relay.native.v1.1.Factory", version = 1)]
pub trait WireFactory {
    fn create(&self) -> std::result::Result<Value<PluginBox>, Value<a::Error>>;
}
