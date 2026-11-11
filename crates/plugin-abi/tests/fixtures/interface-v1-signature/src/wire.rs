// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen xabi interfaces. Every call carries checked typed values.
#![allow(missing_docs, non_camel_case_types)] // xabi generates public interface plumbing.
#![allow(clippy::type_complexity)] // xabi requires explicit return types in generated contracts.
use crate::future::FfiFuture;
use crate::{Json, diagnostics::*, transfer::Value, v1_0::*};

#[xabi::xabi(id = "nemo.relay.native.v1.0.Scope", version = 1)]
pub trait WireScope {
    fn enter(&self) -> std::result::Result<Value<ScopeLeaseBox>, Value<Error>>;
    fn snapshot(&self) -> std::result::Result<Value<Option<Json>>, Value<Error>>;
}

#[xabi::xabi(id = "nemo.relay.native.v1.0.Codec", version = 1)]
pub trait WireCodec {
    fn identity(&self) -> std::result::Result<Value<Json>, Value<Error>>;
    fn decode(&self, value: Value<Json>) -> std::result::Result<Value<Json>, Value<Error>>;
    fn encode(
        &self,
        annotated: Value<Json>,
        original: Value<Json>,
    ) -> std::result::Result<Value<Json>, Value<Error>>;
}

#[xabi::xabi(id = "nemo.relay.native.v1.0.Continuation", version = 1)]
pub trait WireContinuation {
    fn call(
        &self,
        value: Value<Json>,
    ) -> std::result::Result<Value<FfiFuture<Result<Json>>>, Value<Error>>;
    fn call_tool(
        &self,
        value: Value<Json>,
    ) -> std::result::Result<Value<FfiFuture<Result<ToolExecutionResult>>>, Value<Error>>;
    fn open_stream(
        &self,
        value: Value<Json>,
    ) -> std::result::Result<Value<FfiFuture<Result<StreamBox>>>, Value<Error>>;
}

#[xabi::xabi(id = "nemo.relay.native.v1.0.Stream", version = 1)]
pub trait WireStream {
    fn pull(&self) -> std::result::Result<Value<FfiFuture<Result<Option<Json>>>>, Value<Error>>;
    fn close(
        &self,
    ) -> std::result::Result<Value<FfiFuture<Result<crate::transfer::Unit>>>, Value<Error>>;
}

#[xabi::xabi(id = "nemo.relay.native.v1.0.Callback", version = 1)]
pub trait WireCallback {
    fn invoke(
        &self,
        invocation: Value<Invocation>,
        next: Value<Option<ContinuationBox>>,
    ) -> std::result::Result<Value<FfiFuture<Result<Json>>>, Value<Error>>;
    fn invoke_tool(
        &self,
        invocation: Value<Invocation>,
        next: Value<ContinuationBox>,
    ) -> std::result::Result<Value<FfiFuture<Result<ToolExecutionOutcome>>>, Value<Error>>;
    fn invoke_sync(
        &self,
        invocation: Value<Invocation>,
    ) -> std::result::Result<Value<Json>, Value<Error>>;
    fn open_stream(
        &self,
        invocation: Value<Invocation>,
        next: Value<ContinuationBox>,
    ) -> std::result::Result<Value<FfiFuture<Result<StreamBox>>>, Value<Error>>;
}

#[xabi::xabi(id = "nemo.relay.native.v1.0.Component", version = 1)]
pub trait WireComponent {
    fn shutdown(
        &self,
    ) -> std::result::Result<Value<FfiFuture<Result<crate::transfer::Unit>>>, Value<Error>>;
}

#[xabi::xabi(id = "nemo.relay.native.v1.0.Runtime", version = 1)]
pub trait WireRuntime {
    fn call_sync(
        &self,
        operation: Value<String>,
        arguments: Value<Json>,
    ) -> std::result::Result<Value<Json>, Value<Error>>;
    fn call(
        &self,
        operation: Value<String>,
        arguments: Value<Json>,
    ) -> std::result::Result<Value<FfiFuture<Result<Json>>>, Value<Error>>;
    fn diagnostics(
        &self,
    ) -> std::result::Result<Value<crate::diagnostics::DiagnosticsBox>, Value<Error>>;
    fn capture_scope(&self) -> std::result::Result<Value<ScopeBox>, Value<Error>>;
    fn bind_component(
        &self,
        component: Value<ComponentBox>,
    ) -> std::result::Result<Value<crate::transfer::Unit>, Value<Error>>;
    fn register_callback(
        &self,
        registration: Value<Registration>,
    ) -> std::result::Result<Value<crate::transfer::Unit>, Value<Error>>;
    fn register_gate(
        &self,
        registration: Value<Registration>,
    ) -> std::result::Result<Value<String>, Value<Error>>;
}

#[xabi::xabi(id = "nemo.relay.native.v1.0.Plugin", version = 1)]
pub trait WirePlugin {
    fn allows_multiple_components(&self) -> std::result::Result<Value<u32>, Value<Error>>;
    fn validate(
        &self,
        config: Value<Json>,
    ) -> std::result::Result<Value<Vec<ConfigDiagnostic>>, Value<Error>>;
    fn register(
        &self,
        config: Value<Json>,
        runtime: Value<RuntimeBox>,
    ) -> std::result::Result<Value<FfiFuture<Result<RegisteredComponent>>>, Value<Error>>;
}

#[xabi::xabi(id = "nemo.relay.native.v1.0.Factory", version = 1)]
pub trait WireFactory {
    fn create(&self) -> std::result::Result<Value<PluginBox>, Value<Error>>;
}

use crate::borrow::{Output, Reply, SliceMut, Str};
#[xabi::xabi(id = "nemo.relay.native.v1.0.Diagnostics", version = 1)]
pub trait WireDiagnostics {
    fn limits(&self, output: Output<'_, Reply<DiagnosticCapabilities>>) -> xabi::Result<()>;
    fn register(
        &self,
        registration: DiagnosticRegistration<'_>,
        output: Output<'_, Reply<u64>>,
    ) -> xabi::Result<()>;
    fn record(
        &self,
        token: u64,
        observation: DiagnosticObservation<'_>,
        output: Output<'_, Reply<DiagnosticReceipt>>,
    ) -> xabi::Result<()>;
    fn close(&self, token: u64, output: Output<'_, Reply<u32>>) -> xabi::Result<()>;
    fn read(
        &self,
        kind: u32,
        query: Str<'_>,
        output: SliceMut<'_, u8>,
        receipt: Output<'_, Reply<usize>>,
    ) -> xabi::Result<()>;
}
