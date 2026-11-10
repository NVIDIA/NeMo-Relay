// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native capability owners and originating-library adapters.
use crate::{
    Json,
    diagnostics::*,
    future::{FfiFuture, FutureExt as _},
    transfer::{Transfer, Value},
    v1_0::*,
    wire::*,
};
use std::sync::{Arc, Mutex};
use xabi::XabiType;
fn error(value: xabi::XabiCallError<Value<Error>>) -> Error {
    match value {
        xabi::XabiCallError::Export(Value(value)) => value,
        xabi::XabiCallError::Runtime(value) => Error::new("abi", value.to_string()),
    }
}

/// Owned scope capability; the implementation stays in its originating library.
pub struct ScopeBox(Arc<dyn Scope>);
impl ScopeBox {
    /// Own a local implementation of this capability.
    pub fn new<T: Scope + 'static>(value: T) -> Self {
        Self(Arc::new(value))
    }
    /// Bind this scope on the calling thread and return an RAII restoration lease.
    pub fn enter(&self) -> Result<ScopeLeaseBox> {
        self.0.enter()
    }
    /// Inspect the effective stack, including the absence of an active stack.
    pub fn snapshot(&self) -> Result<Option<Json>> {
        self.0.snapshot()
    }
}
struct ScopeAdapter(Arc<dyn Scope>);
struct ScopeRemote(Arc<XabiV1OwnedTraitWireScope>);
impl WireScope for ScopeAdapter {
    fn enter(&self) -> std::result::Result<Value<ScopeLeaseBox>, Value<Error>> {
        self.0.enter().map(Value).map_err(Value)
    }
    fn snapshot(&self) -> std::result::Result<Value<Option<Json>>, Value<Error>> {
        self.0.snapshot().map(Value).map_err(Value)
    }
}
impl Scope for ScopeRemote {
    fn enter(&self) -> Result<ScopeLeaseBox> {
        let owner = self.0.clone();
        owner
            .xabi_borrow()
            .enter()
            .map(|value| value.0)
            .map_err(error)
    }
    fn snapshot(&self) -> Result<Option<Json>> {
        let owner = self.0.clone();
        owner
            .xabi_borrow()
            .snapshot()
            .map(|value| value.0)
            .map_err(error)
    }
}
unsafe impl Transfer for ScopeBox {
    type Wire = <XabiV1OwnedTraitWireScope as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireScope::new(ScopeAdapter(self.0)).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireScope::xabi_take_from_wire(wire) }
            .map(|value| Self::new(ScopeRemote(Arc::new(value))))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireScope::xabi_drop_wire(wire) }
    }
}

/// Owned codec capability; the implementation stays in its originating library.
pub struct CodecBox(Arc<dyn Codec>);
impl CodecBox {
    /// Own a local implementation of this capability.
    pub fn new<T: Codec + 'static>(value: T) -> Self {
        Self(Arc::new(value))
    }
    /// Stable codec identity, including none/runtime/opaque classifications.
    pub fn identity(&self) -> Result<Json> {
        self.0.identity()
    }
    /// Decode an application payload into its annotated representation.
    pub fn decode(&self, value: Json) -> Result<Json> {
        self.0.decode(value)
    }
    /// Encode changes while retaining the original application representation.
    pub fn encode(&self, annotated: Json, original: Json) -> Result<Json> {
        self.0.encode(annotated, original)
    }
}
struct CodecAdapter(Arc<dyn Codec>);
struct CodecRemote(Arc<XabiV1OwnedTraitWireCodec>);
impl WireCodec for CodecAdapter {
    fn identity(&self) -> std::result::Result<Value<Json>, Value<Error>> {
        self.0.identity().map(Value).map_err(Value)
    }
    fn decode(&self, value: Value<Json>) -> std::result::Result<Value<Json>, Value<Error>> {
        self.0.decode(value.0).map(Value).map_err(Value)
    }
    fn encode(
        &self,
        annotated: Value<Json>,
        original: Value<Json>,
    ) -> std::result::Result<Value<Json>, Value<Error>> {
        self.0
            .encode(annotated.0, original.0)
            .map(Value)
            .map_err(Value)
    }
}
impl Codec for CodecRemote {
    fn identity(&self) -> Result<Json> {
        let owner = self.0.clone();
        owner
            .xabi_borrow()
            .identity()
            .map(|value| value.0)
            .map_err(error)
    }
    fn decode(&self, value: Json) -> Result<Json> {
        let owner = self.0.clone();
        owner
            .xabi_borrow()
            .decode(Value(value))
            .map(|value| value.0)
            .map_err(error)
    }
    fn encode(&self, annotated: Json, original: Json) -> Result<Json> {
        let owner = self.0.clone();
        owner
            .xabi_borrow()
            .encode(Value(annotated), Value(original))
            .map(|value| value.0)
            .map_err(error)
    }
}
unsafe impl Transfer for CodecBox {
    type Wire = <XabiV1OwnedTraitWireCodec as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireCodec::new(CodecAdapter(self.0)).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireCodec::xabi_take_from_wire(wire) }
            .map(|value| Self::new(CodecRemote(Arc::new(value))))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireCodec::xabi_drop_wire(wire) }
    }
}

/// Owned continuation capability; the implementation stays in its originating library.
pub struct ContinuationBox(Arc<dyn Continuation>);
impl ContinuationBox {
    /// Own a local implementation of this capability.
    pub fn new<T: Continuation + 'static>(value: T) -> Self {
        Self(Arc::new(value))
    }
    /// Invoke a unary downstream operation.
    pub fn call(&self, value: Json) -> FfiFuture<Result<Json>> {
        self.0.call(value)
    }
    /// Invoke a canonical tool continuation without an untyped result envelope.
    pub fn call_tool(&self, value: Json) -> FfiFuture<Result<ToolExecutionResult>> {
        self.0.call_tool(value)
    }
    /// Open a downstream stream.
    pub fn open_stream(&self, value: Json) -> FfiFuture<Result<StreamBox>> {
        self.0.open_stream(value)
    }
}
struct ContinuationAdapter(Arc<dyn Continuation>);
struct ContinuationRemote(Arc<XabiV1OwnedTraitWireContinuation>);
impl WireContinuation for ContinuationAdapter {
    fn call(
        &self,
        value: Value<Json>,
    ) -> std::result::Result<Value<FfiFuture<Result<Json>>>, Value<Error>> {
        Ok(Value(self.0.call(value.0)))
    }
    fn call_tool(
        &self,
        value: Value<Json>,
    ) -> std::result::Result<Value<FfiFuture<Result<ToolExecutionResult>>>, Value<Error>> {
        Ok(Value(self.0.call_tool(value.0)))
    }
    fn open_stream(
        &self,
        value: Value<Json>,
    ) -> std::result::Result<Value<FfiFuture<Result<StreamBox>>>, Value<Error>> {
        Ok(Value(self.0.open_stream(value.0)))
    }
}
impl Continuation for ContinuationRemote {
    fn call(&self, value: Json) -> FfiFuture<Result<Json>> {
        let owner = self.0.clone();
        match owner
            .xabi_borrow()
            .call(Value(value))
            .map(|value| value.0)
            .map_err(error)
        {
            Ok(future) => future,
            Err(error) => async move { Err(error) }.into_ffi(),
        }
    }
    fn call_tool(&self, value: Json) -> FfiFuture<Result<ToolExecutionResult>> {
        let owner = self.0.clone();
        match owner
            .xabi_borrow()
            .call_tool(Value(value))
            .map(|value| value.0)
            .map_err(error)
        {
            Ok(future) => future,
            Err(error) => async move { Err(error) }.into_ffi(),
        }
    }
    fn open_stream(&self, value: Json) -> FfiFuture<Result<StreamBox>> {
        let owner = self.0.clone();
        match owner
            .xabi_borrow()
            .open_stream(Value(value))
            .map(|value| value.0)
            .map_err(error)
        {
            Ok(future) => future,
            Err(error) => async move { Err(error) }.into_ffi(),
        }
    }
}
unsafe impl Transfer for ContinuationBox {
    type Wire = <XabiV1OwnedTraitWireContinuation as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireContinuation::new(ContinuationAdapter(self.0)).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireContinuation::xabi_take_from_wire(wire) }
            .map(|value| Self::new(ContinuationRemote(Arc::new(value))))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireContinuation::xabi_drop_wire(wire) }
    }
}

/// Owned stream capability; the implementation stays in its originating library.
pub struct StreamBox(Box<dyn Stream>);
impl StreamBox {
    /// Own a local implementation of this capability.
    pub fn new<T: Stream + 'static>(value: T) -> Self {
        Self(Box::new(value))
    }
    /// Pull one item; dropping the returned future cancels that pull.
    pub fn pull(&mut self) -> FfiFuture<Result<Option<Json>>> {
        self.0.pull()
    }
    /// Cancel and await tracked stream cleanup; repeated close is harmless.
    pub fn close(&mut self) -> FfiFuture<Result<()>> {
        self.0.close()
    }
}
struct StreamAdapter(Mutex<Box<dyn Stream>>);
struct StreamRemote(Arc<XabiV1OwnedTraitWireStream>);
impl WireStream for StreamAdapter {
    fn pull(&self) -> std::result::Result<Value<FfiFuture<Result<Option<Json>>>>, Value<Error>> {
        Ok(Value({
            self.0.lock().unwrap_or_else(|e| e.into_inner()).pull()
        }))
    }
    fn close(
        &self,
    ) -> std::result::Result<Value<FfiFuture<Result<crate::transfer::Unit>>>, Value<Error>> {
        Ok(Value({
            self.0.lock().unwrap_or_else(|e| e.into_inner()).close()
        }))
    }
}
impl Stream for StreamRemote {
    fn pull(&mut self) -> FfiFuture<Result<Option<Json>>> {
        let owner = self.0.clone();
        match owner
            .xabi_borrow()
            .pull()
            .map(|value| value.0)
            .map_err(error)
        {
            Ok(future) => future,
            Err(error) => async move { Err(error) }.into_ffi(),
        }
    }
    fn close(&mut self) -> FfiFuture<Result<()>> {
        let owner = self.0.clone();
        match owner
            .xabi_borrow()
            .close()
            .map(|value| value.0)
            .map_err(error)
        {
            Ok(future) => future,
            Err(error) => async move { Err(error) }.into_ffi(),
        }
    }
}
unsafe impl Transfer for StreamBox {
    type Wire = <XabiV1OwnedTraitWireStream as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireStream::new(StreamAdapter(Mutex::new(self.0))).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireStream::xabi_take_from_wire(wire) }
            .map(|value| Self::new(StreamRemote(Arc::new(value))))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireStream::xabi_drop_wire(wire) }
    }
}

/// Owned callback capability; the implementation stays in its originating library.
pub struct CallbackBox(Arc<dyn Callback>);
impl CallbackBox {
    /// Own a local implementation of this capability.
    pub fn new<T: Callback + 'static>(value: T) -> Self {
        Self(Arc::new(value))
    }
    /// Invoke unary middleware on the callback's owning executor.
    pub fn invoke(
        &self,
        invocation: Invocation,
        next: Option<ContinuationBox>,
    ) -> FfiFuture<Result<Json>> {
        self.0.invoke(invocation, next)
    }
    /// Invoke tool middleware with a canonical typed result and deferred marks.
    pub fn invoke_tool(
        &self,
        invocation: Invocation,
        next: ContinuationBox,
    ) -> FfiFuture<Result<ToolExecutionOutcome>> {
        self.0.invoke_tool(invocation, next)
    }
    /// Invoke a synchronous subscriber or raw synchronous callback.
    pub fn invoke_sync(&self, invocation: Invocation) -> Result<Json> {
        self.0.invoke_sync(invocation)
    }
    /// Invoke streaming middleware on its owning executor.
    pub fn open_stream(
        &self,
        invocation: Invocation,
        next: ContinuationBox,
    ) -> FfiFuture<Result<StreamBox>> {
        self.0.open_stream(invocation, next)
    }
}
struct CallbackAdapter(Arc<dyn Callback>);
struct CallbackRemote(Arc<XabiV1OwnedTraitWireCallback>);
impl WireCallback for CallbackAdapter {
    fn invoke(
        &self,
        invocation: Value<Invocation>,
        next: Value<Option<ContinuationBox>>,
    ) -> std::result::Result<Value<FfiFuture<Result<Json>>>, Value<Error>> {
        Ok(Value(self.0.invoke(invocation.0, next.0)))
    }
    fn invoke_tool(
        &self,
        invocation: Value<Invocation>,
        next: Value<ContinuationBox>,
    ) -> std::result::Result<Value<FfiFuture<Result<ToolExecutionOutcome>>>, Value<Error>> {
        Ok(Value(self.0.invoke_tool(invocation.0, next.0)))
    }
    fn invoke_sync(
        &self,
        invocation: Value<Invocation>,
    ) -> std::result::Result<Value<Json>, Value<Error>> {
        self.0.invoke_sync(invocation.0).map(Value).map_err(Value)
    }
    fn open_stream(
        &self,
        invocation: Value<Invocation>,
        next: Value<ContinuationBox>,
    ) -> std::result::Result<Value<FfiFuture<Result<StreamBox>>>, Value<Error>> {
        Ok(Value(self.0.open_stream(invocation.0, next.0)))
    }
}
impl Callback for CallbackRemote {
    fn invoke(
        &self,
        invocation: Invocation,
        next: Option<ContinuationBox>,
    ) -> FfiFuture<Result<Json>> {
        let owner = self.0.clone();
        match owner
            .xabi_borrow()
            .invoke(Value(invocation), Value(next))
            .map(|value| value.0)
            .map_err(error)
        {
            Ok(future) => future,
            Err(error) => async move { Err(error) }.into_ffi(),
        }
    }
    fn invoke_tool(
        &self,
        invocation: Invocation,
        next: ContinuationBox,
    ) -> FfiFuture<Result<ToolExecutionOutcome>> {
        let owner = self.0.clone();
        match owner
            .xabi_borrow()
            .invoke_tool(Value(invocation), Value(next))
            .map(|value| value.0)
            .map_err(error)
        {
            Ok(future) => future,
            Err(error) => async move { Err(error) }.into_ffi(),
        }
    }
    fn invoke_sync(&self, invocation: Invocation) -> Result<Json> {
        let owner = self.0.clone();
        owner
            .xabi_borrow()
            .invoke_sync(Value(invocation))
            .map(|value| value.0)
            .map_err(error)
    }
    fn open_stream(
        &self,
        invocation: Invocation,
        next: ContinuationBox,
    ) -> FfiFuture<Result<StreamBox>> {
        let owner = self.0.clone();
        match owner
            .xabi_borrow()
            .open_stream(Value(invocation), Value(next))
            .map(|value| value.0)
            .map_err(error)
        {
            Ok(future) => future,
            Err(error) => async move { Err(error) }.into_ffi(),
        }
    }
}
unsafe impl Transfer for CallbackBox {
    type Wire = <XabiV1OwnedTraitWireCallback as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireCallback::new(CallbackAdapter(self.0)).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireCallback::xabi_take_from_wire(wire) }
            .map(|value| Self::new(CallbackRemote(Arc::new(value))))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireCallback::xabi_drop_wire(wire) }
    }
}

/// Owned component capability; the implementation stays in its originating library.
pub struct ComponentBox(Arc<dyn Component>);
impl ComponentBox {
    /// Own a local implementation of this capability.
    pub fn new<T: Component + 'static>(value: T) -> Self {
        Self(Arc::new(value))
    }
    /// Stop admission, cancel, drain, and release the component executor.
    pub fn shutdown(&self) -> FfiFuture<Result<()>> {
        self.0.shutdown()
    }
}
struct ComponentAdapter(Arc<dyn Component>);
struct ComponentRemote(Arc<XabiV1OwnedTraitWireComponent>);
impl WireComponent for ComponentAdapter {
    fn shutdown(
        &self,
    ) -> std::result::Result<Value<FfiFuture<Result<crate::transfer::Unit>>>, Value<Error>> {
        Ok(Value(self.0.shutdown()))
    }
}
impl Component for ComponentRemote {
    fn shutdown(&self) -> FfiFuture<Result<()>> {
        let owner = self.0.clone();
        match owner
            .xabi_borrow()
            .shutdown()
            .map(|value| value.0)
            .map_err(error)
        {
            Ok(future) => future,
            Err(error) => async move { Err(error) }.into_ffi(),
        }
    }
}
unsafe impl Transfer for ComponentBox {
    type Wire = <XabiV1OwnedTraitWireComponent as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireComponent::new(ComponentAdapter(self.0)).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireComponent::xabi_take_from_wire(wire) }
            .map(|value| Self::new(ComponentRemote(Arc::new(value))))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireComponent::xabi_drop_wire(wire) }
    }
}

/// Owned runtime capability; the implementation stays in its originating library.
pub struct RuntimeBox(Arc<dyn Runtime>);
impl RuntimeBox {
    /// Own a local implementation of this capability.
    pub fn new<T: Runtime + 'static>(value: T) -> Self {
        Self(Arc::new(value))
    }
    /// Synchronous scope, telemetry, registry, codec, and diagnostic operations.
    pub fn call_sync(&self, operation: String, arguments: Json) -> Result<Json> {
        self.0.call_sync(operation, arguments)
    }
    /// Operations requiring host-owned async execution.
    pub fn call(&self, operation: String, arguments: Json) -> FfiFuture<Result<Json>> {
        self.0.call(operation, arguments)
    }
    /// Bounded diagnostic authoring and separately authorized whole-context reads.
    pub fn diagnostics(&self) -> crate::diagnostics::DiagnosticsBox {
        self.0.diagnostics()
    }
    /// Capture actual host scope/event/publication ownership on the calling thread.
    pub fn capture_scope(&self) -> Result<ScopeBox> {
        self.0.capture_scope()
    }
    /// Transfer cleanup ownership before starting registration work. Exactly one
    /// owner may be bound during activation; retained capabilities cannot rebind.
    pub fn bind_component(&self, component: ComponentBox) -> Result<()> {
        self.0.bind_component(component)
    }
    /// Apply an activation-time callback through the current host transaction.
    pub fn register_callback(&self, registration: Registration) -> Result<()> {
        self.0.register_callback(registration)
    }
    /// Register a component-owned dynamic conditional gate; other surfaces fail.
    pub fn register_gate(&self, registration: Registration) -> Result<String> {
        self.0.register_gate(registration)
    }
}
struct RuntimeAdapter(Arc<dyn Runtime>);
struct RuntimeRemote(Arc<XabiV1OwnedTraitWireRuntime>);
impl WireRuntime for RuntimeAdapter {
    fn call_sync(
        &self,
        operation: Value<String>,
        arguments: Value<Json>,
    ) -> std::result::Result<Value<Json>, Value<Error>> {
        self.0
            .call_sync(operation.0, arguments.0)
            .map(Value)
            .map_err(Value)
    }
    fn call(
        &self,
        operation: Value<String>,
        arguments: Value<Json>,
    ) -> std::result::Result<Value<FfiFuture<Result<Json>>>, Value<Error>> {
        Ok(Value(self.0.call(operation.0, arguments.0)))
    }
    fn diagnostics(
        &self,
    ) -> std::result::Result<Value<crate::diagnostics::DiagnosticsBox>, Value<Error>> {
        Ok(Value(self.0.diagnostics()))
    }
    fn capture_scope(&self) -> std::result::Result<Value<ScopeBox>, Value<Error>> {
        self.0.capture_scope().map(Value).map_err(Value)
    }
    fn bind_component(
        &self,
        component: Value<ComponentBox>,
    ) -> std::result::Result<Value<crate::transfer::Unit>, Value<Error>> {
        self.0.bind_component(component.0).map(Value).map_err(Value)
    }
    fn register_callback(
        &self,
        registration: Value<Registration>,
    ) -> std::result::Result<Value<crate::transfer::Unit>, Value<Error>> {
        self.0
            .register_callback(registration.0)
            .map(Value)
            .map_err(Value)
    }
    fn register_gate(
        &self,
        registration: Value<Registration>,
    ) -> std::result::Result<Value<String>, Value<Error>> {
        self.0
            .register_gate(registration.0)
            .map(Value)
            .map_err(Value)
    }
}
impl Runtime for RuntimeRemote {
    fn call_sync(&self, operation: String, arguments: Json) -> Result<Json> {
        let owner = self.0.clone();
        owner
            .xabi_borrow()
            .call_sync(Value(operation), Value(arguments))
            .map(|value| value.0)
            .map_err(error)
    }
    fn call(&self, operation: String, arguments: Json) -> FfiFuture<Result<Json>> {
        let owner = self.0.clone();
        match owner
            .xabi_borrow()
            .call(Value(operation), Value(arguments))
            .map(|value| value.0)
            .map_err(error)
        {
            Ok(future) => future,
            Err(error) => async move { Err(error) }.into_ffi(),
        }
    }
    fn diagnostics(&self) -> crate::diagnostics::DiagnosticsBox {
        let owner = self.0.clone();
        owner
            .xabi_borrow()
            .diagnostics()
            .map(|value| value.0)
            .map_err(error)
            .unwrap_or_else(|e| panic!("native interface call failed: {:?}", e))
    }
    fn capture_scope(&self) -> Result<ScopeBox> {
        let owner = self.0.clone();
        owner
            .xabi_borrow()
            .capture_scope()
            .map(|value| value.0)
            .map_err(error)
    }
    fn bind_component(&self, component: ComponentBox) -> Result<()> {
        let owner = self.0.clone();
        owner
            .xabi_borrow()
            .bind_component(Value(component))
            .map(|value| value.0)
            .map_err(error)
    }
    fn register_callback(&self, registration: Registration) -> Result<()> {
        let owner = self.0.clone();
        owner
            .xabi_borrow()
            .register_callback(Value(registration))
            .map(|value| value.0)
            .map_err(error)
    }
    fn register_gate(&self, registration: Registration) -> Result<String> {
        let owner = self.0.clone();
        owner
            .xabi_borrow()
            .register_gate(Value(registration))
            .map(|value| value.0)
            .map_err(error)
    }
}
unsafe impl Transfer for RuntimeBox {
    type Wire = <XabiV1OwnedTraitWireRuntime as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireRuntime::new(RuntimeAdapter(self.0)).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireRuntime::xabi_take_from_wire(wire) }
            .map(|value| Self::new(RuntimeRemote(Arc::new(value))))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireRuntime::xabi_drop_wire(wire) }
    }
}

/// Owned plugin capability; the implementation stays in its originating library.
pub struct PluginBox(Arc<dyn Plugin>);
impl PluginBox {
    /// Own a local implementation of this capability.
    pub fn new<T: Plugin + 'static>(value: T) -> Self {
        Self(Arc::new(value))
    }
    /// Whether the plugin permits multiple configured components.
    pub fn allows_multiple_components(&self) -> bool {
        self.0.allows_multiple_components()
    }
    /// Validate one component without installing callbacks.
    pub fn validate(&self, config: Json) -> Vec<ConfigDiagnostic> {
        self.0.validate(config)
    }
    /// Register one component on its owning executor.
    pub fn register(
        &self,
        config: Json,
        runtime: RuntimeBox,
    ) -> FfiFuture<Result<RegisteredComponent>> {
        self.0.register(config, runtime)
    }
}
struct PluginAdapter(Arc<dyn Plugin>);
struct PluginRemote(Arc<XabiV1OwnedTraitWirePlugin>);
impl WirePlugin for PluginAdapter {
    fn allows_multiple_components(&self) -> std::result::Result<Value<bool>, Value<Error>> {
        Ok(Value(self.0.allows_multiple_components()))
    }
    fn validate(
        &self,
        config: Value<Json>,
    ) -> std::result::Result<Value<Vec<ConfigDiagnostic>>, Value<Error>> {
        Ok(Value(self.0.validate(config.0)))
    }
    fn register(
        &self,
        config: Value<Json>,
        runtime: Value<RuntimeBox>,
    ) -> std::result::Result<Value<FfiFuture<Result<RegisteredComponent>>>, Value<Error>> {
        Ok(Value(self.0.register(config.0, runtime.0)))
    }
}
impl Plugin for PluginRemote {
    fn allows_multiple_components(&self) -> bool {
        let owner = self.0.clone();
        owner
            .xabi_borrow()
            .allows_multiple_components()
            .map(|value| value.0)
            .map_err(error)
            .unwrap_or_else(|e| panic!("native interface call failed: {:?}", e))
    }
    fn validate(&self, config: Json) -> Vec<ConfigDiagnostic> {
        let owner = self.0.clone();
        owner
            .xabi_borrow()
            .validate(Value(config))
            .map(|value| value.0)
            .map_err(error)
            .unwrap_or_else(|e| panic!("native interface call failed: {:?}", e))
    }
    fn register(
        &self,
        config: Json,
        runtime: RuntimeBox,
    ) -> FfiFuture<Result<RegisteredComponent>> {
        let owner = self.0.clone();
        match owner
            .xabi_borrow()
            .register(Value(config), Value(runtime))
            .map(|value| value.0)
            .map_err(error)
        {
            Ok(future) => future,
            Err(error) => async move { Err(error) }.into_ffi(),
        }
    }
}
unsafe impl Transfer for PluginBox {
    type Wire = <XabiV1OwnedTraitWirePlugin as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWirePlugin::new(PluginAdapter(self.0)).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWirePlugin::xabi_take_from_wire(wire) }
            .map(|value| Self::new(PluginRemote(Arc::new(value))))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWirePlugin::xabi_drop_wire(wire) }
    }
}

/// Owned factory capability; the implementation stays in its originating library.
pub struct FactoryBox(Arc<dyn Factory>);
impl FactoryBox {
    /// Own a local implementation of this capability.
    pub fn new<T: Factory + 'static>(value: T) -> Self {
        Self(Arc::new(value))
    }
    /// Create one plugin definition without admitting component work.
    pub fn create(&self) -> Result<PluginBox> {
        self.0.create()
    }
}
struct FactoryAdapter(Arc<dyn Factory>);
struct FactoryRemote(Arc<XabiV1OwnedTraitWireFactory>);
impl WireFactory for FactoryAdapter {
    fn create(&self) -> std::result::Result<Value<PluginBox>, Value<Error>> {
        self.0.create().map(Value).map_err(Value)
    }
}
impl Factory for FactoryRemote {
    fn create(&self) -> Result<PluginBox> {
        let owner = self.0.clone();
        owner
            .xabi_borrow()
            .create()
            .map(|value| value.0)
            .map_err(error)
    }
}
unsafe impl Transfer for FactoryBox {
    type Wire = <XabiV1OwnedTraitWireFactory as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireFactory::new(FactoryAdapter(self.0)).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireFactory::xabi_take_from_wire(wire) }
            .map(|value| Self::new(FactoryRemote(Arc::new(value))))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireFactory::xabi_drop_wire(wire) }
    }
}

/// Owned diagnostics capability; the implementation stays in its originating library.
pub struct DiagnosticsBox(Arc<dyn Diagnostics>);
impl DiagnosticsBox {
    /// Own a local implementation of this capability.
    pub fn new<T: Diagnostics + 'static>(value: T) -> Self {
        Self(Arc::new(value))
    }
    /// Non-sensitive limits and the host-selected whole-context read grant.
    pub fn capabilities(&self) -> std::result::Result<DiagnosticCapabilities, u32> {
        self.0.capabilities()
    }
    /// Admit one bounded component and all code metadata together.
    pub fn register(
        &self,
        registration: DiagnosticRegistration<'_>,
    ) -> std::result::Result<u64, u32> {
        self.0.register(registration)
    }
    /// Submit once; an uncertain error must not be automatically retried.
    pub fn record(
        &self,
        token: u64,
        observation: DiagnosticObservation<'_>,
    ) -> std::result::Result<DiagnosticReceipt, u32> {
        self.0.record(token, observation)
    }
    /// Close a declaration without changing source-wide totals.
    pub fn close(&self, token: u64) -> std::result::Result<(), u32> {
        self.0.close(token)
    }
    /// Copy a canonical read into caller-owned storage; no partial output is published.
    pub fn read(
        &self,
        kind: u32,
        query: crate::borrow::Str<'_>,
        output: crate::borrow::SliceMut<'_, u8>,
    ) -> std::result::Result<usize, u32> {
        self.0.read(kind, query, output)
    }
}
struct DiagnosticsAdapter(Arc<dyn Diagnostics>);
struct DiagnosticsRemote(Arc<XabiV1OwnedTraitWireDiagnostics>);
use crate::borrow::{Output, Reply, SliceMut, Str};
impl WireDiagnostics for DiagnosticsAdapter {
    fn limits(&self, mut output: Output<'_, Reply<DiagnosticCapabilities>>) -> xabi::Result<()> {
        output.write(self.0.capabilities());
        Ok(())
    }
    fn register(
        &self,
        registration: DiagnosticRegistration<'_>,
        mut output: Output<'_, Reply<u64>>,
    ) -> xabi::Result<()> {
        output.write(self.0.register(registration));
        Ok(())
    }
    fn record(
        &self,
        token: u64,
        observation: DiagnosticObservation<'_>,
        mut output: Output<'_, Reply<DiagnosticReceipt>>,
    ) -> xabi::Result<()> {
        output.write(self.0.record(token, observation));
        Ok(())
    }
    fn close(&self, token: u64, mut output: Output<'_, Reply<u32>>) -> xabi::Result<()> {
        output.write(self.0.close(token).map(|()| 0));
        Ok(())
    }
    fn read(
        &self,
        kind: u32,
        query: Str<'_>,
        output: SliceMut<'_, u8>,
        mut receipt: Output<'_, Reply<usize>>,
    ) -> xabi::Result<()> {
        receipt.write(self.0.read(kind, query, output));
        Ok(())
    }
}
impl Diagnostics for DiagnosticsRemote {
    fn capabilities(&self) -> std::result::Result<DiagnosticCapabilities, u32> {
        let mut reply = Reply::default();
        self.0
            .xabi_borrow()
            .limits(Output::new(&mut reply))
            .map_err(|_| 11u32)?;
        reply.into_result()
    }
    fn register(&self, registration: DiagnosticRegistration<'_>) -> std::result::Result<u64, u32> {
        let mut reply = Reply::default();
        self.0
            .xabi_borrow()
            .register(registration, Output::new(&mut reply))
            .map_err(|_| 11u32)?;
        reply.into_result()
    }
    fn record(
        &self,
        token: u64,
        observation: DiagnosticObservation<'_>,
    ) -> std::result::Result<DiagnosticReceipt, u32> {
        let mut reply = Reply::default();
        self.0
            .xabi_borrow()
            .record(token, observation, Output::new(&mut reply))
            .map_err(|_| 11u32)?;
        reply.into_result()
    }
    fn close(&self, token: u64) -> std::result::Result<(), u32> {
        let mut reply = Reply::default();
        self.0
            .xabi_borrow()
            .close(token, Output::new(&mut reply))
            .map_err(|_| 11u32)?;
        reply.into_result().map(|_| ())
    }
    fn read(
        &self,
        kind: u32,
        query: Str<'_>,
        output: SliceMut<'_, u8>,
    ) -> std::result::Result<usize, u32> {
        let mut reply = Reply::default();
        self.0
            .xabi_borrow()
            .read(kind, query, output, Output::new(&mut reply))
            .map_err(|_| 11u32)?;
        reply.into_result()
    }
}
unsafe impl Transfer for DiagnosticsBox {
    type Wire = <XabiV1OwnedTraitWireDiagnostics as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireDiagnostics::new(DiagnosticsAdapter(self.0)).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireDiagnostics::xabi_take_from_wire(wire) }
            .map(|value| Self::new(DiagnosticsRemote(Arc::new(value))))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireDiagnostics::xabi_drop_wire(wire) }
    }
}

/// Scope restoration owner. It is deliberately neither Send nor Sync.
pub struct ScopeLeaseBox(Box<dyn ScopeLease>);
impl ScopeLeaseBox {
    /// Own a local implementation of this capability.
    pub fn new<T: ScopeLease + 'static>(value: T) -> Self {
        Self(Box::new(value))
    }
    /// Explicit idempotent restoration before drop.
    pub fn restore(&mut self) -> Result<()> {
        self.0.restore()
    }
}
/// Frozen thread-affine lease callbacks; all code runs on the entering thread.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LeaseWire {
    /// Thread-affine scope restoration state owned by the source library.
    pub instance: *mut std::ffi::c_void,
    /// Restore the previous scope on the entering thread.
    pub restore: unsafe extern "C" fn(*mut std::ffi::c_void) -> xabi::XabiOwnedBytes,
    /// Drop the lease on its entering thread using the source allocator.
    pub release: unsafe extern "C" fn(*mut std::ffi::c_void),
}
unsafe extern "C" fn restore_lease(instance: *mut std::ffi::c_void) -> xabi::XabiOwnedBytes {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { &mut *instance.cast::<ScopeLeaseBox>() }.restore()
    }));
    Value(
        result
            .unwrap_or_else(|_| Err(Error::new("panic", "scope restoration panicked")))
            .err(),
    )
    .into_payload()
}
unsafe extern "C" fn release_lease(instance: *mut std::ffi::c_void) {
    if !instance.is_null() {
        drop(unsafe { Box::from_raw(instance.cast::<ScopeLeaseBox>()) });
    }
}
struct ForeignLease(LeaseWire, std::marker::PhantomData<std::rc::Rc<()>>);
impl ScopeLease for ForeignLease {
    fn restore(&mut self) -> Result<()> {
        let payload = unsafe { (self.0.restore)(self.0.instance) };
        let error = unsafe { Value::<Option<Error>>::from_payload(payload) }
            .map_err(|e| Error::new("abi", e.to_string()))?
            .0;
        match error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}
impl Drop for ForeignLease {
    fn drop(&mut self) {
        unsafe { (self.0.release)(self.0.instance) }
    }
}
unsafe impl Transfer for ScopeLeaseBox {
    type Wire = LeaseWire;
    fn lower(self) -> LeaseWire {
        LeaseWire {
            instance: Box::into_raw(Box::new(self)).cast(),
            restore: restore_lease,
            release: release_lease,
        }
    }
    unsafe fn take(wire: &mut LeaseWire) -> xabi::Result<Self> {
        if wire.instance.is_null() {
            return Err(xabi::Error::NullPointer("scope lease"));
        }
        let value = *wire;
        wire.instance = std::ptr::null_mut();
        Ok(Self::new(ForeignLease(value, std::marker::PhantomData)))
    }
    unsafe fn release(wire: &mut LeaseWire) {
        if !wire.instance.is_null() {
            unsafe { (wire.release)(std::mem::replace(&mut wire.instance, std::ptr::null_mut())) };
        }
    }
}

struct DefinitionFactory(fn() -> Result<PluginBox>);
impl Factory for DefinitionFactory {
    fn create(&self) -> Result<PluginBox> {
        (self.0)()
    }
}
/// Export the factory only after the host checks its complete declared contract.
pub fn export_factory(create: fn() -> Result<PluginBox>) -> *mut std::ffi::c_void {
    FactoryBox::new(DefinitionFactory(create))
        .lower()
        .vtable
        .cast()
}
/// Adopt a checked factory export from a process-resident library.
///
/// # Safety
/// The host must validate the bootstrap, full contract schema, and manifest first.
pub unsafe fn load_factory(export: &xabi::XabiExport) -> xabi::Result<FactoryBox> {
    let vtable = unsafe { (export.make)() }.cast::<XabiV1VtableTraitWireFactory>();
    if vtable.is_null() {
        return Err(xabi::Error::NullPointer("native factory"));
    }
    // xabi accepts prefix tables, but every method in this exact contract is required.
    // Read only the header until the table has proved large enough.
    let size = unsafe { vtable.cast::<usize>().read() };
    if size < XabiV1VtableTraitWireFactory::FULL_SIZE {
        if size >= XabiV1VtableTraitWireFactory::MIN_SIZE {
            let release = unsafe { std::ptr::addr_of!((*vtable).release).read() };
            unsafe { release(vtable) };
        }
        return Err(xabi::Error::AbiMismatch(
            "incomplete native factory module".into(),
        ));
    }
    let mut wire = XabiV1OwnedRefTraitWireFactory {
        size: std::mem::size_of::<XabiV1OwnedRefTraitWireFactory>(),
        abi_version: 1,
        vtable,
    };
    unsafe { FactoryBox::take(&mut wire) }
}
