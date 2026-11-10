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
use std::sync::Mutex;
use xabi::XabiType;
fn error(value: xabi::XabiCallError<Value<Error>>) -> Error {
    match value {
        xabi::XabiCallError::Export(Value(value)) => value,
        xabi::XabiCallError::Runtime(value) => Error::new("abi", value.to_string()),
    }
}

/// Decode a foreign reply using the native error contract.
fn reply<T>(value: std::result::Result<Value<T>, xabi::XabiCallError<Value<Error>>>) -> Result<T> {
    value.map(|value| value.0).map_err(error)
}

/// Preserve eager invocation and turn a call failure into an owned ready future.
fn future<T: Send + 'static>(
    value: std::result::Result<Value<FfiFuture<Result<T>>>, xabi::XabiCallError<Value<Error>>>,
) -> FfiFuture<Result<T>> {
    reply(value).unwrap_or_else(|error| async move { Err(error) }.into_ffi())
}

// Keep Rust-only ownership and delegation uniform; wire declarations and Transfer
// mappings stay explicit so the contract checker can inspect every ABI type.
macro_rules! capability {
    ($(#[$doc:meta])* $owner:ident($contract:ident) {
        $($(#[$method_doc:meta])* fn $method:ident(&self $(, $arg:ident: $ty:ty)* $(,)?) -> $result:ty;)*
    }) => {
        $(#[$doc])*
        pub struct $owner(Box<dyn $contract>);
        impl $owner {
            /// Own a local implementation of this capability.
            pub fn new<T: $contract + 'static>(value: T) -> Self {
                Self(Box::new(value))
            }
            $($(#[$method_doc])*
            pub fn $method(&self, $($arg: $ty),*) -> $result {
                self.0.$method($($arg),*)
            })*
        }
    };
}

capability! {
    /// Owned scope capability; the implementation stays in its originating library.
    ScopeBox(Scope) {
        /// Bind this scope on the calling thread and return an RAII restoration lease.
        fn enter(&self) -> Result<ScopeLeaseBox>;
        /// Inspect the effective stack, including the absence of an active stack.
        fn snapshot(&self) -> Result<Option<Json>>;
    }
}
struct ScopeRemote(XabiV1OwnedTraitWireScope);
impl WireScope for ScopeBox {
    fn enter(&self) -> std::result::Result<Value<ScopeLeaseBox>, Value<Error>> {
        self.0.enter().map(Value).map_err(Value)
    }
    fn snapshot(&self) -> std::result::Result<Value<Option<Json>>, Value<Error>> {
        self.0.snapshot().map(Value).map_err(Value)
    }
}
impl Scope for ScopeRemote {
    fn enter(&self) -> Result<ScopeLeaseBox> {
        reply(self.0.xabi_borrow().enter())
    }
    fn snapshot(&self) -> Result<Option<Json>> {
        reply(self.0.xabi_borrow().snapshot())
    }
}
unsafe impl Transfer for ScopeBox {
    type Wire = <XabiV1OwnedTraitWireScope as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireScope::new(self).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireScope::xabi_take_from_wire(wire) }
            .map(|value| Self::new(ScopeRemote(value)))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireScope::xabi_drop_wire(wire) }
    }
}

capability! {
    /// Owned codec capability; the implementation stays in its originating library.
    CodecBox(Codec) {
        /// Stable codec identity, including none/runtime/opaque classifications.
        fn identity(&self) -> Result<Json>;
        /// Decode an application payload into its annotated representation.
        fn decode(&self, value: Json) -> Result<Json>;
        /// Encode changes while retaining the original application representation.
        fn encode(&self, annotated: Json, original: Json) -> Result<Json>;
    }
}
struct CodecRemote(XabiV1OwnedTraitWireCodec);
impl WireCodec for CodecBox {
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
        reply(self.0.xabi_borrow().identity())
    }
    fn decode(&self, value: Json) -> Result<Json> {
        reply(self.0.xabi_borrow().decode(Value(value)))
    }
    fn encode(&self, annotated: Json, original: Json) -> Result<Json> {
        reply(
            self.0
                .xabi_borrow()
                .encode(Value(annotated), Value(original)),
        )
    }
}
unsafe impl Transfer for CodecBox {
    type Wire = <XabiV1OwnedTraitWireCodec as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireCodec::new(self).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireCodec::xabi_take_from_wire(wire) }
            .map(|value| Self::new(CodecRemote(value)))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireCodec::xabi_drop_wire(wire) }
    }
}

capability! {
    /// Owned continuation capability; the implementation stays in its originating library.
    ContinuationBox(Continuation) {
        /// Invoke a unary downstream operation.
        fn call(&self, value: Json) -> FfiFuture<Result<Json>>;
        /// Invoke a canonical tool continuation without an untyped result envelope.
        fn call_tool(&self, value: Json) -> FfiFuture<Result<ToolExecutionResult>>;
        /// Open a downstream stream.
        fn open_stream(&self, value: Json) -> FfiFuture<Result<StreamBox>>;
    }
}
struct ContinuationRemote(XabiV1OwnedTraitWireContinuation);
impl WireContinuation for ContinuationBox {
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
        future(self.0.xabi_borrow().call(Value(value)))
    }
    fn call_tool(&self, value: Json) -> FfiFuture<Result<ToolExecutionResult>> {
        future(self.0.xabi_borrow().call_tool(Value(value)))
    }
    fn open_stream(&self, value: Json) -> FfiFuture<Result<StreamBox>> {
        future(self.0.xabi_borrow().open_stream(Value(value)))
    }
}
unsafe impl Transfer for ContinuationBox {
    type Wire = <XabiV1OwnedTraitWireContinuation as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireContinuation::new(self).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireContinuation::xabi_take_from_wire(wire) }
            .map(|value| Self::new(ContinuationRemote(value)))
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
struct StreamRemote(XabiV1OwnedTraitWireStream);
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
        future(self.0.xabi_borrow().pull())
    }
    fn close(&mut self) -> FfiFuture<Result<()>> {
        future(self.0.xabi_borrow().close())
    }
}
unsafe impl Transfer for StreamBox {
    type Wire = <XabiV1OwnedTraitWireStream as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireStream::new(StreamAdapter(Mutex::new(self.0))).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireStream::xabi_take_from_wire(wire) }
            .map(|value| Self::new(StreamRemote(value)))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireStream::xabi_drop_wire(wire) }
    }
}

capability! {
    /// Owned callback capability; the implementation stays in its originating library.
    CallbackBox(Callback) {
        /// Invoke unary middleware on the callback's owning executor.
        fn invoke(
            &self,
            invocation: Invocation,
            next: Option<ContinuationBox>,
        ) -> FfiFuture<Result<Json>>;
        /// Invoke tool middleware with a canonical typed result and deferred marks.
        fn invoke_tool(
            &self,
            invocation: Invocation,
            next: ContinuationBox,
        ) -> FfiFuture<Result<ToolExecutionOutcome>>;
        /// Invoke a synchronous subscriber or raw synchronous callback.
        fn invoke_sync(&self, invocation: Invocation) -> Result<Json>;
        /// Invoke streaming middleware on its owning executor.
        fn open_stream(
            &self,
            invocation: Invocation,
            next: ContinuationBox,
        ) -> FfiFuture<Result<StreamBox>>;
    }
}
struct CallbackRemote(XabiV1OwnedTraitWireCallback);
impl WireCallback for CallbackBox {
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
        future(self.0.xabi_borrow().invoke(Value(invocation), Value(next)))
    }
    fn invoke_tool(
        &self,
        invocation: Invocation,
        next: ContinuationBox,
    ) -> FfiFuture<Result<ToolExecutionOutcome>> {
        future(
            self.0
                .xabi_borrow()
                .invoke_tool(Value(invocation), Value(next)),
        )
    }
    fn invoke_sync(&self, invocation: Invocation) -> Result<Json> {
        reply(self.0.xabi_borrow().invoke_sync(Value(invocation)))
    }
    fn open_stream(
        &self,
        invocation: Invocation,
        next: ContinuationBox,
    ) -> FfiFuture<Result<StreamBox>> {
        future(
            self.0
                .xabi_borrow()
                .open_stream(Value(invocation), Value(next)),
        )
    }
}
unsafe impl Transfer for CallbackBox {
    type Wire = <XabiV1OwnedTraitWireCallback as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireCallback::new(self).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireCallback::xabi_take_from_wire(wire) }
            .map(|value| Self::new(CallbackRemote(value)))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireCallback::xabi_drop_wire(wire) }
    }
}

capability! {
    /// Owned component capability; the implementation stays in its originating library.
    ComponentBox(Component) {
        /// Stop admission, cancel, drain, and release the component executor.
        fn shutdown(&self) -> FfiFuture<Result<()>>;
    }
}
struct ComponentRemote(XabiV1OwnedTraitWireComponent);
impl WireComponent for ComponentBox {
    fn shutdown(
        &self,
    ) -> std::result::Result<Value<FfiFuture<Result<crate::transfer::Unit>>>, Value<Error>> {
        Ok(Value(self.0.shutdown()))
    }
}
impl Component for ComponentRemote {
    fn shutdown(&self) -> FfiFuture<Result<()>> {
        future(self.0.xabi_borrow().shutdown())
    }
}
unsafe impl Transfer for ComponentBox {
    type Wire = <XabiV1OwnedTraitWireComponent as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireComponent::new(self).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireComponent::xabi_take_from_wire(wire) }
            .map(|value| Self::new(ComponentRemote(value)))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireComponent::xabi_drop_wire(wire) }
    }
}

capability! {
    /// Owned runtime capability; the implementation stays in its originating library.
    RuntimeBox(Runtime) {
        /// Synchronous scope, telemetry, registry, codec, and diagnostic operations.
        fn call_sync(&self, operation: String, arguments: Json) -> Result<Json>;
        /// Operations requiring host-owned async execution.
        fn call(&self, operation: String, arguments: Json) -> FfiFuture<Result<Json>>;
        /// Bounded diagnostic authoring and separately authorized whole-context reads.
        fn diagnostics(&self) -> crate::diagnostics::DiagnosticsBox;
        /// Capture actual host scope/event/publication ownership on the calling thread.
        fn capture_scope(&self) -> Result<ScopeBox>;
        /// Transfer cleanup ownership before starting registration work. Exactly one
        /// owner may be bound during activation; retained capabilities cannot rebind.
        fn bind_component(&self, component: ComponentBox) -> Result<()>;
        /// Apply an activation-time callback through the current host transaction.
        fn register_callback(&self, registration: Registration) -> Result<()>;
        /// Register a component-owned dynamic conditional gate; other surfaces fail.
        fn register_gate(&self, registration: Registration) -> Result<String>;
    }
}
struct RuntimeRemote(XabiV1OwnedTraitWireRuntime);
impl WireRuntime for RuntimeBox {
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
        reply(
            self.0
                .xabi_borrow()
                .call_sync(Value(operation), Value(arguments)),
        )
    }
    fn call(&self, operation: String, arguments: Json) -> FfiFuture<Result<Json>> {
        future(
            self.0
                .xabi_borrow()
                .call(Value(operation), Value(arguments)),
        )
    }
    fn diagnostics(&self) -> crate::diagnostics::DiagnosticsBox {
        reply(self.0.xabi_borrow().diagnostics())
            .unwrap_or_else(|e| panic!("native interface call failed: {:?}", e))
    }
    fn capture_scope(&self) -> Result<ScopeBox> {
        reply(self.0.xabi_borrow().capture_scope())
    }
    fn bind_component(&self, component: ComponentBox) -> Result<()> {
        reply(self.0.xabi_borrow().bind_component(Value(component)))
    }
    fn register_callback(&self, registration: Registration) -> Result<()> {
        reply(self.0.xabi_borrow().register_callback(Value(registration)))
    }
    fn register_gate(&self, registration: Registration) -> Result<String> {
        reply(self.0.xabi_borrow().register_gate(Value(registration)))
    }
}
unsafe impl Transfer for RuntimeBox {
    type Wire = <XabiV1OwnedTraitWireRuntime as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireRuntime::new(self).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireRuntime::xabi_take_from_wire(wire) }
            .map(|value| Self::new(RuntimeRemote(value)))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireRuntime::xabi_drop_wire(wire) }
    }
}

capability! {
    /// Owned plugin capability; the implementation stays in its originating library.
    PluginBox(Plugin) {
        /// Whether the plugin permits multiple configured components.
        fn allows_multiple_components(&self) -> bool;
        /// Validate one component without installing callbacks.
        fn validate(&self, config: Json) -> Vec<ConfigDiagnostic>;
        /// Register one component on its owning executor.
        fn register(
            &self,
            config: Json,
            runtime: RuntimeBox,
        ) -> FfiFuture<Result<RegisteredComponent>>;
    }
}
struct PluginRemote(XabiV1OwnedTraitWirePlugin);
impl WirePlugin for PluginBox {
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
        reply(self.0.xabi_borrow().allows_multiple_components())
            .unwrap_or_else(|e| panic!("native interface call failed: {:?}", e))
    }
    fn validate(&self, config: Json) -> Vec<ConfigDiagnostic> {
        reply(self.0.xabi_borrow().validate(Value(config)))
            .unwrap_or_else(|e| panic!("native interface call failed: {:?}", e))
    }
    fn register(
        &self,
        config: Json,
        runtime: RuntimeBox,
    ) -> FfiFuture<Result<RegisteredComponent>> {
        future(self.0.xabi_borrow().register(Value(config), Value(runtime)))
    }
}
unsafe impl Transfer for PluginBox {
    type Wire = <XabiV1OwnedTraitWirePlugin as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWirePlugin::new(self).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWirePlugin::xabi_take_from_wire(wire) }
            .map(|value| Self::new(PluginRemote(value)))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWirePlugin::xabi_drop_wire(wire) }
    }
}

capability! {
    /// Owned factory capability; the implementation stays in its originating library.
    FactoryBox(Factory) {
        /// Create one plugin definition without admitting component work.
        fn create(&self) -> Result<PluginBox>;
    }
}
struct FactoryRemote(XabiV1OwnedTraitWireFactory);
impl WireFactory for FactoryBox {
    fn create(&self) -> std::result::Result<Value<PluginBox>, Value<Error>> {
        self.0.create().map(Value).map_err(Value)
    }
}
impl Factory for FactoryRemote {
    fn create(&self) -> Result<PluginBox> {
        reply(self.0.xabi_borrow().create())
    }
}
unsafe impl Transfer for FactoryBox {
    type Wire = <XabiV1OwnedTraitWireFactory as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireFactory::new(self).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireFactory::xabi_take_from_wire(wire) }
            .map(|value| Self::new(FactoryRemote(value)))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireFactory::xabi_drop_wire(wire) }
    }
}

capability! {
    /// Owned diagnostics capability; the implementation stays in its originating library.
    DiagnosticsBox(Diagnostics) {
        /// Non-sensitive limits and the host-selected whole-context read grant.
        fn capabilities(&self) -> std::result::Result<DiagnosticCapabilities, u32>;
        /// Admit one bounded component and all code metadata together.
        fn register(
            &self,
            registration: DiagnosticRegistration<'_>,
        ) -> std::result::Result<u64, u32>;
        /// Submit once; an uncertain error must not be automatically retried.
        fn record(
            &self,
            token: u64,
            observation: DiagnosticObservation<'_>,
        ) -> std::result::Result<DiagnosticReceipt, u32>;
        /// Close a declaration without changing source-wide totals.
        fn close(&self, token: u64) -> std::result::Result<(), u32>;
        /// Copy a canonical read into caller-owned storage; no partial output is published.
        fn read(
            &self,
            kind: u32,
            query: crate::borrow::Str<'_>,
            output: crate::borrow::SliceMut<'_, u8>,
        ) -> std::result::Result<usize, u32>;
    }
}
struct DiagnosticsRemote(XabiV1OwnedTraitWireDiagnostics);
use crate::borrow::{Output, Reply, SliceMut, Str};
impl WireDiagnostics for DiagnosticsBox {
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
        XabiV1OwnedTraitWireDiagnostics::new(self).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireDiagnostics::xabi_take_from_wire(wire) }
            .map(|value| Self::new(DiagnosticsRemote(value)))
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
