// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Originating-library ownership adapters for the additive native v1.1 lane.
use crate::{
    Json,
    future::{FfiFuture, FutureExt as _},
    transfer::{Transfer, Unit, Value},
    v1_0 as a,
    v1_1::*,
    wire_v1_1::*,
};
use xabi::XabiType;

fn reply<T>(
    value: std::result::Result<Value<T>, xabi::XabiCallError<Value<a::Error>>>,
) -> a::Result<T> {
    value.map(|value| value.0).map_err(|error| match error {
        xabi::XabiCallError::Export(Value(error)) => error,
        xabi::XabiCallError::Runtime(error) => a::Error::new("abi", error.to_string()),
    })
}
fn future<T: Send + 'static>(
    value: std::result::Result<
        Value<FfiFuture<a::Result<T>>>,
        xabi::XabiCallError<Value<a::Error>>,
    >,
) -> FfiFuture<a::Result<T>> {
    reply(value).unwrap_or_else(|error| async move { Err(error) }.into_ffi())
}

/// Owned context-aware tool callback, implemented in its originating library.
pub struct ToolCallbackBox(Box<dyn ToolCallback>);
impl ToolCallbackBox {
    /// Own a local callback implementation.
    pub fn new<T: ToolCallback + 'static>(value: T) -> Self {
        Self(Box::new(value))
    }
    /// Invoke the owned callback without moving execution to a different executor.
    pub fn invoke_tool(
        &self,
        invocation: ToolInvocation,
        next: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::ToolExecutionOutcome>> {
        self.0.invoke_tool(invocation, next)
    }
}
struct ToolCallbackRemote(XabiV1OwnedTraitWireToolCallback);
impl WireToolCallback for ToolCallbackBox {
    fn invoke_tool(
        &self,
        invocation: Value<ToolInvocation>,
        next: Value<a::ContinuationBox>,
    ) -> std::result::Result<Value<FfiFuture<a::Result<a::ToolExecutionOutcome>>>, Value<a::Error>>
    {
        Ok(Value(self.0.invoke_tool(invocation.0, next.0)))
    }
}
impl ToolCallback for ToolCallbackRemote {
    fn invoke_tool(
        &self,
        invocation: ToolInvocation,
        next: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::ToolExecutionOutcome>> {
        future(
            self.0
                .xabi_borrow()
                .invoke_tool(Value(invocation), Value(next)),
        )
    }
}

/// Owned runtime authority scoped to one component activation.
pub struct RuntimeBox(Box<dyn Runtime>);
impl RuntimeBox {
    /// Own a local runtime implementation.
    pub fn new<T: Runtime + 'static>(value: T) -> Self {
        Self(Box::new(value))
    }
    /// Obtain unchanged baseline authority from this runtime.
    pub fn baseline(&self) -> a::Result<a::RuntimeBox> {
        self.0.baseline()
    }
    /// Register the additive tool callback in the activation transaction.
    pub fn register_tool_execution(&self, registration: ToolRegistration) -> a::Result<()> {
        self.0.register_tool_execution(registration)
    }
}
struct RuntimeRemote(XabiV1OwnedTraitWireRuntime);
impl WireRuntime for RuntimeBox {
    fn baseline(&self) -> std::result::Result<Value<a::RuntimeBox>, Value<a::Error>> {
        self.0.baseline().map(Value).map_err(Value)
    }
    fn register_tool_execution(
        &self,
        registration: Value<ToolRegistration>,
    ) -> std::result::Result<Value<Unit>, Value<a::Error>> {
        self.0
            .register_tool_execution(registration.0)
            .map(|()| Value(()))
            .map_err(Value)
    }
}
impl Runtime for RuntimeRemote {
    fn baseline(&self) -> a::Result<a::RuntimeBox> {
        reply(self.0.xabi_borrow().baseline())
    }
    fn register_tool_execution(&self, registration: ToolRegistration) -> a::Result<()> {
        reply(
            self.0
                .xabi_borrow()
                .register_tool_execution(Value(registration)),
        )
    }
}

/// Owned native v1.1 plugin definition.
pub struct PluginBox(Box<dyn Plugin>);
impl PluginBox {
    /// Own a local plugin definition.
    pub fn new<T: Plugin + 'static>(value: T) -> Self {
        Self(Box::new(value))
    }
    /// Whether the definition permits multiple components.
    pub fn allows_multiple_components(&self) -> bool {
        self.0.allows_multiple_components()
    }
    /// Validate configuration without installing callbacks.
    pub fn validate(&self, config: Json) -> Vec<a::ConfigDiagnostic> {
        self.0.validate(config)
    }
    /// Register one component through the additive runtime.
    pub fn register(
        &self,
        config: Json,
        runtime: RuntimeBox,
    ) -> FfiFuture<a::Result<a::RegisteredComponent>> {
        self.0.register(config, runtime)
    }
}
struct PluginRemote(XabiV1OwnedTraitWirePlugin);
impl WirePlugin for PluginBox {
    fn allows_multiple_components(&self) -> std::result::Result<Value<bool>, Value<a::Error>> {
        Ok(Value(self.0.allows_multiple_components()))
    }
    fn validate(
        &self,
        config: Value<Json>,
    ) -> std::result::Result<Value<Vec<a::ConfigDiagnostic>>, Value<a::Error>> {
        Ok(Value(self.0.validate(config.0)))
    }
    fn register(
        &self,
        config: Value<Json>,
        runtime: Value<RuntimeBox>,
    ) -> std::result::Result<Value<FfiFuture<a::Result<a::RegisteredComponent>>>, Value<a::Error>>
    {
        Ok(Value(self.0.register(config.0, runtime.0)))
    }
}
impl Plugin for PluginRemote {
    fn allows_multiple_components(&self) -> bool {
        reply(self.0.xabi_borrow().allows_multiple_components()).unwrap_or(false)
    }
    fn validate(&self, config: Json) -> Vec<a::ConfigDiagnostic> {
        reply(self.0.xabi_borrow().validate(Value(config))).unwrap_or_else(|error| {
            vec![a::ConfigDiagnostic {
                is_error: true,
                code: error.code,
                component: None,
                field: None,
                message: error.message,
            }]
        })
    }
    fn register(
        &self,
        config: Json,
        runtime: RuntimeBox,
    ) -> FfiFuture<a::Result<a::RegisteredComponent>> {
        future(self.0.xabi_borrow().register(Value(config), Value(runtime)))
    }
}

/// Owned native v1.1 factory selected only after contract validation.
pub struct FactoryBox(Box<dyn Factory>);
impl FactoryBox {
    /// Own a local definition factory.
    pub fn new<T: Factory + 'static>(value: T) -> Self {
        Self(Box::new(value))
    }
    /// Construct one plugin definition.
    pub fn create(&self) -> a::Result<PluginBox> {
        self.0.create()
    }
}
struct FactoryRemote(XabiV1OwnedTraitWireFactory);
impl WireFactory for FactoryBox {
    fn create(&self) -> std::result::Result<Value<PluginBox>, Value<a::Error>> {
        self.0.create().map(Value).map_err(Value)
    }
}
impl Factory for FactoryRemote {
    fn create(&self) -> a::Result<PluginBox> {
        reply(self.0.xabi_borrow().create())
    }
}

unsafe impl Transfer for ToolCallbackBox {
    type Wire = <XabiV1OwnedTraitWireToolCallback as XabiType>::Wire;
    fn lower(self) -> Self::Wire {
        XabiV1OwnedTraitWireToolCallback::new(self).into_wire()
    }
    unsafe fn take(wire: &mut Self::Wire) -> xabi::Result<Self> {
        unsafe { XabiV1OwnedTraitWireToolCallback::xabi_take_from_wire(wire) }
            .map(|value| Self::new(ToolCallbackRemote(value)))
    }
    unsafe fn release(wire: &mut Self::Wire) {
        unsafe { XabiV1OwnedTraitWireToolCallback::xabi_drop_wire(wire) }
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

struct DefinitionFactory(fn() -> a::Result<PluginBox>);
impl Factory for DefinitionFactory {
    fn create(&self) -> a::Result<PluginBox> {
        (self.0)()
    }
}
/// Export a definition factory after the host checks the complete v1.1 contract.
pub fn export_factory(create: fn() -> a::Result<PluginBox>) -> *mut std::ffi::c_void {
    FactoryBox::new(DefinitionFactory(create))
        .lower()
        .vtable
        .cast()
}
/// Adopt a checked v1.1 factory from a process-resident library.
///
/// # Safety
/// Bootstrap, complete v1.1 contract and manifest must be validated first.
pub unsafe fn load_factory(export: &xabi::XabiExport) -> xabi::Result<FactoryBox> {
    let vtable = unsafe { (export.make)() }.cast::<XabiV1VtableTraitWireFactory>();
    if vtable.is_null() {
        return Err(xabi::Error::NullPointer("native v1.1 factory"));
    }
    let size = unsafe { vtable.cast::<usize>().read() };
    if size < XabiV1VtableTraitWireFactory::FULL_SIZE {
        if size >= XabiV1VtableTraitWireFactory::MIN_SIZE {
            let release = unsafe { std::ptr::addr_of!((*vtable).release).read() };
            unsafe { release(vtable) };
        }
        return Err(xabi::Error::AbiMismatch(
            "incomplete native v1.1 factory module".into(),
        ));
    }
    let mut wire = XabiV1OwnedRefTraitWireFactory {
        size: std::mem::size_of::<XabiV1OwnedRefTraitWireFactory>(),
        abi_version: 1,
        vtable,
    };
    unsafe { FactoryBox::take(&mut wire) }
}
