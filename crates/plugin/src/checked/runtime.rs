// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::ScopeAttributes;
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};
/// Optional parent, behavior flags, and start time for a native scope.
#[derive(Clone)]
pub struct ScopeOptions<'a> {
    /// Explicit event parent. The current scope or active event is the default.
    pub parent: Option<&'a ScopeHandle<'a>>,
    /// Scope behavior flags. Empty flags are the default.
    pub attributes: ScopeAttributes,
    /// Explicit start time. The current UTC time is the default.
    pub timestamp: Option<SystemTime>,
}
impl Default for ScopeOptions<'_> {
    fn default() -> Self {
        Self {
            parent: None,
            attributes: ScopeAttributes::empty(),
            timestamp: None,
        }
    }
}

/// Optional event parent, time, schema, and severity for a native mark.
#[derive(Clone, Default)]
pub struct EmitMarkOptions<'a> {
    /// Explicit event parent. The current scope or active event is the default.
    pub parent: Option<&'a ScopeHandle<'a>>,
    /// Explicit event time. The current UTC time is the default.
    pub timestamp: Option<SystemTime>,
    /// Schema identifier for the mark payload.
    pub data_schema: Option<DataSchema>,
    /// Telemetry severity for the mark.
    pub severity: Option<LogSeverity>,
}

#[derive(Clone)]
pub struct PluginRuntime {
    pub(super) raw: Arc<a::RuntimeBox>,
    pub(super) diagnostic_transport: Arc<abi::diagnostics::DiagnosticsBox>,
    pub(crate) diagnostic_state: Arc<crate::diagnostics::SdkState>,
}
impl PluginRuntime {
    /// Adapt an owned, checked runtime capability supplied by a native host.
    #[doc(hidden)]
    pub fn from_abi(runtime: a::RuntimeBox) -> Self {
        Self {
            diagnostic_transport: Arc::new(runtime.diagnostics()),
            raw: Arc::new(runtime),
            diagnostic_state: Arc::new(crate::diagnostics::SdkState::default()),
        }
    }

    pub fn diagnostics(&self) -> crate::diagnostics::Diagnostics {
        crate::diagnostics::Diagnostics::new(
            self.diagnostic_transport.clone(),
            self.diagnostic_state.clone(),
        )
    }
    pub(super) fn call<T: DeserializeOwned>(
        &self,
        operation: &str,
        args: impl Serialize,
    ) -> Result<T> {
        decode(local(self.raw.call_sync(operation.into(), encode(args)?))?)
    }
    pub fn current_scope_stack_snapshot(&self) -> Result<Option<ScopeStackSnapshot>> {
        self.call("scope.snapshot", Json::Null)
    }
    pub fn list_runtime_registrations(
        &self,
        kinds: Option<&BTreeSet<RuntimeRegistrationKind>>,
    ) -> Result<Vec<RuntimeRegistrationIdentity>> {
        self.call("registry.list", kinds)
    }
    pub fn scope_stack_active(&self) -> bool {
        self.call("scope.active", Json::Null).unwrap_or(false)
    }
    pub fn current_scope(&self) -> Result<ScopeHandle<'_>> {
        Ok(ScopeHandle {
            value: self.call("scope.current", Json::Null)?,
            _borrow: PhantomData,
        })
    }
    pub fn push_scope(
        &self,
        name: &str,
        scope_type: ScopeType,
        data: Option<&Json>,
        metadata: Option<&Json>,
        input: Option<&Json>,
    ) -> Result<ScopeHandle<'_>> {
        self.push_scope_with_options(
            name,
            scope_type,
            data,
            metadata,
            input,
            ScopeOptions::default(),
        )
    }
    /// Push a scope with an explicit parent, behavior flags, or start time.
    pub fn push_scope_with_options(
        &self,
        name: &str,
        scope_type: ScopeType,
        data: Option<&Json>,
        metadata: Option<&Json>,
        input: Option<&Json>,
        options: ScopeOptions<'_>,
    ) -> Result<ScopeHandle<'_>> {
        let timestamp = options.timestamp.map(micros).transpose()?;
        Ok(ScopeHandle {
            value: self.call("scope.push", json!({"name":name,"scope_type":scope_type,"data":data,"metadata":metadata,"input":input,
                "timestamp":timestamp,"parent":options.parent.map(|p| &p.value),"attributes":options.attributes.bits()}))?,
            _borrow: PhantomData,
        })
    }
    pub fn pop_scope(
        &self,
        handle: &ScopeHandle<'_>,
        output: Option<&Json>,
        metadata: Option<&Json>,
    ) -> Result<()> {
        self.call(
            "scope.pop",
            json!({"handle":handle.value,"output":output,"metadata":metadata}),
        )
    }
    pub fn scope(
        &self,
        name: &str,
        scope_type: ScopeType,
        data: Option<&Json>,
        metadata: Option<&Json>,
        input: Option<&Json>,
    ) -> Result<ScopeGuard<'_>> {
        Ok(ScopeGuard {
            runtime: self,
            handle: Some(self.push_scope(name, scope_type, data, metadata, input)?),
        })
    }
    /// Open a scope guard with an explicit parent, behavior flags, or start time.
    pub fn scope_with_options(
        &self,
        name: &str,
        scope_type: ScopeType,
        data: Option<&Json>,
        metadata: Option<&Json>,
        input: Option<&Json>,
        options: ScopeOptions<'_>,
    ) -> Result<ScopeGuard<'_>> {
        Ok(ScopeGuard {
            runtime: self,
            handle: Some(
                self.push_scope_with_options(name, scope_type, data, metadata, input, options)?,
            ),
        })
    }
    pub fn scope_at(
        &self,
        name: &str,
        scope_type: ScopeType,
        data: Option<&Json>,
        metadata: Option<&Json>,
        input: Option<&Json>,
        started_at: SystemTime,
    ) -> Result<ScopeGuard<'_>> {
        self.scope_with_options(
            name,
            scope_type,
            data,
            metadata,
            input,
            ScopeOptions {
                timestamp: Some(started_at),
                ..Default::default()
            },
        )
    }

    pub fn emit_mark(
        &self,
        name: &str,
        data: Option<&Json>,
        metadata: Option<&Json>,
    ) -> Result<()> {
        self.emit_mark_with_options(name, data, metadata, None, None)
    }
    pub fn emit_mark_with_options(
        &self,
        name: &str,
        data: Option<&Json>,
        metadata: Option<&Json>,
        data_schema: Option<&DataSchema>,
        severity: Option<LogSeverity>,
    ) -> Result<()> {
        self.emit_mark_with_context(
            name,
            data,
            metadata,
            EmitMarkOptions {
                data_schema: data_schema.cloned(),
                severity,
                ..Default::default()
            },
        )
    }
    /// Emit a mark with an explicit parent, timestamp, schema, or severity.
    pub fn emit_mark_with_context(
        &self,
        name: &str,
        data: Option<&Json>,
        metadata: Option<&Json>,
        options: EmitMarkOptions<'_>,
    ) -> Result<()> {
        let timestamp = options.timestamp.map(micros).transpose()?;
        self.call("mark", json!({"name":name,"data":data,"metadata":metadata,"parent":options.parent.map(|p| &p.value),
            "timestamp":timestamp,"data_schema":options.data_schema,"severity":options.severity}))
    }
    pub fn emit_metric(
        &self,
        name: &str,
        measurements: Vec<MetricMeasurement>,
        metadata: Option<&Json>,
    ) -> Result<()> {
        let value = MetricEnvelope { measurements };
        value.validate().map_err(|e| e.to_string())?;
        let data = serde_json::to_value(value).map_err(|e| e.to_string())?;
        let schema = DataSchema::builder()
            .name(METRIC_DATA_SCHEMA_NAME)
            .version(METRIC_DATA_SCHEMA_VERSION)
            .build();
        self.emit_mark_with_options(name, Some(&data), metadata, Some(&schema), None)
    }
    pub fn runtime_diagnostics(&self) -> Result<RuntimeDiagnostics> {
        self.call("runtime.diagnostics", Json::Null)
    }
    pub fn log(
        &self,
        level: PluginLogLevel,
        target: &str,
        message: &str,
        fields: Option<&Json>,
    ) -> Result<()> {
        let level = match level {
            PluginLogLevel::Error => 0,
            PluginLogLevel::Warn => 1,
            PluginLogLevel::Info => 2,
            PluginLogLevel::Debug => 3,
            PluginLogLevel::Trace => 4,
        };
        self.call(
            "log",
            json!({"level":level,"target":target,"message":message,"fields":fields}),
        )
    }
    pub fn create_scope_stack(&self) -> Result<ScopeStack<'_>> {
        Ok(ScopeStack {
            runtime: self,
            id: self.call("stack.create", Json::Null)?,
        })
    }
    pub fn capture_scope_stack_thread(&self) -> Result<ScopeStackBinding<'_>> {
        Ok(ScopeStackBinding {
            runtime: self,
            id: Some(self.call("stack.capture", Json::Null)?),
        })
    }
    pub fn bind_scope_stack_thread<'a>(
        &'a self,
        stack: &'a ScopeStack<'a>,
    ) -> Result<ThreadScopeStackGuard<'a>> {
        let previous = self.capture_scope_stack_thread()?;
        self.call::<()>("stack.set", &stack.id)?;
        Ok(ThreadScopeStackGuard {
            previous: Some(previous),
        })
    }
    pub fn register_conditional_middleware_guardrail<F>(
        &self,
        name: &str,
        kinds: &BTreeSet<RuntimeRegistrationKind>,
        registration_name: &str,
        callback: F,
    ) -> Result<ConditionalMiddlewareGuardrailHandle>
    where
        F: Fn(&BTreeSet<RuntimeRegistrationKind>, &str) -> Option<String> + Send + Sync + 'static,
    {
        let descriptor = a::Registration {
            name: name.into(),
            surface: a::Surface::ConditionalMiddleware,
            priority: 0,
            break_chain: false,
            options: encode(json!({"kinds":kinds,"registration_name":registration_name}))?,
            callback: a::Callback_TO::from_value(Gate(callback), TD_Opaque),
        };
        Ok(ConditionalMiddlewareGuardrailHandle(
            local(self.raw.register_gate(descriptor))?.to_string(),
        ))
    }
    pub fn deregister_conditional_middleware_guardrail(
        &self,
        handle: &ConditionalMiddlewareGuardrailHandle,
    ) -> Result<bool> {
        self.call("gate.remove", &handle.0)
    }
}
macro_rules! logs{($($name:ident=>$level:ident),*)=>{$(impl PluginRuntime{
pub fn $name(&self,target:&str,message:&str,fields:Option<&Json>)->Result<()>{self.log(PluginLogLevel::$level,target,message,fields)}})*}}
logs!(trace=>Trace,debug=>Debug,info=>Info,warn=>Warn,error=>Error);
fn micros(value: SystemTime) -> Result<i64> {
    let micros = match value.duration_since(UNIX_EPOCH) {
        Ok(duration) => i128::try_from(duration.as_micros()),
        Err(error) => {
            let duration = error.duration();
            i128::try_from(duration.as_micros())
                .map(|micros| -micros - i128::from(duration.subsec_nanos() % 1_000 != 0))
        }
    }
    .map_err(|_| "timestamp overflow".to_owned())?;
    i64::try_from(micros).map_err(|_| "timestamp overflow".into())
}
pub struct ScopeHandle<'a> {
    pub(super) value: Json,
    _borrow: PhantomData<&'a ()>,
}
pub struct ScopeGuard<'a> {
    runtime: &'a PluginRuntime,
    handle: Option<ScopeHandle<'a>>,
}
impl<'a> ScopeGuard<'a> {
    pub fn handle(&self) -> Option<&ScopeHandle<'a>> {
        self.handle.as_ref()
    }
    pub fn close(&mut self, output: Option<&Json>, metadata: Option<&Json>) -> Result<()> {
        if let Some(handle) = &self.handle {
            self.runtime.pop_scope(handle, output, metadata)?;
            self.handle.take();
        }
        Ok(())
    }
    pub fn close_at(
        &mut self,
        output: Option<&Json>,
        metadata: Option<&Json>,
        ended_at: SystemTime,
    ) -> Result<()> {
        if let Some(handle) = &self.handle {
            self.runtime.call::<()>("scope.pop",json!({"handle":handle.value,"output":output,"metadata":metadata,"timestamp":micros(ended_at)?}))?;
            self.handle.take();
        }
        Ok(())
    }
}
impl Drop for ScopeGuard<'_> {
    fn drop(&mut self) {
        let _ = self.close(None, None);
    }
}
pub struct ScopeStack<'a> {
    runtime: &'a PluginRuntime,
    id: String,
}
impl ScopeStack<'_> {
    pub fn set_thread(&self) -> Result<()> {
        self.runtime.call("stack.set", &self.id)
    }
    pub fn with_current<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        let _guard = self.runtime.bind_scope_stack_thread(self)?;
        f()
    }
}
impl Drop for ScopeStack<'_> {
    fn drop(&mut self) {
        let _ = self.runtime.call::<()>("stack.release", &self.id);
    }
}
pub struct ScopeStackBinding<'a> {
    runtime: &'a PluginRuntime,
    id: Option<String>,
}
impl ScopeStackBinding<'_> {
    pub fn restore(mut self) -> Result<()> {
        match self.id.take() {
            Some(id) => self.runtime.call("stack.restore", id),
            None => Ok(()),
        }
    }
}
impl Drop for ScopeStackBinding<'_> {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            let _ = self.runtime.call::<()>("stack.binding_release", id);
        }
    }
}
pub struct ThreadScopeStackGuard<'a> {
    previous: Option<ScopeStackBinding<'a>>,
}
impl ThreadScopeStackGuard<'_> {
    pub fn restore(mut self) -> Result<()> {
        if let Some(previous) = self.previous.take() {
            previous.restore()?;
        }
        Ok(())
    }
}
impl Drop for ThreadScopeStackGuard<'_> {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            let _ = previous.restore();
        }
    }
}
pub struct ConditionalMiddlewareGuardrailHandle(String);
struct Gate<F>(F);
impl<F: Fn(&BTreeSet<RuntimeRegistrationKind>, &str) -> Option<String> + Send + Sync + 'static>
    a::Callback for Gate<F>
{
    fn invoke_tool(
        &self,
        _: a::Invocation,
        _: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::ToolExecutionOutcome>> {
        async { a::Result::RErr(failure("gate cannot execute tools")) }.into_ffi()
    }
    fn invoke_sync(&self, i: a::Invocation) -> a::Result<abi::Json> {
        boundary(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let kinds = decode(i.value)?;
                encode((self.0)(&kinds, i.name.as_str()))
            }))
            .unwrap_or_else(|_| Err("native gate callback panicked".into())),
        )
    }
    fn invoke(
        &self,
        _: a::Invocation,
        _: ROption<a::ContinuationBox>,
    ) -> FfiFuture<a::Result<abi::Json>> {
        async { a::Result::RErr(failure("gate is synchronous")) }.into_ffi()
    }
    fn open_stream(
        &self,
        _: a::Invocation,
        _: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::StreamBox>> {
        async { a::Result::RErr(failure("gate is synchronous")) }.into_ffi()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/checked_runtime.rs"]
mod tests;
