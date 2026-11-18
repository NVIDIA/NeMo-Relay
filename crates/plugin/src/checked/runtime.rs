// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::{AnnotatedLlmResponse, CostEstimate, ScopeAttributes, Usage};
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
    pub(super) executor: Option<ComponentExecutor>,
}
impl PluginRuntime {
    /// Adapt an owned, checked runtime capability supplied by a native host.
    #[doc(hidden)]
    pub fn from_abi(runtime: a::RuntimeBox) -> Self {
        Self {
            diagnostic_transport: Arc::new(runtime.diagnostics()),
            raw: Arc::new(runtime),
            diagnostic_state: Arc::new(crate::diagnostics::SdkState::default()),
            executor: None,
        }
    }

    pub fn diagnostics(&self) -> crate::diagnostics::Diagnostics {
        crate::diagnostics::Diagnostics::new(
            self.diagnostic_transport.clone(),
            self.diagnostic_state.clone(),
        )
    }
    /// Serialize a runtime operation and decode the host's checked response.
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
    /// Get the current scope handle or fail if the host reports no current scope.
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
    /// Push a scope and return a guard that closes it when dropped.
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
    /// Validate metric measurements and emit them using Relay's reserved metric schema.
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
    /// Estimate cost with the host's active model pricing catalog.
    ///
    /// Returns `Ok(None)` when the catalog has no pricing for the model. Hosts
    /// older than this operation return an error, which callers should treat
    /// as an unknown cost.
    pub fn estimate_cost_for_provider(
        &self,
        provider: Option<&str>,
        model: &str,
        usage: &Usage,
    ) -> Result<Option<CostEstimate>> {
        self.call(
            "pricing.estimate_cost_for_provider",
            json!({"provider":provider,"model":model,"usage":usage}),
        )
    }
    /// Price a normalized response with the host's full pricing rules.
    ///
    /// Returns the response's reported cost when present; otherwise the host's
    /// estimate for its model and usage, or `Ok(None)` when unknown. Hosts older
    /// than this operation return an error, which callers should treat as an
    /// unknown cost.
    pub fn response_cost_for_provider(
        &self,
        response: &AnnotatedLlmResponse,
        provider: Option<&str>,
    ) -> Result<Option<CostEstimate>> {
        self.call(
            "pricing.response_cost_for_provider",
            json!({"response":response,"provider":provider}),
        )
    }
    /// Send a process log record with optional structured fields through the host runtime capability.
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
    /// Create an isolated host-owned scope stack whose handle releases it when dropped.
    pub fn create_scope_stack(&self) -> Result<ScopeStack<'_>> {
        Ok(ScopeStack {
            runtime: self,
            id: self.call("stack.create", Json::Null)?,
        })
    }
    /// Capture this thread's scope binding so it can be restored later.
    pub fn capture_scope_stack_thread(&self) -> Result<ScopeStackBinding<'_>> {
        Ok(ScopeStackBinding {
            runtime: self,
            id: Some(self.call("stack.capture", Json::Null)?),
        })
    }
    /// Install an isolated stack until the returned guard restores the previous thread binding.
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
    /// Register a revocable synchronous gate scoped to this component's lifetime.
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
            callback: a::CallbackBox::new(Gate(callback, self.executor.clone())),
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
/// Convert a system timestamp to signed Unix microseconds, flooring fractional values before the epoch.
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
    /// Close this scope and retain its handle when the host reports a cleanup failure.
    pub fn close(&mut self, output: Option<&Json>, metadata: Option<&Json>) -> Result<()> {
        if let Some(handle) = &self.handle {
            self.runtime.pop_scope(handle, output, metadata)?;
            self.handle.take();
        }
        Ok(())
    }
    /// Pop this scope at an explicit timestamp and release its handle only after successful cleanup.
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
    /// Run a synchronous closure with this stack selected, then restore the previous thread binding.
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
    /// Restore the captured thread scope binding and consume its host-issued token.
    pub fn restore(mut self) -> Result<()> {
        match self.id.take() {
            Some(id) => self.runtime.call("stack.restore", id),
            None => Ok(()),
        }
    }
}
impl Drop for ScopeStackBinding<'_> {
    /// Release an unused binding token without changing the current thread scope stack.
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
    /// Restore the previous scope stack explicitly, leaving guard drop with no remaining work.
    pub fn restore(mut self) -> Result<()> {
        if let Some(previous) = self.previous.take() {
            previous.restore()?;
        }
        Ok(())
    }
}
impl Drop for ThreadScopeStackGuard<'_> {
    /// Restore the previous thread scope stack when the guard leaves scope.
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            let _ = previous.restore();
        }
    }
}
pub struct ConditionalMiddlewareGuardrailHandle(String);
struct Gate<F>(F, Option<ComponentExecutor>);
impl<F: Fn(&BTreeSet<RuntimeRegistrationKind>, &str) -> Option<String> + Send + Sync + 'static>
    a::Callback for Gate<F>
{
    fn invoke_tool(
        &self,
        _: a::Invocation,
        _: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::ToolExecutionOutcome>> {
        async { a::Result::Err(failure("gate cannot execute tools")) }.into_ffi()
    }
    /// Hold component admission during a synchronous gate call and contain author panics.
    fn invoke_sync(&self, i: a::Invocation) -> a::Result<abi::Json> {
        boundary(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _admission = self
                    .1
                    .as_ref()
                    .map(|executor| local(executor.enter_sync()))
                    .transpose()?;
                let kinds = decode(i.value)?;
                encode((self.0)(&kinds, i.name.as_str()))
            }))
            .unwrap_or_else(|_| Err("native gate callback panicked".into())),
        )
    }
    fn invoke(
        &self,
        _: a::Invocation,
        _: Option<a::ContinuationBox>,
    ) -> FfiFuture<a::Result<abi::Json>> {
        async { a::Result::Err(failure("gate is synchronous")) }.into_ffi()
    }
    fn open_stream(
        &self,
        _: a::Invocation,
        _: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::StreamBox>> {
        async { a::Result::Err(failure("gate is synchronous")) }.into_ffi()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/checked_runtime.rs"]
mod tests;

impl PluginRuntime {
    /// Enable caller-poll timing for Python streams while the guard is alive.
    ///
    /// Relay records when a Python caller requests and receives each streamed
    /// chunk only while at least one recorder holds this guard.
    pub fn enable_stream_consumer_timing(&self) -> Result<StreamConsumerTimingGuard> {
        self.call::<Json>("stream_consumer_clock.enable", Json::Null)?;
        Ok(StreamConsumerTimingGuard {
            runtime: self.clone(),
        })
    }

    /// Capture the caller-poll clock of the stream call that is running now.
    ///
    /// Call this from an LLM stream execution intercept. Returns `None` when
    /// the caller has no clock, such as a non-Python caller or timing that is
    /// not enabled.
    pub fn capture_stream_consumer_clock(&self) -> Result<Option<StreamConsumerClock>> {
        let handle: Option<u64> = self.call("stream_consumer_clock.capture", Json::Null)?;
        Ok(handle.map(|handle| StreamConsumerClock {
            runtime: self.clone(),
            handle,
        }))
    }
}

/// Keeps caller-poll timing enabled; dropping it disables timing again.
pub struct StreamConsumerTimingGuard {
    runtime: PluginRuntime,
}

impl Drop for StreamConsumerTimingGuard {
    fn drop(&mut self) {
        let _ = self
            .runtime
            .call::<Json>("stream_consumer_clock.disable", Json::Null);
    }
}

/// Caller-side timing for one captured stream.
pub struct StreamConsumerClock {
    runtime: PluginRuntime,
    handle: u64,
}

/// When the caller received each chunk and when it stopped reading.
#[derive(Debug, Clone)]
pub struct StreamConsumerReport {
    /// Times the caller received each chunk, in order.
    pub delivered: Vec<std::time::Instant>,
    /// Time the caller reached the end of the stream or stopped reading.
    pub ended: std::time::Instant,
    /// True only when the caller read the final item of the stream.
    pub complete: bool,
}

#[derive(serde::Deserialize)]
struct StreamConsumerReceipt {
    delivered_us_ago: Vec<u64>,
    ended_us_ago: u64,
    complete: bool,
}

impl StreamConsumerClock {
    /// Whether chunk `index` became available before the caller asked for it.
    pub fn was_prefetched(&self, index: usize) -> Result<bool> {
        self.runtime.call(
            "stream_consumer_clock.prefetched",
            json!({"handle": self.handle, "index": index}),
        )
    }

    /// Wait until the caller finishes reading, then report its timing.
    ///
    /// The clock stays usable afterward; dropping it releases the handle.
    pub async fn finished(&self) -> Result<StreamConsumerReport> {
        let value = self
            .runtime
            .raw
            .call(
                "stream_consumer_clock.finished".into(),
                encode(json!({"handle": self.handle}))?,
            )
            .await;
        let receipt: StreamConsumerReceipt = decode(local(value)?)?;
        let now = std::time::Instant::now();
        let at = |us_ago: u64| {
            now.checked_sub(std::time::Duration::from_micros(us_ago))
                .unwrap_or(now)
        };
        Ok(StreamConsumerReport {
            delivered: receipt.delivered_us_ago.into_iter().map(at).collect(),
            ended: at(receipt.ended_us_ago),
            complete: receipt.complete,
        })
    }
}

impl Drop for StreamConsumerClock {
    fn drop(&mut self) {
        let _ = self.runtime.call::<Json>(
            "stream_consumer_clock.release",
            json!({"handle": self.handle}),
        );
    }
}
