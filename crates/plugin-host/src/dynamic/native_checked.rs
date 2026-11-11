// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host adapters for the checked native contract.
#![allow(missing_docs)]
use a::Component as _;
use futures_util::{
    FutureExt as _, Stream, StreamExt,
    future::{BoxFuture, Shared},
};
use nemo_relay_plugin::{
    abi::{
        self,
        future::{FfiFuture, FutureExt as _},
        v1_0 as a,
    },
    abi_executor::{ComponentExecutor, DirectCancellation},
    abi_value::{from_abi, into_abi},
};
use nemo_relay_runtime::api::{
    registry::{
        RuntimeRegistrationKind, deregister_conditional_middleware_guardrail,
        list_runtime_registrations, register_conditional_middleware_guardrail,
    },
    runtime::*,
    scope::*,
};
use nemo_relay_types::{
    Json,
    error::{FlowError, Result},
};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::json;
use std::{
    any::Any,
    collections::{BTreeSet, HashMap},
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
};
use uuid::Uuid;

fn err(value: impl ToString) -> FlowError {
    FlowError::InvalidArgument(value.to_string())
}
fn encode(value: impl Serialize) -> Result<abi::Json> {
    serde_json::to_value(value).map(into_abi).map_err(err)
}
/// Decode a host value from the checked JSON boundary, preserving conversion failures.
fn decode<T: DeserializeOwned>(value: abi::Json) -> Result<T> {
    serde_json::from_value(from_abi(value).map_err(err)?).map_err(err)
}
/// Convert a runtime failure to an ABI error while preserving an upstream HTTP status.
fn abi_error(error: FlowError) -> a::Error {
    let mut out = a::Error::new("runtime", super::http_error::to_message(&error));
    if let FlowError::Upstream(failure) = error {
        out.http_status = failure.status;
    }
    out
}
fn boundary<T>(value: Result<T>) -> a::Result<T> {
    value.map_err(abi_error)
}
/// Restore a runtime failure from its ABI message and optional upstream HTTP status.
fn local<T>(value: a::Result<T>) -> Result<T> {
    value.map_err(|e| {
        let message = e.message.to_string();
        if let Some(status) = e.http_status
            && super::http_error::upstream_from_message(&message).is_none()
        {
            return FlowError::Upstream(nemo_relay_types::error::UpstreamFailure {
                status: Some(status),
                body: message,
                headers: Default::default(),
                class: nemo_relay_types::error::UpstreamFailureClass::Other,
            });
        }
        super::http_error::from_message(message)
    })
}

pub struct HostRuntime {
    active: Arc<AtomicBool>,
    executor: ComponentExecutor,
    namespace: String,
    plugin_kind: String,
    state: Mutex<RuntimeState>,
    diagnostics: super::native::CheckedDiagnostics,
    registration: Mutex<Option<tokio::sync::mpsc::Sender<RegistrationRequest>>>,
    component: Arc<Mutex<Option<a::ComponentBox>>>,
    streams: Mutex<Vec<Arc<HostStreamState>>>,
    observer: Mutex<super::diagnostics::PluginDiagnostics>,
}
#[derive(Default)]
struct RuntimeState {
    stacks: HashMap<String, ScopeStackHandle>,
    bindings: HashMap<String, (ThreadScopeStackBinding, Option<ScopeStackHandle>)>,
    gates: HashMap<String, String>,
    scopes: Vec<OwnedScope>,
    callbacks: Vec<Weak<CallbackOwner>>,
}
// Registry snapshots retain a revocable host proxy for each foreign callback.
struct CallbackOwner {
    value: Mutex<Option<Arc<a::CallbackBox>>>,
}
impl CallbackOwner {
    /// Admit a foreign call and retain its callback until the call returns.
    fn with<T>(&self, owner: &HostRuntime, invoke: impl FnOnce(&a::CallbackBox) -> T) -> Result<T> {
        let _call = local(owner.executor.enter_sync())?;
        owner.ensure()?;
        let callback = self
            .value
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .cloned()
            .ok_or_else(|| err("native callback is revoked"))?;
        // Never hold the slot lock while calling foreign code or retain the
        // callback owner in an async waiter after its foreign future is created.
        Ok(invoke(&callback))
    }
}
struct OwnedScope {
    uuid: Uuid,
    context: MiddlewareContinuationContext,
}
impl HostRuntime {
    pub fn new(namespace: String) -> crate::Result<Arc<Self>> {
        Self::with_diagnostics(
            namespace,
            nemo_relay_diagnostics::Diagnostics::process_default().clone(),
            false,
            "native".into(),
        )
    }
    pub fn with_diagnostics(
        namespace: String,
        diagnostics: nemo_relay_diagnostics::Diagnostics,
        read: bool,
        plugin_kind: String,
    ) -> crate::Result<Arc<Self>> {
        Ok(Arc::new(Self {
            observer: Mutex::new(Default::default()),
            registration: Mutex::new(None),
            component: Arc::new(Mutex::new(None)),
            streams: Mutex::new(Vec::new()),
            active: Arc::new(AtomicBool::new(true)),
            executor: ComponentExecutor::for_host(),
            diagnostics: super::native::CheckedDiagnostics::new(diagnostics, read, &namespace),
            namespace,
            plugin_kind,
            state: Mutex::new(RuntimeState::default()),
        }))
    }
    /// Convert a callback result and record failures that did not originate in a downstream continuation.
    fn callback_result<T>(&self, value: a::Result<T>, downstream: &DownstreamErrors) -> Result<T> {
        if let a::Result::Err(error) = &value {
            let inherited = downstream
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains(error.message.as_str());
            if !inherited
                && self.active.load(Ordering::Acquire)
                && error.code.as_str() != "cancelled"
            {
                self.observer
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .failure(
                        nemo_relay_diagnostics::PluginOperationFailureCode::NativeCallback,
                        nemo_relay_diagnostics::FailureClass::ExecutionFailed,
                        nemo_relay_diagnostics::PluginPhase::Runtime,
                    );
            }
        }
        local(value)
    }
    pub fn registered(&self) {
        self.diagnostics.registered();
    }
    /// Reject operations after this component has revoked its runtime authority.
    fn ensure(&self) -> Result<()> {
        if self.active.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(err("native runtime capability is revoked"))
        }
    }
    /// Retain a revocable foreign callback while this component's runtime authority is live.
    fn own_callback(&self, callback: a::CallbackBox) -> Result<Arc<CallbackOwner>> {
        let callback = Arc::new(CallbackOwner {
            value: Mutex::new(Some(Arc::new(callback))),
        });
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.ensure()?;
        // Weak entries avoid a cycle when a callback captures its runtime.
        state.callbacks.retain(|entry| entry.strong_count() > 0);
        state.callbacks.push(Arc::downgrade(&callback));
        Ok(callback)
    }
    /// Release retained foreign callback state after component cleanup has completed.
    fn release_callbacks(&self) {
        let callbacks = std::mem::take(
            &mut self
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .callbacks,
        );
        for callback in callbacks.into_iter().filter_map(|entry| entry.upgrade()) {
            let value = callback
                .value
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            drop(value);
        }
    }
    pub fn runtime_box(self: &Arc<Self>) -> a::RuntimeBox {
        a::RuntimeBox::new(RuntimeProxy(self.clone()))
    }
    fn scope(&self) -> a::ScopeBox {
        a::ScopeBox::new(HostScope {
            active: self.active.clone(),
            context: Arc::new(Mutex::new(ScopeBinding::capture())),
        })
    }
    /// Stop runtime admission and remove owned gates, retaining gates whose removal failed.
    pub fn revoke(&self) -> crate::Result<()> {
        self.active.store(false, Ordering::Release);
        self.executor.stop_admission();
        self.diagnostics.revoke();
        self.registration
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let gates = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.stacks.clear();
            state.bindings.clear();
            std::mem::take(&mut state.gates)
        };
        let mut retained = HashMap::new();
        let mut failures = vec![];
        for (id, name) in gates {
            if let Err(error) = deregister_conditional_middleware_guardrail(&name) {
                failures.push(error.to_string());
                retained.insert(id, name);
            }
        }
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .gates
            .extend(retained);
        if failures.is_empty() {
            Ok(())
        } else {
            Err(crate::PluginError::RegistrationFailed(failures.join("; ")))
        }
    }
    /// Drain component work, close streams and scopes, then stop the executor and release callbacks.
    /// Failed resource cleanup retains ownership so the caller can retry.
    pub async fn shutdown(&self) -> crate::Result<()> {
        self.revoke()?;
        self.executor.drain_user_tasks().await;
        let streams = self
            .streams
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        for state in streams {
            local(state.close(&self.executor).await)
                .map_err(|e| crate::PluginError::RegistrationFailed(e.to_string()))?;
        }
        self.streams
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.close_scopes()?;
        local(self.executor.shutdown().await)
            .map_err(|e| crate::PluginError::RegistrationFailed(e.to_string()))?;
        self.release_callbacks();
        Ok(())
    }
    /// Unwind owned scopes on their original stacks and retain any scopes that could not close.
    fn close_scopes(&self) -> crate::Result<()> {
        // Revoked SDK guards cannot call back into the runtime. Retain host
        // ownership until every scope is unwound on its original stack.
        let scopes =
            std::mem::take(&mut self.state.lock().unwrap_or_else(|e| e.into_inner()).scopes);
        let mut retained = Vec::new();
        let mut failures = Vec::new();
        for scope in scopes.into_iter().rev() {
            let result = scope.context.run_sync(|| {
                let present = current_scope_stack_snapshot()?
                    .is_some_and(|stack| stack.scopes.iter().any(|v| v.scope_id == scope.uuid));
                if present {
                    pop_scope(PopScopeParams::builder().handle_uuid(&scope.uuid).build())?;
                }
                Ok::<_, FlowError>(())
            });
            if let Err(error) = result {
                failures.push(error.to_string());
                retained.push(scope);
            }
        }
        retained.reverse();
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .scopes
            .extend(retained);
        if failures.is_empty() {
            Ok(())
        } else {
            Err(crate::PluginError::RegistrationFailed(failures.join("; ")))
        }
    }
    /// Dispatch an authorized runtime operation and validate its arguments before changing host state.
    fn operation(&self, operation: &str, args: Json) -> Result<Json> {
        self.ensure()?;
        let null = || Ok(Json::Null);
        let string = |field: &str| {
            args[field]
                .as_str()
                .ok_or_else(|| err(format!("missing {field}")))
        };
        let optional = |field: &str| args.get(field).filter(|v| !v.is_null()).cloned();
        let timestamp = || {
            args.get("timestamp")
                .filter(|v| !v.is_null())
                .map(|v| {
                    let micros = v.as_i64().ok_or_else(|| err("invalid timestamp"))?;
                    chrono::DateTime::from_timestamp_micros(micros)
                        .ok_or_else(|| err("invalid timestamp"))
                })
                .transpose()
        };
        let parent = || -> Result<Option<ScopeHandle>> {
            optional("parent")
                .map(|value| {
                    let uuid = serde_json::from_value(value["uuid"].clone()).map_err(err)?;
                    // Core parent resolution uses only the scope identity.
                    Ok(ScopeHandle::builder()
                        .uuid(uuid)
                        .name("")
                        .scope_type(ScopeType::Unknown)
                        .build())
                })
                .transpose()
        };
        match operation {
            "scope.active" => Ok(json!(scope_stack_active())),
            "scope.snapshot" => serde_json::to_value(current_scope_stack_snapshot()?).map_err(err),
            "scope.current" => Ok(json!({"uuid":get_handle()?.uuid})),
            "scope.push" => {
                let scope_type = serde_json::from_value(args["scope_type"].clone()).map_err(err)?;
                let attributes = args
                    .get("attributes")
                    .filter(|v| !v.is_null())
                    .map(|v| {
                        v.as_u64()
                            .and_then(|v| u32::try_from(v).ok())
                            .and_then(ScopeAttributes::from_bits)
                            .ok_or_else(|| err("invalid scope attributes"))
                    })
                    .transpose()?
                    .unwrap_or_else(ScopeAttributes::empty);
                let parent = parent()?;
                let handle = push_scope(
                    PushScopeParams::builder()
                        .parent_opt(parent.as_ref())
                        .attributes(attributes)
                        .name(string("name")?)
                        .scope_type(scope_type)
                        .data_opt(optional("data"))
                        .metadata_opt(optional("metadata"))
                        .input_opt(optional("input"))
                        .timestamp_opt(timestamp()?)
                        .build(),
                )?;
                self.state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .scopes
                    .push(OwnedScope {
                        uuid: handle.uuid,
                        context: MiddlewareContinuationContext::capture(),
                    });
                Ok(json!({"uuid":handle.uuid}))
            }
            "scope.pop" => {
                let id: Uuid =
                    serde_json::from_value(args["handle"]["uuid"].clone()).map_err(err)?;
                pop_scope(
                    PopScopeParams::builder()
                        .handle_uuid(&id)
                        .output_opt(optional("output"))
                        .metadata_opt(optional("metadata"))
                        .timestamp_opt(timestamp()?)
                        .build(),
                )?;
                self.state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .scopes
                    .retain(|scope| scope.uuid != id);
                null()
            }
            "mark" => {
                let schema = optional("data_schema")
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(err)?;
                let severity = optional("severity")
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(err)?;
                let parent = parent()?;
                event(
                    EmitMarkEventParams::builder()
                        .parent_opt(parent.as_ref())
                        .timestamp_opt(timestamp()?)
                        .name(string("name")?)
                        .data_opt(optional("data"))
                        .metadata_opt(optional("metadata"))
                        .data_schema_opt(schema)
                        .severity_opt(severity)
                        .build(),
                )?;
                null()
            }
            "registry.list" => {
                let kinds: Option<BTreeSet<RuntimeRegistrationKind>> =
                    serde_json::from_value(args).map_err(err)?;
                serde_json::to_value(list_runtime_registrations(kinds.as_ref())?).map_err(err)
            }
            "runtime.diagnostics" => {
                Ok(json!({"entries": crate::active_runtime_diagnostics_snapshot()}))
            }
            "stack.create" => {
                let id = Uuid::now_v7().to_string();
                self.state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .stacks
                    .insert(id.clone(), create_scope_stack());
                Ok(json!(id))
            }
            "stack.capture" => {
                let id = Uuid::now_v7().to_string();
                self.state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .bindings
                    .insert(
                        id.clone(),
                        (capture_thread_scope_stack(), current_scope_stack_override()),
                    );
                Ok(json!(id))
            }
            "stack.set" => {
                let id = args.as_str().ok_or_else(|| err("invalid stack handle"))?;
                let stack = self
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .stacks
                    .get(id)
                    .cloned()
                    .ok_or_else(|| err("unknown stack"))?;
                install_scope_stack_override(stack.clone());
                set_thread_scope_stack(stack);
                null()
            }
            "stack.restore" => {
                let id = args.as_str().ok_or_else(|| err("invalid binding"))?;
                let binding = self
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .bindings
                    .remove(id)
                    .ok_or_else(|| err("unknown binding"))?;
                restore_thread_scope_stack(binding.0);
                restore_scope_stack_override(binding.1);
                null()
            }
            "stack.release" => {
                let id = args.as_str().ok_or_else(|| err("invalid stack"))?;
                self.state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .stacks
                    .remove(id);
                null()
            }
            "stack.binding_release" => {
                let id = args.as_str().ok_or_else(|| err("invalid binding"))?;
                self.state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .bindings
                    .remove(id);
                null()
            }
            "gate.remove" => {
                let id = args.as_str().ok_or_else(|| err("invalid gate"))?;
                let name = self
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .gates
                    .get(id)
                    .cloned();
                if let Some(name) = name {
                    deregister_conditional_middleware_guardrail(&name)?;
                    self.state
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .gates
                        .remove(id);
                    Ok(json!(true))
                } else {
                    Ok(json!(false))
                }
            }
            "log" => {
                let target = string("target")?;
                if target.len() > 256 || target.bytes().any(|b| b == 0 || b.is_ascii_control()) {
                    return Err(err("invalid plugin log target"));
                }
                let target = if target.is_empty() {
                    "nemo_relay.plugin.native".to_owned()
                } else {
                    format!("nemo_relay.plugin.native.{target}")
                };
                let level = match args["level"].as_u64() {
                    Some(0) => log::Level::Error,
                    Some(1) => log::Level::Warn,
                    Some(2) => log::Level::Info,
                    Some(3) => log::Level::Debug,
                    Some(4) => log::Level::Trace,
                    _ => return Err(err("invalid log level")),
                };
                let fields = optional("fields").unwrap_or(json!({}));
                let fields = fields
                    .as_object()
                    .ok_or_else(|| err("plugin log fields must be an object"))?;
                let fields = LogFields(fields);
                let message = string("message")?;
                let args = format_args!("{message}");
                log::logger().log(
                    &log::Record::builder()
                        .args(args)
                        .target(&target)
                        .level(level)
                        .key_values(&fields)
                        .build(),
                );
                null()
            }
            _ => Err(err(format!(
                "unsupported native runtime operation '{operation}'"
            ))),
        }
    }
}
struct LogFields<'a>(&'a serde_json::Map<String, Json>);
impl log::kv::Source for LogFields<'_> {
    /// Expose structured plugin log fields to the host logger without flattening their JSON values.
    fn visit<'kvs>(
        &'kvs self,
        visitor: &mut dyn log::kv::VisitSource<'kvs>,
    ) -> std::result::Result<(), log::kv::Error> {
        for (k, v) in self.0 {
            visitor.visit_pair(log::kv::Key::from_str(k), log::kv::Value::from_serde(v))?;
        }
        Ok(())
    }
}
struct RegistrationRequest {
    registration: a::Registration,
    reply: std::sync::mpsc::SyncSender<a::Result<()>>,
}

/// Apply registration requests until the foreign factory settles, canceling it if admission closes.
async fn drive_registration(
    ctx: &mut dyn crate::PluginRegistrationContext,
    runtime: Arc<HostRuntime>,
    mut receive: tokio::sync::mpsc::Receiver<RegistrationRequest>,
    foreign: impl Future<Output = a::Result<a::RegisteredComponent>>,
    cancelled: tokio_util::sync::CancellationToken,
) -> a::Result<a::RegisteredComponent> {
    tokio::pin!(foreign);
    let mut open = true;
    let registered = loop {
        tokio::select! {
            biased;
            request = receive.recv(), if open => match request {
                Some(request) => {
                    let result = apply_registration(ctx, request.registration, runtime.clone())
                        .map_err(|error| a::Error::new("registration", error.to_string()));
                    let _ = request.reply.send(result);
                }
                None => {
                    open = false;
                    cancelled.cancel();
                }
            },
            result = &mut foreign => break result,
        }
    };
    receive.close();
    while let Ok(request) = receive.try_recv() {
        let _ = request.reply.send(a::Result::Err(a::Error::new(
            "closed",
            "native registration transaction has ended",
        )));
    }
    registered
}
struct RuntimeProxy(Arc<HostRuntime>);
impl a::Runtime for RuntimeProxy {
    /// Admit a synchronous runtime operation and contain host-side panics at the ABI boundary.
    fn call_sync(&self, operation: String, args: abi::Json) -> a::Result<abi::Json> {
        boundary(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Runtime capabilities can be called from foreign threads, outside
                // an SDK task. Drain admitted operations before final cleanup.
                let _call = local(self.0.executor.enter_sync())?;
                self.0
                    .operation(operation.as_str(), from_abi(args).map_err(err)?)
                    .map(into_abi)
            }))
            .unwrap_or_else(|_| Err(err("native host runtime operation panicked"))),
        )
    }
    fn call(&self, operation: String, args: abi::Json) -> FfiFuture<a::Result<abi::Json>> {
        let owner = self.0.clone();
        let binding = ScopeBinding::capture();
        self.0.executor.execute_direct(async move {
            let _lease = binding.context.enter_thread_binding(binding.explicit);
            RuntimeProxy(owner).call_sync(operation, args)
        })
    }
    fn diagnostics(&self) -> abi::diagnostics::DiagnosticsBox {
        abi::diagnostics::DiagnosticsBox::new(self.0.diagnostics.clone())
    }
    fn capture_scope(&self) -> a::Result<a::ScopeBox> {
        boundary(self.0.ensure().map(|_| self.0.scope()))
    }
    /// Bind one component owner during an active registration transaction and reject duplicate ownership.
    fn bind_component(&self, component: a::ComponentBox) -> a::Result<()> {
        if let Err(error) = self.0.ensure() {
            return a::Result::Err(abi_error(error));
        }
        if self
            .0
            .registration
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_none()
        {
            return a::Result::Err(a::Error::new(
                "closed",
                "native component binding requires an active registration transaction",
            ));
        }
        let mut slot = self.0.component.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(error) = self.0.ensure() {
            return a::Result::Err(abi_error(error));
        }
        if slot.is_some() {
            return a::Result::Err(a::Error::new(
                "conflict",
                "native component owner is already bound",
            ));
        }
        *slot = Some(component);
        a::Result::Ok(())
    }
    /// Send a callback to the active registration transaction and wait for its admission result.
    fn register_callback(&self, registration: a::Registration) -> a::Result<()> {
        let sender = self
            .0
            .registration
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let Some(sender) = sender else {
            return a::Result::Err(a::Error::new(
                "closed",
                "native registration transaction has ended",
            ));
        };
        let (send, receive) = std::sync::mpsc::sync_channel(1);
        match sender.try_send(RegistrationRequest {
            registration,
            reply: send,
        }) {
            Err(_) => a::Result::Err(a::Error::new(
                "capacity",
                "native registration queue is closed or full",
            )),
            Ok(()) => receive.recv().unwrap_or_else(|_| {
                a::Result::Err(a::Error::new(
                    "closed",
                    "native registration transaction was cancelled",
                ))
            }),
        }
    }
    /// Register a revocable conditional gate within this component's runtime authority and namespace.
    fn register_gate(&self, registration: a::Registration) -> a::Result<String> {
        boundary((|| {
            self.0.ensure()?;
            if registration.surface != a::Surface::ConditionalMiddleware {
                return Err(err(
                    "runtime authority only permits dynamic conditional gates",
                ));
            }
            let id = Uuid::now_v7().to_string();
            let name = format!(
                "{}{}",
                self.0.namespace,
                crate::encode_plugin_component_field(registration.name.as_str())
            );
            let callback =
                gate_callback(self.0.own_callback(registration.callback)?, self.0.clone());
            let options = from_abi(registration.options).map_err(err)?;
            let kinds = serde_json::from_value(options["kinds"].clone()).map_err(err)?;
            let target = options["registration_name"]
                .as_str()
                .ok_or_else(|| err("missing gate target"))?;
            let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
            self.0.ensure()?;
            register_conditional_middleware_guardrail(&name, kinds, target, callback)?;
            state.gates.insert(id.clone(), name);
            Ok(id)
        })())
    }
}
#[derive(Clone)]
struct ScopeBinding {
    context: MiddlewareContinuationContext,
    explicit: bool,
}
impl ScopeBinding {
    fn capture() -> Self {
        Self {
            context: MiddlewareContinuationContext::capture(),
            explicit: scope_stack_active(),
        }
    }
}
struct HostScope {
    active: Arc<AtomicBool>,
    context: Arc<Mutex<ScopeBinding>>,
}
struct HostLease {
    guard: Option<Box<dyn Any>>,
    context: Arc<Mutex<ScopeBinding>>,
    restored: bool,
}
impl HostLease {
    /// Restore the previous thread binding once and save scope changes made during the invocation.
    fn restore_inner(&mut self) {
        if self.restored {
            return;
        }
        self.restored = true;
        *self.context.lock().unwrap_or_else(|e| e.into_inner()) = ScopeBinding::capture();
        self.guard.take();
    }
}
impl Drop for HostLease {
    fn drop(&mut self) {
        self.restore_inner();
    }
}
impl a::ScopeLease for HostLease {
    fn restore(&mut self) -> a::Result<()> {
        self.restore_inner();
        a::Result::Ok(())
    }
}
impl a::Scope for HostScope {
    /// Bind the captured Relay scope to this thread until the returned lease restores the previous binding.
    fn enter(&self) -> a::Result<a::ScopeLeaseBox> {
        if !self.active.load(Ordering::Acquire) {
            return a::Result::Err(a::Error::new("revoked", "component scope is revoked"));
        }
        let context = self
            .context
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let guard =
            Some(Box::new(context.context.enter_thread_binding(context.explicit)) as Box<dyn Any>);
        a::Result::Ok(a::ScopeLeaseBox::new(HostLease {
            guard,
            context: self.context.clone(),
            restored: false,
        }))
    }
    /// Read the captured scope stack while component authority is active; implicit scopes have no snapshot.
    fn snapshot(&self) -> a::Result<Option<abi::Json>> {
        if !self.active.load(Ordering::Acquire) {
            return a::Result::Err(a::Error::new("revoked", "component scope is revoked"));
        }
        let binding = self
            .context
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        boundary(if binding.explicit {
            binding
                .context
                .run_sync(|| current_scope_stack_snapshot().and_then(|v| v.map(encode).transpose()))
        } else {
            Ok(Option::None)
        })
    }
}
fn guarded<T>(call: impl FnOnce() -> Result<T>) -> a::Result<T> {
    boundary(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(call))
            .unwrap_or_else(|_| Err(err("native host codec callback panicked"))),
    )
}
struct Codec {
    request: Option<LlmRequestContext>,
    response: Option<LlmResponseContext>,
    active: Arc<AtomicBool>,
    component_active: Arc<AtomicBool>,
}
impl Codec {
    /// Require both invocation and component authority before a codec operation.
    fn ensure_active(&self) -> Result<()> {
        if self.active.load(Ordering::Acquire) && self.component_active.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(err("codec capability expired"))
        }
    }
}
impl a::Codec for Codec {
    /// Return codec identity and availability while its invocation and component remain active.
    fn identity(&self) -> a::Result<abi::Json> {
        if self.ensure_active().is_err() {
            return a::Result::Err(a::Error::new("revoked", "codec capability expired"));
        }
        guarded(|| {
            if let Some(v) = &self.request {
                encode(json!({"identity":v.codec(),"available":v.resolve_codec().is_some()}))
            } else {
                let v = self.response.as_ref().unwrap();
                encode(json!({"identity":v.codec(),"available":v.resolve_codec().is_some()}))
            }
        })
    }
    /// Decode request or response data with the host codec after checking invocation and component authority.
    fn decode(&self, value: abi::Json) -> a::Result<abi::Json> {
        guarded(|| {
            self.ensure_active()?;
            if let Some(v) = &self.request {
                let codec = v.resolve_codec().ok_or_else(|| err("no request codec"))?;
                encode(codec.decode(&decode(value)?)?)
            } else {
                let codec = self
                    .response
                    .as_ref()
                    .unwrap()
                    .resolve_codec()
                    .ok_or_else(|| err("no response codec"))?;
                encode(codec.decode_response(&from_abi(value).map_err(err)?)?)
            }
        })
    }
    /// Encode an annotated request with its original provider fields through a live codec capability.
    fn encode(&self, annotated: abi::Json, original: abi::Json) -> a::Result<abi::Json> {
        guarded(|| {
            self.ensure_active()?;
            let codec = self
                .request
                .as_ref()
                .and_then(|v| v.resolve_codec())
                .ok_or_else(|| err("no request encoder"))?;
            encode(codec.encode(&decode(annotated)?, &decode(original)?)?)
        })
    }
}
struct InvocationLifetime {
    active: Arc<AtomicBool>,
}
impl Drop for InvocationLifetime {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
    }
}
fn invocation(
    owner: &HostRuntime,
    surface: a::Surface,
    name: String,
    value: abi::Json,
    id: Option<String>,
    request: Option<LlmRequestContext>,
    response: Option<LlmResponseContext>,
) -> (a::Invocation, InvocationLifetime) {
    let active = Arc::new(AtomicBool::new(true));
    let request_codec = request.map(|request| {
        a::CodecBox::new(Codec {
            request: Some(request),
            response: None,
            active: active.clone(),
            component_active: owner.active.clone(),
        })
    });
    let response_codec = response.map(|response| {
        a::CodecBox::new(Codec {
            request: None,
            response: Some(response),
            active: active.clone(),
            component_active: owner.active.clone(),
        })
    });
    (
        a::Invocation {
            surface,
            name,
            value,
            tool_call_id: id,
            context: a::Context {
                scope: owner.scope(),
                request_codec,
                response_codec,
            },
        },
        InvocationLifetime { active },
    )
}
/// Build an admitted synchronous gate proxy that fails open on callback errors.
fn gate_callback(
    callback: Arc<CallbackOwner>,
    owner: Arc<HostRuntime>,
) -> ConditionalMiddlewareGuardrailFn {
    Arc::new(move |kinds, name| {
        let result = owner.ensure().and_then(|_| {
            let _admission = local(owner.executor.enter_sync())?;
            let (i, _life) = invocation(
                &owner,
                a::Surface::ConditionalMiddleware,
                name.to_owned(),
                encode(kinds)?,
                None,
                None,
                None,
            );
            decode(owner.callback_result(
                callback.with(&owner, |callback| callback.invoke_sync(i))?,
                &DownstreamErrors::default(),
            )?)
        });
        result.unwrap_or_else(|_| {log::warn!(target:"nemo_relay.plugin",event="native_conditional_middleware_guardrail_failed",plugin_id=owner.plugin_kind.as_str();"Native conditional middleware guardrail failed open");None})
    })
}
enum Next {
    Tool(ToolExecutionNextFn),
    Llm(LlmExecutionNextFn),
    Stream(LlmStreamExecutionNextFn),
}
type DownstreamErrors = Arc<Mutex<BTreeSet<String>>>;
/// Remember downstream failures before converting them to checked ABI errors.
fn downstream_boundary<T>(result: Result<T>, errors: &DownstreamErrors) -> a::Result<T> {
    if let Err(error) = &result {
        errors
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(super::http_error::to_message(error));
    }
    boundary(result)
}
struct Continuation {
    next: Arc<Next>,
    lease: MiddlewareContinuationLease,
    downstream: DownstreamErrors,
    owner: Arc<HostRuntime>,
}
impl a::Continuation for Continuation {
    /// Poll a unary continuation in host runtime context and retain the origin of downstream failures.
    fn call(&self, value: abi::Json) -> FfiFuture<a::Result<abi::Json>> {
        let errors = self.downstream.clone();
        let begin = self.owner.ensure().and_then(|_| self.lease.begin());
        let next = self.next.clone();
        self.owner.executor.execute_direct(async move {
            downstream_boundary(
                async {
                    let begin = begin?;
                    let context = begin.context().clone();
                    host_bound(
                        context,
                        begin.invoke(|| async move {
                            match next.as_ref() {
                                Next::Tool(callback) => {
                                    encode(callback(from_abi(value).map_err(err)?).await?)
                                }
                                Next::Llm(callback) => encode(callback(decode(value)?).await?),
                                Next::Stream(_) => {
                                    Err(err("stream continuation requires open_stream"))
                                }
                            }
                        }),
                    )
                    .await
                }
                .await,
                &errors,
            )
        })
    }
    /// Invoke a tool continuation and preserve its typed result and optional annotation.
    fn call_tool(&self, value: abi::Json) -> FfiFuture<a::Result<a::ToolExecutionResult>> {
        let errors = self.downstream.clone();
        let begin = self.owner.ensure().and_then(|_| self.lease.begin());
        let next = self.next.clone();
        self.owner.executor.execute_direct(async move {
            downstream_boundary(
                async {
                    let begin = begin?;
                    let context = begin.context().clone();
                    host_bound(
                        context,
                        begin.invoke(|| async move {
                            match next.as_ref() {
                                Next::Tool(callback) => {
                                    let value = callback(from_abi(value).map_err(err)?).await?;
                                    Ok(a::ToolExecutionResult {
                                        result: into_abi(value.result),
                                        annotation: value
                                            .annotation
                                            .filter(|v| !v.is_null())
                                            .map(into_abi),
                                    })
                                }
                                _ => Err(err("not a tool continuation")),
                            }
                        }),
                    )
                    .await
                }
                .await,
                &errors,
            )
        })
    }
    /// Open a downstream stream under the captured continuation context and track it for component cleanup.
    fn open_stream(&self, value: abi::Json) -> FfiFuture<a::Result<a::StreamBox>> {
        let errors = self.downstream.clone();
        let begin = self.owner.ensure().and_then(|_| self.lease.begin());
        let next = self.next.clone();
        let owner = self.owner.clone();
        self.owner.executor.execute_direct(async move {
            downstream_boundary(
                async {
                    let begin = begin?;
                    let context = begin.context().clone();
                    let stream = host_bound(
                        context.clone(),
                        begin.invoke(|| async move {
                            match next.as_ref() {
                                Next::Stream(callback) => callback(decode(value)?).await,
                                _ => Err(err("unary continuation cannot open a stream")),
                            }
                        }),
                    )
                    .await?;
                    let state = Arc::new(HostStreamState {
                        value: tokio::sync::Mutex::new(Some(stream)),
                        context,
                        cancelled: tokio_util::sync::CancellationToken::new(),
                        terminal: AtomicBool::new(false),
                        closed: AtomicBool::new(false),
                        closing: Mutex::new(None),
                        downstream: errors.clone(),
                        pulls: Mutex::new(Vec::new()),
                    });
                    {
                        let mut streams = owner.streams.lock().unwrap_or_else(|e| e.into_inner());
                        streams.retain(|v| !v.closed.load(Ordering::Acquire));
                        streams.push(state.clone());
                    }
                    Ok(a::StreamBox::new(HostStream { state, owner }))
                }
                .await,
                &errors,
            )
        })
    }
}
// Inline continuations are polled inside the plugin's thread scope binding.
// Clear that override so the continuation's task context and nested scopes can apply.
struct HostBound<F> {
    context: MiddlewareContinuationContext,
    future: Option<Pin<Box<F>>>,
}
impl<F: Future> Future for HostBound<F> {
    type Output = F::Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let _binding = this.context.enter_thread_binding(false);
        this.future.as_mut().unwrap().as_mut().poll(cx)
    }
}
impl<F> Drop for HostBound<F> {
    fn drop(&mut self) {
        let _binding = self.context.enter_thread();
        self.future.take();
    }
}
fn host_bound<F: Future>(context: MiddlewareContinuationContext, future: F) -> HostBound<F> {
    HostBound {
        context,
        future: Some(Box::pin(future)),
    }
}

type HostCloseFuture = Shared<BoxFuture<'static, a::Result<()>>>;
struct HostStreamState {
    value: tokio::sync::Mutex<Option<LlmJsonStream>>,
    context: MiddlewareContinuationContext,
    cancelled: tokio_util::sync::CancellationToken,
    terminal: AtomicBool,
    closed: AtomicBool,
    closing: Mutex<Option<HostCloseFuture>>,
    downstream: DownstreamErrors,
    pulls: Mutex<Vec<DirectCancellation>>,
}
impl HostStreamState {
    /// Cancel further pulls and share stream cleanup, allowing a failed cleanup attempt to be retried.
    fn close(self: &Arc<Self>, executor: &ComponentExecutor) -> HostCloseFuture {
        self.cancelled.cancel();
        let pulls = std::mem::take(&mut *self.pulls.lock().unwrap_or_else(|e| e.into_inner()));
        for pull in pulls {
            pull.cancel();
        }
        self.terminal.store(true, Ordering::Release);
        let mut closing = self.closing.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(future) = closing.as_ref()
            && future.peek().is_none_or(|v| v.is_ok())
        {
            return future.clone();
        }
        let state = self.clone();
        let context = state.context.clone();
        let foreign = executor.execute_cleanup(async move {
            boundary(
                context
                    .run(async {
                        let mut stream = state.value.lock().await;
                        if let Some(stream) = stream.as_mut() {
                            stream.close().await?;
                        }
                        stream.take();
                        state.closed.store(true, Ordering::Release);
                        Ok(())
                    })
                    .await,
            )
        });
        let future = foreign.boxed().shared();
        *closing = Some(future.clone());
        future
    }
}
struct HostStream {
    state: Arc<HostStreamState>,
    owner: Arc<HostRuntime>,
}
impl a::Stream for HostStream {
    /// Poll one downstream stream item in host runtime context and make EOF or an error terminal.
    fn pull(&mut self) -> FfiFuture<a::Result<Option<abi::Json>>> {
        if let Err(error) = self.owner.ensure() {
            return async move { a::Result::Err(abi_error(error)) }.into_ffi();
        }
        let mut pulls = self.state.pulls.lock().unwrap_or_else(|e| e.into_inner());
        if self.state.cancelled.is_cancelled() {
            return async { a::Result::Err(a::Error::new("closed", "native stream is closing")) }
                .into_ffi();
        }
        let state = self.state.clone();
        let context = state.context.clone();
        let (future, cancellation) = self.owner.executor.execute_direct_tracked(async move {
            downstream_boundary(host_bound(context.clone(), context.run(async {
                if state.terminal.load(Ordering::Acquire) { return Ok(Option::None); }
                let mut value = state.value.lock().await;
                let Some(stream) = value.as_mut() else { return Ok(Option::None); };
                tokio::select! { biased;
                    () = state.cancelled.cancelled() => Err(err("native stream is closing")),
                    next = stream.next() => match next {
                        Some(Ok(value)) => Ok(Option::Some(into_abi(value))),
                        Some(Err(error)) => { state.terminal.store(true, Ordering::Release); stream.terminalize(); Err(error) },
                        None => { state.terminal.store(true, Ordering::Release); stream.terminalize(); Ok(Option::None) },
                    }
                }
            })).await, &state.downstream)
        });
        pulls.retain(DirectCancellation::is_alive);
        pulls.push(cancellation);
        future
    }
    fn close(&mut self) -> FfiFuture<a::Result<()>> {
        self.state.close(&self.owner.executor).into_ffi()
    }
}
impl Drop for HostStream {
    fn drop(&mut self) {
        drop(self.state.close(&self.owner.executor));
    }
}

struct Call {
    surface: a::Surface,
    name: String,
    value: abi::Json,
    id: Option<String>,
    request: Option<LlmRequestContext>,
    response: Option<LlmResponseContext>,
    next: Option<Next>,
}
/// Invoke checked middleware with scoped runtime authority and optional downstream continuation.
async fn unary<T: DeserializeOwned>(
    owner: Arc<HostRuntime>,
    callback: Arc<CallbackOwner>,
    call: Call,
) -> Result<T> {
    let Call {
        surface,
        name,
        value,
        id,
        request,
        response,
        next,
    } = call;
    owner.ensure()?;
    let (invocation, _lifetime) = invocation(&owner, surface, name, value, id, request, response);
    let (_lease, _guard) = MiddlewareContinuationLease::capture();
    let plugin_kind = owner.plugin_kind.clone();
    let downstream = DownstreamErrors::default();
    let next = next.map(|next| {
        a::ContinuationBox::new(Continuation {
            next: Arc::new(next),
            lease: _lease,
            downstream: downstream.clone(),
            owner: owner.clone(),
        })
    });
    let future = callback.with(&owner, |callback| callback.invoke(invocation, next))?;
    decode(
        owner
            .callback_result(future.await, &downstream)
            .map_err(|e| super::http_error::reported_by(e, &plugin_kind, "native"))?,
    )
}

/// Adapt a synchronous native subscriber and log callback failures without interrupting event delivery.
fn subscriber_callback(callback: Arc<CallbackOwner>, owner: Arc<HostRuntime>) -> EventSubscriberFn {
    Arc::new(move |event| {
        let result = owner.ensure().and_then(|_| {
            let (i, _life) = invocation(
                &owner,
                a::Surface::Subscriber,
                String::new(),
                encode(event)?,
                None,
                None,
                None,
            );
            owner.callback_result(
                callback.with(&owner, |callback| callback.invoke_sync(i))?,
                &DownstreamErrors::default(),
            )
        });
        if result.is_err() {
            log::error!(target:"nemo_relay.plugin",event="native_subscriber_callback_failed",plugin_id=owner.plugin_kind.as_str();"Native subscriber failed");
        }
    })
}

/// Apply each descriptor through the existing host transaction and name policy.
pub fn apply_registration(
    ctx: &mut dyn crate::PluginRegistrationContext,
    registration: a::Registration,
    owner: Arc<HostRuntime>,
) -> crate::Result<()> {
    let a::Registration {
        name,
        surface,
        priority,
        break_chain,
        options,
        callback,
    } = registration;
    let name = name.as_str();
    let callback = owner
        .own_callback(callback)
        .map_err(|e| crate::PluginError::RegistrationFailed(e.to_string()))?;
    if surface == a::Surface::ConditionalMiddleware {
        let options = from_abi(options).map_err(crate::PluginError::RegistrationFailed)?;
        let kinds = serde_json::from_value(options["kinds"].clone())
            .map_err(|e| crate::PluginError::RegistrationFailed(e.to_string()))?;
        let target = options["registration_name"].as_str().ok_or_else(|| {
            crate::PluginError::RegistrationFailed("missing conditional gate target".into())
        })?;
        return ctx.register_conditional_middleware_guardrail(
            name,
            kinds,
            target,
            gate_callback(callback, owner),
        );
    }
    match surface {
        a::Surface::Subscriber => {
            ctx.register_subscriber(name, subscriber_callback(callback, owner))
        }
        a::Surface::EventMetadataInjector => ctx.register_event_metadata_injector(
            name,
            priority,
            Arc::new(move |event| {
                let owner = owner.clone();
                let callback = callback.clone();
                Box::pin(async move {
                    unary(
                        owner,
                        callback,
                        Call {
                            surface,
                            name: String::new(),
                            value: encode(&*event)?,
                            id: None,
                            request: None,
                            response: None,
                            next: None,
                        },
                    )
                    .await
                })
            }),
        ),
        a::Surface::MarkSanitize => ctx.register_mark_sanitize_guardrail(
            name,
            priority,
            Arc::new(move |event, fields| {
                let owner = owner.clone();
                let callback = callback.clone();
                Box::pin(async move {
                    unary(
                        owner,
                        callback,
                        Call {
                            surface,
                            name: String::new(),
                            value: encode(json!({"event":&*event,"fields":fields}))?,
                            id: None,
                            request: None,
                            response: None,
                            next: None,
                        },
                    )
                    .await
                })
            }),
        ),
        a::Surface::ScopeSanitizeStart => ctx.register_scope_sanitize_start_guardrail(
            name,
            priority,
            Arc::new(move |event, fields| {
                let owner = owner.clone();
                let callback = callback.clone();
                Box::pin(async move {
                    unary(
                        owner,
                        callback,
                        Call {
                            surface,
                            name: String::new(),
                            value: encode(json!({"event":&*event,"fields":fields}))?,
                            id: None,
                            request: None,
                            response: None,
                            next: None,
                        },
                    )
                    .await
                })
            }),
        ),
        a::Surface::ScopeSanitizeEnd => ctx.register_scope_sanitize_end_guardrail(
            name,
            priority,
            Arc::new(move |event, fields| {
                let owner = owner.clone();
                let callback = callback.clone();
                Box::pin(async move {
                    unary(
                        owner,
                        callback,
                        Call {
                            surface,
                            name: String::new(),
                            value: encode(json!({"event":&*event,"fields":fields}))?,
                            id: None,
                            request: None,
                            response: None,
                            next: None,
                        },
                    )
                    .await
                })
            }),
        ),
        a::Surface::ToolSanitizeRequest => ctx.register_tool_sanitize_request_guardrail(
            name,
            priority,
            Arc::new(move |name, value| {
                let owner = owner.clone();
                let callback = callback.clone();
                Box::pin(async move {
                    unary(
                        owner,
                        callback,
                        Call {
                            surface,
                            name,
                            value: into_abi(value),
                            id: None,
                            request: None,
                            response: None,
                            next: None,
                        },
                    )
                    .await
                })
            }),
        ),
        a::Surface::ToolSanitizeResponse => ctx.register_tool_sanitize_response_guardrail(
            name,
            priority,
            Arc::new(move |name, value| {
                let owner = owner.clone();
                let callback = callback.clone();
                Box::pin(async move {
                    unary(
                        owner,
                        callback,
                        Call {
                            surface,
                            name,
                            value: into_abi(value),
                            id: None,
                            request: None,
                            response: None,
                            next: None,
                        },
                    )
                    .await
                })
            }),
        ),
        a::Surface::ToolConditionalExecution => ctx.register_tool_conditional_execution_guardrail(
            name,
            priority,
            Arc::new(move |name, value| {
                let owner = owner.clone();
                let callback = callback.clone();
                Box::pin(async move {
                    unary(
                        owner,
                        callback,
                        Call {
                            surface,
                            name,
                            value: into_abi(value),
                            id: None,
                            request: None,
                            response: None,
                            next: None,
                        },
                    )
                    .await
                })
            }),
        ),
        a::Surface::ToolRequestIntercept => ctx.register_tool_request_intercept(
            name,
            priority,
            break_chain,
            Arc::new(move |name, value| {
                let owner = owner.clone();
                let callback = callback.clone();
                Box::pin(async move {
                    unary(
                        owner,
                        callback,
                        Call {
                            surface,
                            name,
                            value: into_abi(value),
                            id: None,
                            request: None,
                            response: None,
                            next: None,
                        },
                    )
                    .await
                })
            }),
        ),
        a::Surface::ToolExecutionIntercept => ctx.register_tool_execution_intercept(
            name,
            priority,
            Arc::new(move |context, next| {
                let owner = owner.clone();
                let callback = callback.clone();
                Box::pin(async move {
                    owner.ensure()?;
                    let (i, _life) = invocation(
                        &owner,
                        surface,
                        context.tool_name().to_owned(),
                        encode(context.args())?,
                        context.tool_call_id().map(str::to_owned),
                        None,
                        None,
                    );
                    let (lease, _guard) = MiddlewareContinuationLease::capture();
                    let downstream = DownstreamErrors::default();
                    let next = a::ContinuationBox::new(Continuation {
                        next: Arc::new(Next::Tool(next)),
                        lease,
                        downstream: downstream.clone(),
                        owner: owner.clone(),
                    });
                    let future = callback.with(&owner, |callback| callback.invoke_tool(i, next))?;
                    let value = owner
                        .callback_result(future.await, &downstream)
                        .map_err(|e| {
                            super::http_error::reported_by(e, &owner.plugin_kind, "native")
                        })?;
                    Ok(nemo_relay_types::api::tool::ToolExecutionInterceptOutcome {
                        result: from_abi(value.result.result).map_err(err)?,
                        annotation: value
                            .result
                            .annotation
                            .map(from_abi)
                            .transpose()
                            .map_err(err)?
                            .filter(|v| !v.is_null()),
                        pending_marks: value
                            .pending_marks
                            .into_iter()
                            .map(decode)
                            .collect::<Result<Vec<_>>>()?,
                    })
                })
            }),
        ),
        a::Surface::LlmConditionalExecution => ctx.register_llm_conditional_execution_guardrail(
            name,
            priority,
            Arc::new(move |request| {
                let owner = owner.clone();
                let callback = callback.clone();
                Box::pin(async move {
                    unary(
                        owner,
                        callback,
                        Call {
                            surface,
                            name: String::new(),
                            value: encode(request)?,
                            id: None,
                            request: None,
                            response: None,
                            next: None,
                        },
                    )
                    .await
                })
            }),
        ),
        a::Surface::LlmRequestIntercept => ctx.register_llm_request_intercept(
            name,
            priority,
            break_chain,
            Arc::new(move |name, request, annotated| {
                let owner = owner.clone();
                let callback = callback.clone();
                Box::pin(async move {
                    unary(
                        owner,
                        callback,
                        Call {
                            surface,
                            name,
                            value: encode(json!({"request":request,"annotated":annotated}))?,
                            id: None,
                            request: None,
                            response: None,
                            next: None,
                        },
                    )
                    .await
                })
            }),
        ),
        a::Surface::LlmSanitizeRequest => ctx.register_llm_sanitize_request_guardrail(
            name,
            priority,
            Arc::new(move |request, context| {
                let owner = owner.clone();
                let callback = callback.clone();
                Box::pin(async move {
                    unary(
                        owner,
                        callback,
                        Call {
                            surface,
                            name: String::new(),
                            value: encode(request)?,
                            id: None,
                            request: Some(context),
                            response: None,
                            next: None,
                        },
                    )
                    .await
                })
            }),
        ),
        a::Surface::LlmSanitizeResponse => ctx.register_llm_sanitize_response_guardrail(
            name,
            priority,
            Arc::new(move |response, context| {
                let owner = owner.clone();
                let callback = callback.clone();
                Box::pin(async move {
                    unary(
                        owner,
                        callback,
                        Call {
                            surface,
                            name: String::new(),
                            value: into_abi(response),
                            id: None,
                            request: None,
                            response: Some(context),
                            next: None,
                        },
                    )
                    .await
                })
            }),
        ),
        a::Surface::LlmExecutionIntercept => ctx.register_llm_execution_intercept(
            name,
            priority,
            Arc::new(move |name, request, context, next| {
                let name = name.to_owned();
                let owner = owner.clone();
                let callback = callback.clone();
                Box::pin(async move {
                    unary(
                        owner,
                        callback,
                        Call {
                            surface,
                            name,
                            value: encode(request)?,
                            id: None,
                            request: Some(context.request_codec().clone()),
                            response: context.response_codec().cloned(),
                            next: Some(Next::Llm(next)),
                        },
                    )
                    .await
                })
            }),
        ),

        a::Surface::LlmStreamExecutionIntercept => ctx.register_llm_stream_execution_intercept(
            name,
            priority,
            Arc::new(move |name, request, context, next| {
                let name = name.to_owned();
                let owner = owner.clone();
                let callback = callback.clone();
                Box::pin(async move {
                    owner.ensure()?;
                    let (invocation, life) = invocation(
                        &owner,
                        surface,
                        name,
                        encode(request)?,
                        None,
                        Some(context.request_codec().clone()),
                        None,
                    );
                    let (lease, guard) = MiddlewareContinuationLease::capture();
                    let downstream = DownstreamErrors::default();
                    let next = a::ContinuationBox::new(Continuation {
                        next: Arc::new(Next::Stream(next)),
                        lease,
                        downstream: downstream.clone(),
                        owner: owner.clone(),
                    });
                    let plugin_kind = owner.plugin_kind.clone();
                    let observer = owner.clone();
                    let errors = downstream.clone();
                    let first = owner.executor.execute_direct(async move {
                        let future = match callback
                            .with(&observer, |callback| callback.open_stream(invocation, next))
                        {
                            Ok(future) => future,
                            Err(error) => return a::Result::Err(abi_error(error)),
                        };
                        let stream = match observer
                            .callback_result(future.await, &errors)
                            .map_err(|e| super::http_error::reported_by(e, &plugin_kind, "native"))
                        {
                            Ok(v) => Arc::new(Mutex::new(v)),
                            Err(e) => return a::Result::Err(abi_error(e)),
                        };
                        let pull = stream.lock().unwrap_or_else(|e| e.into_inner()).pull();
                        let first = pull.await;
                        a::Result::Ok(OpenedStream { stream, first })
                    });
                    let opening = Box::pin(async move { local(first.await) });
                    Ok(LlmJsonStream::from_closeable(ForeignStream {
                        stream: None,
                        opening: Some(opening),
                        pending: None,
                        terminal: false,
                        _lifetime: Some(Box::new((life, guard))),
                        owner,
                        downstream,
                    }))
                })
            }),
        ),
        a::Surface::ConditionalMiddleware => unreachable!(),
    }
}

struct OpenedStream {
    stream: Arc<Mutex<a::StreamBox>>,
    first: a::Result<Option<abi::Json>>,
}
struct ForeignStream {
    stream: Option<Arc<Mutex<a::StreamBox>>>,
    opening: Option<Pin<Box<dyn Future<Output = Result<OpenedStream>> + Send>>>,
    pending: Option<FfiFuture<a::Result<Option<abi::Json>>>>,
    terminal: bool,
    _lifetime: Option<Box<dyn Any + Send>>,
    owner: Arc<HostRuntime>,
    downstream: DownstreamErrors,
}
impl Unpin for ForeignStream {}
impl Stream for ForeignStream {
    type Item = Result<Json>;
    /// Open the foreign stream lazily and poll one item while retaining its invocation capabilities.
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.terminal {
            return Poll::Ready(None);
        }
        if self.stream.is_none() {
            let result = match self.opening.as_mut() {
                Some(opening) => match opening.as_mut().poll(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(v) => v,
                },
                None => return Poll::Ready(None),
            };
            self.opening.take();
            match result {
                Ok(opened) => {
                    self.stream = Some(opened.stream);
                    self.pending = Some(async move { opened.first }.into_ffi());
                }
                Err(error) => {
                    self.terminal = true;
                    self._lifetime.take();
                    return Poll::Ready(Some(Err(error)));
                }
            }
        }
        if self.pending.is_none() {
            let future = self
                .stream
                .as_ref()
                .unwrap()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .pull();
            self.pending = Some(future);
        }
        let result = match Pin::new(self.pending.as_mut().unwrap()).poll(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(v) => v,
        };
        self.pending.take();
        match self.owner.callback_result(result, &self.downstream) {
            Ok(Option::Some(v)) => Poll::Ready(Some(from_abi(v).map_err(err))),
            Ok(Option::None) => {
                self.terminal = true;
                self._lifetime.take();
                Poll::Ready(None)
            }
            Err(e) => {
                self.terminal = true;
                self._lifetime.take();
                Poll::Ready(Some(Err(super::http_error::reported_by(
                    e,
                    &self.owner.plugin_kind,
                    "native",
                ))))
            }
        }
    }
}
impl LlmStreamInner for ForeignStream {
    fn terminalize(mut self: Pin<&mut Self>) {
        self._lifetime.take();
    }
    /// Cancel pending foreign reads and release invocation capabilities after stream cleanup succeeds.
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(async move {
            let this = self.get_mut();
            this.pending.take();
            this.opening.take();
            if let Some(stream) = &this.stream {
                let future = stream.lock().unwrap_or_else(|e| e.into_inner()).close();
                local(future.await)?;
            }
            this.terminal = true;
            this._lifetime.take();
            Ok(())
        })
    }
}
impl Drop for ForeignStream {
    /// Schedule foreign stream cleanup and retain the invocation lifetime until that cleanup task finishes.
    fn drop(&mut self) {
        self.pending.take();
        self.opening.take();
        let lifetime = self._lifetime.take();
        if let Some(stream) = self.stream.take() {
            drop(self.owner.executor.execute_cleanup(async move {
                let _lifetime = lifetime;
                let future = stream.lock().unwrap_or_else(|e| e.into_inner()).close();
                future.await
            }));
        }
    }
}

/// Checked plugin definition with fresh component state for each registration.
pub struct CheckedPlugin {
    kind: String,
    resident_path: std::path::PathBuf,
    plugin: Arc<a::PluginBox>,
    diagnostics: nemo_relay_diagnostics::Diagnostics,
    read: bool,
    observer: super::diagnostics::PluginDiagnostics,
}
impl CheckedPlugin {
    pub(super) fn resident_path(&self) -> &std::path::Path {
        &self.resident_path
    }
    pub fn new(
        library: &super::native_library::CheckedNativeLibrary,
        diagnostics: nemo_relay_diagnostics::Diagnostics,
        read: bool,
    ) -> crate::Result<Self> {
        Self::with_observer(library, diagnostics, read, Default::default())
    }
    /// Create a fresh plugin object tied to resident code and the activation's diagnostic service.
    pub(super) fn with_observer(
        library: &super::native_library::CheckedNativeLibrary,
        diagnostics: nemo_relay_diagnostics::Diagnostics,
        read: bool,
        observer: super::diagnostics::PluginDiagnostics,
    ) -> crate::Result<Self> {
        Ok(Self {
            kind: library.descriptor().plugin_kind.to_string(),
            resident_path: library.resident_path().to_owned(),
            plugin: Arc::new(
                library
                    .create_plugin()
                    .map_err(crate::PluginError::RegistrationFailed)?,
            ),
            diagnostics,
            read,
            observer,
        })
    }
}
struct ComponentOwner {
    component: Arc<Mutex<Option<a::ComponentBox>>>,
    runtime: Arc<HostRuntime>,
}
impl ComponentOwner {
    /// Run component teardown on a dedicated thread and retain ownership when cleanup fails or exceeds its deadline.
    fn close(&self) -> crate::Result<()> {
        let revoked = self.runtime.revoke();
        std::thread::scope(|scope| {
            let task=std::thread::Builder::new().name("nemo-relay-native-close".into()).spawn_scoped(scope,||{
                let runtime=tokio::runtime::Builder::new_current_thread().enable_all().build().map_err(|e|crate::PluginError::Internal(e.to_string()))?;
                runtime.block_on(async{
                    tokio::time::timeout(std::time::Duration::from_secs(5),async{
                        // Inline foreign callbacks must leave before their
                        // component can release resources or stop its executor.
                        self.runtime.executor.drain_user_tasks().await;
                        let future=self.component.lock().unwrap_or_else(|e|e.into_inner()).as_ref().map(|c|c.shutdown());
                        if let Some(future)=future{local(future.await).map_err(|e|crate::PluginError::RegistrationFailed(e.to_string()))?;}
                        self.runtime.shutdown().await?;
                        revoked?;
                        self.component.lock().unwrap_or_else(|e|e.into_inner()).take();
                        Ok(())
                    }).await.map_err(|_|crate::PluginError::RegistrationFailed("native cleanup is still in progress; ownership retained for close retry".into()))?
                })
            }).map_err(|e|crate::PluginError::Internal(format!("cannot start native teardown thread: {e}")))?;
            task.join().map_err(|_| {
                crate::PluginError::Internal("native teardown thread panicked".into())
            })?
        })
    }
}
impl crate::Plugin for CheckedPlugin {
    fn plugin_kind(&self) -> &str {
        &self.kind
    }
    fn allows_multiple_components(&self) -> bool {
        self.plugin.allows_multiple_components()
    }
    /// Convert checked plugin validation results to host diagnostics and record whether validation succeeded.
    fn validate(&self, config: &serde_json::Map<String, Json>) -> Vec<crate::ConfigDiagnostic> {
        let diagnostics: Vec<crate::ConfigDiagnostic> = self
            .plugin
            .validate(into_abi(Json::Object(config.clone())))
            .into_iter()
            .map(|d| crate::ConfigDiagnostic {
                level: if d.is_error {
                    crate::DiagnosticLevel::Error
                } else {
                    crate::DiagnosticLevel::Warning
                },
                code: d.code.to_string(),
                component: d.component.map(|s| s.to_string()),
                field: d.field.map(|s| s.to_string()),
                message: d.message.to_string(),
            })
            .collect();
        if diagnostics
            .iter()
            .any(|d| d.level == crate::DiagnosticLevel::Error)
        {
            self.observer.failure(
                nemo_relay_diagnostics::PluginOperationFailureCode::NativeValidation,
                nemo_relay_diagnostics::FailureClass::InvalidInput,
                nemo_relay_diagnostics::PluginPhase::Validation,
            );
        } else {
            self.observer.success(
                nemo_relay_diagnostics::SuccessCode::PluginNativeValidated,
                nemo_relay_diagnostics::PluginPhase::Validation,
            );
        }
        diagnostics
    }
    /// Create fresh runtime authority and register a component through a bounded host transaction.
    fn register<'a>(
        &'a self,
        config: &serde_json::Map<String, Json>,
        ctx: &'a mut dyn crate::PluginRegistrationContext,
    ) -> crate::PluginFuture<'a, ()> {
        let config = config.clone();
        Box::pin(async move {
            let result = async {
            let runtime = HostRuntime::with_diagnostics(
                ctx.qualify_name(""),
                self.diagnostics.clone(),
                self.read,
                self.kind.clone(),
            )?;
            *runtime.observer.lock().unwrap_or_else(|e|e.into_inner()) = self.observer.clone();
            let owner = Arc::new(ComponentOwner {
                component: runtime.component.clone(),
                runtime: runtime.clone(),
            });
            let cleanup = owner.clone();
            ctx.add_registration(crate::PluginRegistration::new(
                "native_component",
                ctx.qualify_name(""),
                Box::new(move || cleanup.close()),
            ));
            let (sender, receive) = tokio::sync::mpsc::channel::<RegistrationRequest>(32);
            *runtime
                .registration
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(sender);
            let plugin = self.plugin.clone();
            let registration_runtime = runtime.runtime_box();
            let binding = ScopeBinding::capture();
            // Raw plugins can synchronously submit registrations while constructing
            // or polling this future. Keep that work off the transaction driver.
            let cancelled = tokio_util::sync::CancellationToken::new();
            let guard = cancelled.clone().drop_guard();
            let cancellation = cancelled.clone();
            let work = runtime.executor.execute_blocking(move || {
                let _binding = binding.context.enter_thread_binding(binding.explicit);
                tokio::runtime::Handle::current().block_on(async move {
                    tokio::select! {
                        biased;
                        () = cancelled.cancelled() => a::Result::Err(a::Error::new("cancelled", "native registration was cancelled")),
                        result = plugin.register(into_abi(Json::Object(config)), registration_runtime) => result,
                    }
                })
            });
            let foreign = async move {
                let _guard = guard;
                work.await
            };
            let registered = local(drive_registration(ctx, runtime.clone(), receive, foreign, cancellation).await);
            runtime
                .registration
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            let registered =
                registered.map_err(|e| crate::PluginError::RegistrationFailed(e.to_string()))?;
            {
                let mut slot = owner.component.lock().unwrap_or_else(|e| e.into_inner());
                if slot.is_none() {
                    *slot = Some(registered.component);
                }
            }
            if let Some(error) = registered.error {
                return Err(crate::PluginError::RegistrationFailed(
                    error.message.to_string(),
                ));
            }
            for registration in registered.registrations {
                apply_registration(ctx, registration, runtime.clone())?;
            }
            runtime.registered();
            self.observer.success(
                nemo_relay_diagnostics::SuccessCode::PluginNativeRegistered,
                nemo_relay_diagnostics::PluginPhase::Activation,
            );
            Ok(())
            }.await;
            if result.is_err() {
                self.observer.failure(
                    nemo_relay_diagnostics::PluginOperationFailureCode::NativeRegistration,
                    nemo_relay_diagnostics::FailureClass::ExecutionFailed,
                    nemo_relay_diagnostics::PluginPhase::Activation,
                );
            }
            result
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/native_checked_adapter_tests.rs"]
mod tests;
