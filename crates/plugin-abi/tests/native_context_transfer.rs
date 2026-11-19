// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The concrete native context retains ownership and lossless JSON behavior.
use nemo_relay_plugin_abi::{
    Field, Json,
    future::{FfiFuture, FutureExt as _},
    transfer::Value,
    v1_0 as a,
    xabi::XabiType as _,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Scope(Arc<AtomicUsize>);
impl Drop for Scope {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
impl a::Scope for Scope {
    fn enter(&self) -> a::Result<a::ScopeLeaseBox> {
        unreachable!()
    }
    fn snapshot(&self) -> a::Result<Option<Json>> {
        Ok(None)
    }
}
fn invocation(drops: Arc<AtomicUsize>, context: Option<Json>) -> a::Invocation {
    a::Invocation {
        surface: a::Surface::ToolExecutionIntercept,
        name: "files.write".into(),
        value: Json::String("business input".into()),
        tool_call_id: Some("provider-call-123".into()),
        context: a::Context {
            scope: a::ScopeBox::new(Scope(drops)),
            request_codec: None,
            response_codec: None,
        },
        execution_context: context.map(|value| {
            let Json::Object(fields) = value else {
                unreachable!()
            };
            a::ApplicationExecutionContext { data: Some(fields) }
        }),
    }
}

#[test]
fn tool_context_is_separate_and_lossless_through_owned_record_transfer() {
    let drops = Arc::new(AtomicUsize::new(0));
    let context = Json::Object(vec![Field {
        key: "example.application".into(),
        value: Json::Array(vec![
            Json::Unsigned(u64::MAX),
            Json::Null,
            Json::String("雪\0".into()),
        ]),
    }]);
    let payload = Value(invocation(drops.clone(), Some(context.clone()))).into_payload();
    let decoded = unsafe { Value::<a::Invocation>::from_payload(payload) }
        .unwrap()
        .0;
    assert_eq!(decoded.value, Json::String("business input".into()));
    assert_eq!(decoded.tool_call_id.as_deref(), Some("provider-call-123"));
    let Json::Object(fields) = context else {
        unreachable!()
    };
    assert_eq!(
        decoded.execution_context.as_ref().unwrap().data,
        Some(fields)
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(decoded);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

struct Next(Arc<AtomicUsize>);
impl Drop for Next {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
impl a::Continuation for Next {
    fn call(&self, _: Json) -> FfiFuture<a::Result<Json>> {
        unreachable!()
    }
    fn call_tool(&self, _: Json) -> FfiFuture<a::Result<a::ToolExecutionResult>> {
        unreachable!()
    }
    fn open_stream(&self, _: Json) -> FfiFuture<a::Result<a::StreamBox>> {
        unreachable!()
    }
}
struct Callback(Arc<AtomicUsize>);
impl a::Callback for Callback {
    fn invoke(
        &self,
        _: a::Invocation,
        _: Option<a::ContinuationBox>,
    ) -> FfiFuture<a::Result<Json>> {
        unreachable!()
    }
    fn invoke_sync(&self, _: a::Invocation) -> a::Result<Json> {
        unreachable!()
    }
    fn open_stream(
        &self,
        _: a::Invocation,
        _: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::StreamBox>> {
        unreachable!()
    }

    fn invoke_tool(
        &self,
        invocation: a::Invocation,
        next: a::ContinuationBox,
    ) -> FfiFuture<a::Result<a::ToolExecutionOutcome>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        async move {
            let _invocation = invocation;
            let _next = next;
            std::future::pending().await
        }
        .into_ffi()
    }
}

#[test]
fn malformed_context_releases_nested_scope_and_later_continuation_without_callback() {
    let scopes = Arc::new(AtomicUsize::new(0));
    let continuations = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let payload = Value(a::CallbackBox::new(Callback(calls.clone()))).into_payload();
    let callback = unsafe { Value::<a::CallbackBox>::from_payload(payload) }
        .unwrap()
        .0;
    let future = callback.invoke_tool(
        invocation(
            scopes.clone(),
            Some(Json::Object(vec![Field {
                key: "invalid".into(),
                value: Json::Float(f64::NAN),
            }])),
        ),
        a::ContinuationBox::new(Next(continuations.clone())),
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(scopes.load(Ordering::SeqCst), 1);
    assert_eq!(continuations.load(Ordering::SeqCst), 1);
    drop(future);
}

#[test]
fn unpolled_callback_future_releases_invocation_and_continuation() {
    let scopes = Arc::new(AtomicUsize::new(0));
    let continuations = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let payload = Value(a::CallbackBox::new(Callback(calls.clone()))).into_payload();
    let callback = unsafe { Value::<a::CallbackBox>::from_payload(payload) }
        .unwrap()
        .0;
    let future = callback.invoke_tool(
        invocation(scopes.clone(), None),
        a::ContinuationBox::new(Next(continuations.clone())),
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(scopes.load(Ordering::SeqCst), 0);
    assert_eq!(continuations.load(Ordering::SeqCst), 0);
    drop(callback);
    drop(future);
    assert_eq!(scopes.load(Ordering::SeqCst), 1);
    assert_eq!(continuations.load(Ordering::SeqCst), 1);
}
