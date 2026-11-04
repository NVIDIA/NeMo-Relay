// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_relay_runtime::api::runtime::current_scope_stack;

#[tokio::test]
async fn execution_next_context_restores_scope() {
    let scope_stack = nemo_relay_runtime::api::runtime::create_scope_stack();
    let context = nemo_relay_runtime::api::runtime::with_scope_stack(scope_stack.clone(), || {
        MiddlewareContinuationContext::capture()
    });

    let observed =
        tokio::spawn(async move { context.run(async move { current_scope_stack() }).await })
            .await
            .unwrap();

    assert!(Arc::ptr_eq(&observed, &scope_stack));
}

fn json_completion() -> (
    CallCompletion,
    tokio::sync::oneshot::Receiver<FlowResult<Json>>,
) {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    (
        CallCompletion::Json(JsonCompletion {
            sender,
            finished: Arc::new(AtomicBool::new(false)),
        }),
        receiver,
    )
}

fn assert_sender_dropped(mut receiver: tokio::sync::oneshot::Receiver<FlowResult<Json>>) {
    assert!(matches!(
        receiver.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Closed)
    ));
}

#[test]
fn cancel_drops_a_json_call_that_never_settles() {
    let registry = CallRegistry::default();
    let call_id = registry.allocate();
    let (completion, receiver) = json_completion();
    registry.insert(call_id, completion);
    registry.cancel(call_id);
    registry.resolve(call_id, || Ok(Json::Null));
    assert!(registry.stream(call_id).is_none());
    assert_sender_dropped(receiver);
}

#[test]
fn cancel_keeps_a_stream_call() {
    let registry = CallRegistry::default();
    let call_id = registry.allocate();
    let (ready_sender, _ready_receiver) = tokio::sync::oneshot::channel();
    let (chunk_sender, _chunk_receiver) = tokio::sync::mpsc::channel(1);
    let (closed, _closed_rx) = tokio::sync::watch::channel(None);
    let completion = CallCompletion::Stream(Arc::new(StreamCompletion {
        ready: Mutex::new(Some(ready_sender)),
        chunks: Mutex::new(Some(chunk_sender)),
        terminal_error: Mutex::new(None),
        closed,
        finished: Arc::default(),
        parked: Mutex::new(None),
    }));
    registry.insert(call_id, completion);
    registry.cancel(call_id);
    assert!(registry.stream(call_id).is_some());
}
