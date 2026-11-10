// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::checked::lifecycle_tests::{Next, raw_invocation};
use a::Callback as _;

#[tokio::test]
async fn conditional_gates_reject_async_dispatch_and_contain_author_panics() {
    let gate = Gate(
        |kinds: &BTreeSet<RuntimeRegistrationKind>, name: &str| {
            assert!(kinds.contains(&RuntimeRegistrationKind::ToolExecutionIntercept));
            assert_eq!(name, "raw-tool");
            Some("blocked".to_owned())
        },
        None,
    );
    let invocation = || raw_invocation(json!(["tool_execution_intercept"]));
    assert_eq!(
        from_abi(gate.invoke_sync(invocation()).unwrap()).unwrap(),
        json!("blocked")
    );
    for value in [Json::Null, json!(["unknown_kind"]), json!({})] {
        assert!(gate.invoke_sync(raw_invocation(value)).is_err());
    }
    assert!(gate.invoke(invocation(), Option::None).await.is_err());
    assert!(
        gate.invoke_tool(invocation(), a::ContinuationBox::new(Next))
            .await
            .is_err()
    );
    assert!(
        gate.open_stream(invocation(), a::ContinuationBox::new(Next))
            .await
            .is_err()
    );
    let gate = Gate(
        |_: &BTreeSet<RuntimeRegistrationKind>, _: &str| -> Option<String> {
            panic!("author callback panic")
        },
        None,
    );
    let error = gate.invoke_sync(invocation()).unwrap_err();
    assert!(error.message.contains("native gate callback panicked"));
}

#[tokio::test]
async fn component_shutdown_waits_for_an_admitted_dynamic_gate() {
    use a::Component as _;
    let executor = ComponentExecutor::inline("dynamic-gate-drain");
    let (entered, ready) = tokio::sync::oneshot::channel();
    let entered = Mutex::new(Some(entered));
    let (release, wait) = std::sync::mpsc::channel();
    let wait = Mutex::new(wait);
    let gate = Arc::new(Gate(
        move |_: &BTreeSet<RuntimeRegistrationKind>, _: &str| {
            entered.lock().unwrap().take().unwrap().send(()).unwrap();
            wait.lock().unwrap().recv().unwrap();
            None
        },
        Some(executor.clone()),
    ));
    let invoked = gate.clone();
    let thread = std::thread::spawn(move || invoked.invoke_sync(raw_invocation(json!([]))));
    ready.await.unwrap();
    let mut closing = Box::pin(executor.shutdown());
    assert!(futures::poll!(&mut closing).is_pending());
    assert!(!executor.is_closed());
    release.send(()).unwrap();
    thread.join().unwrap().unwrap();
    closing.await.unwrap();
    assert!(gate.invoke_sync(raw_invocation(json!([]))).is_err());
}
