// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Application-facing context propagation uses the SDK alone.
use nemo_relay::{context, scope, subscriber};
use std::sync::{Arc, Mutex};

#[tokio::test(flavor = "current_thread")]
async fn concurrent_requests_and_worker_threads_keep_their_parentage() {
    nemo_relay::initialize_runtime();
    let events = Arc::new(Mutex::new(Vec::new()));
    let observed = events.clone();
    subscriber::register_subscriber(
        "sdk-context",
        Arc::new(move |event| {
            observed.lock().unwrap().push(event.clone());
        }),
    )
    .unwrap();
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let mut requests = Vec::new();
    for name in ["request-a", "request-b"] {
        let barrier = barrier.clone();
        requests.push(tokio::spawn(context::with_task_scope_stack(
            context::create_scope_stack(),
            async move {
                let handle = scope::push_scope(
                    scope::PushScopeParams::builder()
                        .name(name)
                        .scope_type(scope::ScopeType::Agent)
                        .build(),
                )
                .unwrap();
                barrier.wait().await;
                tokio::task::yield_now().await;
                assert_eq!(scope::get_handle().unwrap().uuid, handle.uuid);
                scope::event(
                    scope::EmitMarkEventParams::builder()
                        .name(&format!("{name}-task"))
                        .build(),
                )
                .unwrap();
                let binding = context::propagate_scope_to_thread().unwrap();
                std::thread::spawn(move || {
                    context::set_thread_scope_stack(binding);
                    scope::event(
                        scope::EmitMarkEventParams::builder()
                            .name(&format!("{name}-thread"))
                            .build(),
                    )
                    .unwrap();
                })
                .join()
                .unwrap();
                let propagation = context::capture_propagation_context().unwrap();
                assert_eq!(propagation.parent_uuid, handle.uuid);
                scope::pop_scope(
                    scope::PopScopeParams::builder()
                        .handle_uuid(&handle.uuid)
                        .build(),
                )
                .unwrap();
                (name, handle.uuid)
            },
        )));
    }
    let mut parents = Vec::new();
    for request in requests {
        parents.push(request.await.unwrap());
    }
    subscriber::flush_subscribers().unwrap();
    subscriber::deregister_subscriber("sdk-context").unwrap();
    let events = events.lock().unwrap();
    for (name, expected_parent) in parents {
        for suffix in ["task", "thread"] {
            let event = events
                .iter()
                .find(|event| event.name() == format!("{name}-{suffix}"))
                .unwrap();
            assert_eq!(event.parent_uuid(), Some(expected_parent));
        }
    }
}
