// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scope admission, deactivation, and blocked native-operation regressions.

use super::*;

fn credentials() -> Option<pb::Credentials> {
    Some(pb::Credentials {
        activation_id: ACTIVATION_ID.into(),
        auth_token: AUTH_TOKEN.into(),
    })
}

fn push_request(stack: &str, parent: &str, name: &str) -> Request<PushScopeRequest> {
    Request::new(PushScopeRequest {
        credentials: credentials(),
        scope: Some(ScopeContext {
            scope_stack_id: stack.into(),
            parent_scope_id: parent.into(),
        }),
        name: name.into(),
        scope_type: ProtoScopeType::Function as i32,
        ..Default::default()
    })
}

async fn create_stack(service: &WorkerHostRuntimeService) -> String {
    service
        .create_scope_stack(Request::new(CreateScopeStackRequest {
            credentials: credentials(),
        }))
        .await
        .unwrap()
        .into_inner()
        .scope_stack_id
}

struct SubscriberCleanup(&'static str);
impl Drop for SubscriberCleanup {
    fn drop(&mut self) {
        let _ = nemo_relay_runtime::api::subscriber::deregister_subscriber(self.0);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn successful_shutdown_unwinds_owned_scopes_and_rejects_new_scope_work() {
    let (instance, _worker_shutdown) = fake_worker_instance(Vec::new()).await;
    let service = WorkerHostRuntimeService {
        state: instance.host_state.clone(),
    };
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let subscriber = "worker-scope-lifetime-shutdown";
    nemo_relay_runtime::api::subscriber::register_subscriber(subscriber, {
        let events = events.clone();
        Arc::new(move |event| {
            if event.name().starts_with("lifetime-owned-") {
                events.lock().unwrap().push(event.clone());
            }
        })
    })
    .unwrap();
    let _subscriber = SubscriberCleanup(subscriber);
    let stack_id = create_stack(&service).await;
    let a = service
        .push_scope(push_request(&stack_id, "", "lifetime-owned-A"))
        .await
        .unwrap()
        .into_inner()
        .scope_handle_id;
    service
        .push_scope(push_request(&stack_id, "", "lifetime-owned-B"))
        .await
        .unwrap();
    service
        .push_scope(push_request(&stack_id, &a, "lifetime-owned-C"))
        .await
        .unwrap();
    let stack = instance.host_state.stack(&stack_id).unwrap().unwrap();
    let outcome = instance.shutdown_checked();
    assert!(outcome.safe_to_unload, "{outcome:?}");
    assert!(instance.host_state.scope_stacks.lock().unwrap().is_empty());
    assert!(instance.host_state.scope_handles.lock().unwrap().is_empty());
    assert_eq!(stack.read().unwrap().scopes().len(), 1);
    nemo_relay_runtime::api::subscriber::flush_subscribers().unwrap();
    let mut ended = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.scope_category() == Some(ScopeCategory::End))
        .map(|event| event.name().to_owned())
        .collect::<Vec<_>>();
    ended.sort();
    assert_eq!(
        ended,
        ["lifetime-owned-A", "lifetime-owned-B", "lifetime-owned-C"]
    );
    assert!(
        service
            .create_scope_stack(Request::new(CreateScopeStackRequest {
                credentials: credentials(),
            }))
            .await
            .is_err()
    );
    assert!(
        service
            .push_scope(push_request(&stack_id, "", "after-shutdown"))
            .await
            .is_err()
    );
    assert!(instance.host_state.scope_stacks.lock().unwrap().is_empty());
    assert!(instance.host_state.scope_handles.lock().unwrap().is_empty());
}

#[test]
fn dropping_stack_revokes_an_admitted_push_blocked_on_the_native_stack() {
    let runtime = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let state = Arc::new(WorkerHostRuntimeState::new(
        ACTIVATION_ID.into(),
        AUTH_TOKEN.into(),
    ));
    let service = WorkerHostRuntimeService {
        state: state.clone(),
    };
    let stack_id = runtime.block_on(create_stack(&service));
    let stack = state.stack(&stack_id).unwrap().unwrap();
    let baseline = Arc::strong_count(&stack);
    let held = stack.write().unwrap();
    let pushing = {
        let runtime = runtime.clone();
        let state = state.clone();
        let request = push_request(&stack_id, "", "late-lifetime-push");
        std::thread::spawn(move || {
            runtime.block_on(WorkerHostRuntimeService { state }.push_scope(request))
        })
    };
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while Arc::strong_count(&stack) == baseline && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert!(
        Arc::strong_count(&stack) > baseline,
        "push must select its native context before revocation"
    );
    // Dropping revokes admission without waiting for this native lock. The
    // admitted push owns deferred cleanup until it can observe revocation.
    runtime
        .block_on(
            service.drop_scope_stack(Request::new(DropScopeStackRequest {
                credentials: credentials(),
                scope_stack_id: stack_id,
            })),
        )
        .unwrap();
    drop(held);
    assert!(
        pushing.join().unwrap().is_err(),
        "a revoked push must not publish a handle"
    );
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let root_only = stack
                    .try_read()
                    .is_ok_and(|stack| stack.scopes().len() == 1);
                if root_only
                    && state.scope_stacks.lock().unwrap().is_empty()
                    && state.scope_handles.lock().unwrap().is_empty()
                {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("revoked push must release its scope and handle after native progress resumes");
    });
}

#[test]
fn scope_drain_bounds_wait_for_a_blocked_push_and_retains_cleanup_ownership() {
    let runtime = RuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let state = Arc::new(WorkerHostRuntimeState::new(
        ACTIVATION_ID.into(),
        AUTH_TOKEN.into(),
    ));
    let service = WorkerHostRuntimeService {
        state: state.clone(),
    };
    let stack_id = runtime.block_on(create_stack(&service));
    let stack = state.stack(&stack_id).unwrap().unwrap();
    let baseline = Arc::strong_count(&stack);
    let held = stack.write().unwrap();
    let pushing = {
        let runtime = runtime.handle().clone();
        let state = state.clone();
        let request = push_request(&stack_id, "", "blocked-shutdown-push");
        std::thread::spawn(move || {
            runtime.block_on(WorkerHostRuntimeService { state }.push_scope(request))
        })
    };
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while Arc::strong_count(&stack) == baseline && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert!(
        Arc::strong_count(&stack) > baseline,
        "push must be admitted before shutdown closes the binding"
    );
    state.close_scope_admission().unwrap();
    runtime.block_on(async {
        tokio::time::pause();
        let mut draining = Box::pin(state.drain_owned_scopes());
        assert!(futures_util::poll!(draining.as_mut()).is_pending());
        tokio::time::advance(WORKER_RPC_TIMEOUT + Duration::from_millis(1)).await;
        let error = draining
            .await
            .expect_err("blocked native work must not permit a successful drain");
        tokio::time::resume();
        assert!(
            error.to_string().contains("scope cleanup timed out"),
            "{error}"
        );
        assert!(
            *state.scope_activity.borrow() > 0,
            "timed-out cleanup must retain ownership of admitted work"
        );
        assert!(
            service
                .create_scope_stack(Request::new(CreateScopeStackRequest {
                    credentials: credentials(),
                }))
                .await
                .is_err()
        );
        assert!(
            service
                .push_scope(push_request(&stack_id, "", "late-shutdown-push"))
                .await
                .is_err()
        );
    });
    drop(held);
    assert!(pushing.join().unwrap().is_err());
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(3), state.drain_owned_scopes())
            .await
            .unwrap()
            .unwrap();
    });
    assert_eq!(*state.scope_activity.borrow(), 0);
    assert!(state.scope_stacks.lock().unwrap().is_empty());
    assert!(state.scope_handles.lock().unwrap().is_empty());
    assert_eq!(stack.read().unwrap().scopes().len(), 1);
}

#[test]
fn scope_drain_bounds_native_unwind_even_without_an_admitted_push() {
    let runtime = RuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let state = Arc::new(WorkerHostRuntimeState::new(
        ACTIVATION_ID.into(),
        AUTH_TOKEN.into(),
    ));
    let service = WorkerHostRuntimeService {
        state: state.clone(),
    };
    let stack_id = runtime.block_on(create_stack(&service));
    runtime
        .block_on(service.push_scope(push_request(&stack_id, "", "blocked-native-unwind")))
        .unwrap();
    assert_eq!(
        *state.scope_activity.borrow(),
        0,
        "the push already completed"
    );
    let stack = state.stack(&stack_id).unwrap().unwrap();
    let held = stack.write().unwrap();
    state.close_scope_admission().unwrap();
    runtime.block_on(async {
        tokio::time::pause();
        let mut draining = Box::pin(state.drain_owned_scopes());
        assert!(futures_util::poll!(draining.as_mut()).is_pending());
        tokio::time::advance(WORKER_RPC_TIMEOUT + Duration::from_millis(1)).await;
        let error = draining
            .await
            .expect_err("a blocked native unwind must be bounded");
        tokio::time::resume();
        assert!(
            error.to_string().contains("scope cleanup timed out"),
            "{error}"
        );
        assert!(
            *state.scope_activity.borrow() > 0,
            "timed-out native unwind must retain tracked ownership"
        );
    });
    drop(held);
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(3), state.drain_owned_scopes())
            .await
            .unwrap()
            .unwrap();
    });
    assert_eq!(*state.scope_activity.borrow(), 0);
    assert!(state.scope_stacks.lock().unwrap().is_empty());
    assert!(state.scope_handles.lock().unwrap().is_empty());
    assert_eq!(stack.read().unwrap().scopes().len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn poisoned_scope_ownership_makes_shutdown_unsafe() {
    let (instance, _worker_shutdown) = fake_worker_instance(Vec::new()).await;
    let service = WorkerHostRuntimeService {
        state: instance.host_state.clone(),
    };
    let stack_id = create_stack(&service).await;
    let pushed = service
        .push_scope(push_request(&stack_id, "", "poisoned-shutdown-scope"))
        .await
        .unwrap()
        .into_inner()
        .scope_handle_id;
    let stack = instance.host_state.stack(&stack_id).unwrap().unwrap();
    let scope_uuid = instance.host_state.scope_handles.lock().unwrap()[&pushed]
        .handle
        .uuid;
    let poison = instance.host_state.clone();
    assert!(
        std::thread::spawn(move || {
            let _owned = poison.scope_handles.lock().unwrap();
            panic!("poison scope handle ownership for teardown regression");
        })
        .join()
        .is_err()
    );
    assert!(instance.host_state.scope_handles.is_poisoned());
    let outcome = instance.shutdown_checked();
    assert!(
        !outcome.safe_to_unload,
        "poisoned ownership cannot acknowledge safe cleanup: {outcome:?}"
    );
    assert!(
        outcome.errors.iter().any(|error| error.contains("poison")),
        "cleanup must report the ownership failure: {outcome:?}"
    );
    let repeated = instance.shutdown_checked();
    assert!(
        !repeated.safe_to_unload,
        "repeated teardown must not erase an ownership failure: {repeated:?}"
    );
    // Explicitly unwind any native scope left by the failed cleanup so the
    // test does not retain live native state after proving unsafe ownership.
    let still_present = stack
        .read()
        .unwrap()
        .scopes()
        .iter()
        .any(|scope| scope.uuid == scope_uuid);
    if still_present {
        nemo_relay_runtime::api::runtime::with_scope_stack(stack, || {
            pop_scope(PopScopeParams::builder().handle_uuid(&scope_uuid).build())
        })
        .unwrap();
    }
}

#[test]
fn repeated_scope_drains_track_unbound_cleanup_after_the_handle_map_is_empty() {
    let runtime = RuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let state = Arc::new(WorkerHostRuntimeState::new(
        ACTIVATION_ID.into(),
        AUTH_TOKEN.into(),
    ));
    let service = WorkerHostRuntimeService {
        state: state.clone(),
    };
    let stack = nemo_relay_runtime::api::runtime::create_scope_stack();
    let pushed = nemo_relay_runtime::api::runtime::with_scope_stack(stack.clone(), || {
        runtime.block_on(service.push_scope(Request::new(PushScopeRequest {
            credentials: credentials(),
            scope: None,
            name: "blocked-unbound-unwind".into(),
            scope_type: ProtoScopeType::Function as i32,
            ..Default::default()
        })))
    })
    .unwrap()
    .into_inner()
    .scope_handle_id;
    {
        let handles = state.scope_handles.lock().unwrap();
        assert!(handles[&pushed].scope_stack_id.is_empty());
        assert!(Arc::ptr_eq(&handles[&pushed].context.scope_stack, &stack));
    }
    assert!(state.scope_stacks.lock().unwrap().is_empty());
    let held = stack.write().unwrap();
    state.close_scope_admission().unwrap();
    runtime.block_on(async {
        tokio::time::pause();
        let mut first = Box::pin(state.drain_owned_scopes());
        assert!(futures_util::poll!(first.as_mut()).is_pending());
        // Wait until the blocking job has taken ownership out of the map and
        // is blocked by our native lock. An empty registry is not completion.
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !state.scope_handles.lock().unwrap().is_empty() {
            assert!(
                std::time::Instant::now() < deadline,
                "cleanup must acquire the unbound handle"
            );
            tokio::task::yield_now().await;
        }
        tokio::time::advance(WORKER_RPC_TIMEOUT + Duration::from_millis(1)).await;
        let error = first
            .await
            .expect_err("blocked unbound cleanup must time out");
        assert!(
            error.to_string().contains("scope cleanup timed out"),
            "{error}"
        );
        assert!(*state.scope_activity.borrow() > 0);
        let mut second = Box::pin(state.drain_owned_scopes());
        assert!(
            futures_util::poll!(second.as_mut()).is_pending(),
            "retry must wait for cleanup that already owns the removed handle"
        );
        tokio::time::advance(WORKER_RPC_TIMEOUT + Duration::from_millis(1)).await;
        let error = second
            .await
            .expect_err("retry cannot report success while native unwind is blocked");
        tokio::time::resume();
        assert!(
            error.to_string().contains("scope cleanup timed out"),
            "{error}"
        );
        assert!(state.scope_handles.lock().unwrap().is_empty());
        assert!(*state.scope_activity.borrow() > 0);
    });
    drop(held);
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(3), state.drain_owned_scopes())
            .await
            .unwrap()
            .unwrap();
    });
    assert_eq!(*state.scope_activity.borrow(), 0);
    assert!(state.scope_handles.lock().unwrap().is_empty());
    assert!(state.scope_stacks.lock().unwrap().is_empty());
    assert_eq!(stack.read().unwrap().scopes().len(), 1);
}
