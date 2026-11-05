// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Service-level scope-parent selection, branch ownership, and retry regressions.

use super::*;

fn credentials() -> Option<pb::Credentials> {
    Some(pb::Credentials {
        activation_id: ACTIVATION_ID.into(),
        auth_token: AUTH_TOKEN.into(),
    })
}

fn service() -> WorkerHostRuntimeService {
    WorkerHostRuntimeService {
        state: Arc::new(WorkerHostRuntimeState::new(
            ACTIVATION_ID.into(),
            AUTH_TOKEN.into(),
        )),
    }
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

fn selected(stack: &str, parent: &str) -> ScopeContext {
    ScopeContext {
        scope_stack_id: stack.into(),
        parent_scope_id: parent.into(),
    }
}

async fn push(service: &WorkerHostRuntimeService, stack: &str, parent: &str, name: &str) -> String {
    service
        .push_scope(Request::new(PushScopeRequest {
            credentials: credentials(),
            scope: Some(selected(stack, parent)),
            name: name.into(),
            scope_type: ProtoScopeType::Function as i32,
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner()
        .scope_handle_id
}

fn pop_request(handle: &str) -> Request<PopScopeRequest> {
    Request::new(PopScopeRequest {
        credentials: credentials(),
        scope_handle_id: handle.into(),
        ..Default::default()
    })
}

async fn ids(service: &WorkerHostRuntimeService, stack: &str, parent: &str) -> Vec<String> {
    service
        .get_current_scope_stack_snapshot(snapshot_request(Some(selected(stack, parent))))
        .await
        .unwrap()
        .into_inner()
        .snapshot()
        .unwrap()
        .scopes
        .into_iter()
        .map(|scope| scope.scope_id)
        .collect()
}

fn uuid(service: &WorkerHostRuntimeService, handle: &str) -> Uuid {
    service.state.scope_handles.lock().unwrap()[handle]
        .handle
        .uuid
}

async fn mark(service: &WorkerHostRuntimeService, stack: &str, parent: &str, name: &str) {
    service
        .emit_mark(Request::new(EmitMarkRequest {
            credentials: credentials(),
            scope: Some(selected(stack, parent)),
            mark: Some(pb::PendingMarkSpec {
                name: name.into(),
                ..Default::default()
            }),
        }))
        .await
        .unwrap();
}

#[tokio::test]
async fn explicit_parent_excludes_descendants_and_their_scoped_subscribers() {
    let service = service();
    let stack = create_stack(&service).await;
    let a = push(&service, &stack, "", "prefix-A").await;
    let b = push(&service, &stack, "", "prefix-B").await;
    let a_uuid = uuid(&service, &a);
    let b_uuid = uuid(&service, &b);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let native_stack = service.state.stack(&stack).unwrap().unwrap();
    nemo_relay_runtime::api::runtime::with_scope_stack(native_stack, || {
        let captured = seen.clone();
        nemo_relay_runtime::api::subscriber::scope_register_subscriber(
            &b_uuid,
            "descendant-only",
            Arc::new(move |event| {
                if event.name().starts_with("prefix-mark-") {
                    captured.lock().unwrap().push(event.name().to_owned());
                }
            }),
        )
        .unwrap();
    });
    let original = ids(&service, &stack, "").await;
    assert_eq!(original.last(), Some(&b_uuid.to_string()));
    let prefix = ids(&service, &stack, &a).await;
    assert_eq!(prefix.last(), Some(&a_uuid.to_string()));
    assert!(!prefix.contains(&b_uuid.to_string()));
    mark(&service, &stack, &a, "prefix-mark-selected").await;
    mark(&service, &stack, "", "prefix-mark-original").await;
    nemo_relay_runtime::api::subscriber::flush_subscribers().unwrap();
    assert_eq!(*seen.lock().unwrap(), ["prefix-mark-original"]);
    assert_eq!(ids(&service, &stack, "").await, original);
    service.pop_scope(pop_request(&b)).await.unwrap();
    service.pop_scope(pop_request(&a)).await.unwrap();
}

#[tokio::test]
async fn explicit_parent_branches_keep_their_context_and_cannot_resurrect_expired_handles() {
    let service = service();
    let stack = create_stack(&service).await;
    let a = push(&service, &stack, "", "branch-A").await;
    let b = push(&service, &stack, "", "branch-B").await;
    let original = ids(&service, &stack, "").await;
    let c = push(&service, &stack, &a, "branch-C").await;
    let c_uuid = uuid(&service, &c).to_string();
    let branch = ids(&service, &stack, &c).await;
    assert_eq!(branch.last(), Some(&c_uuid));
    assert!(!branch.contains(&uuid(&service, &b).to_string()));
    assert_eq!(ids(&service, &stack, "").await, original);
    // D retains C in its private prefix, but that snapshot must not make a
    // revoked public C binding usable again after C has been popped.
    let d = push(&service, &stack, &c, "branch-D").await;
    service.pop_scope(pop_request(&c)).await.unwrap();
    assert!(ids(&service, &stack, &d).await.contains(&c_uuid));
    assert_eq!(
        service
            .get_current_scope_stack_snapshot(snapshot_request(Some(selected(&stack, &c))))
            .await
            .unwrap_err()
            .code(),
        tonic::Code::NotFound
    );
    assert_eq!(ids(&service, &stack, "").await, original);
    service.pop_scope(pop_request(&d)).await.unwrap();
    service.pop_scope(pop_request(&b)).await.unwrap();
    service.pop_scope(pop_request(&a)).await.unwrap();
    assert!(service.state.scope_handles.lock().unwrap().is_empty());
}

#[tokio::test]
async fn invalid_parent_selections_do_not_mutate_scope_state() {
    let service = service();
    let stack = create_stack(&service).await;
    let foreign = create_stack(&service).await;
    let foreign_parent = push(&service, &foreign, "", "foreign").await;
    let expired = push(&service, &stack, "", "expired").await;
    service.pop_scope(pop_request(&expired)).await.unwrap();
    let original = ids(&service, &stack, "").await;
    for (parent, code) in [
        ("scope-unknown", tonic::Code::NotFound),
        (foreign_parent.as_str(), tonic::Code::PermissionDenied),
        (expired.as_str(), tonic::Code::NotFound),
    ] {
        let request = PushScopeRequest {
            credentials: credentials(),
            scope: Some(selected(&stack, parent)),
            name: "must-not-push".into(),
            scope_type: ProtoScopeType::Function as i32,
            ..Default::default()
        };
        assert_eq!(
            service
                .push_scope(Request::new(request))
                .await
                .unwrap_err()
                .code(),
            code
        );
        assert_eq!(
            service
                .get_current_scope_stack_snapshot(snapshot_request(Some(selected(&stack, parent))))
                .await
                .unwrap_err()
                .code(),
            code
        );
        let error = service
            .emit_mark(Request::new(EmitMarkRequest {
                credentials: credentials(),
                scope: Some(selected(&stack, parent)),
                mark: Some(pb::PendingMarkSpec {
                    name: "must-not-emit".into(),
                    ..Default::default()
                }),
            }))
            .await
            .unwrap_err();
        assert_eq!(error.code(), code);
        assert_eq!(ids(&service, &stack, "").await, original);
    }
    service
        .pop_scope(pop_request(&foreign_parent))
        .await
        .unwrap();
    assert!(service.state.scope_handles.lock().unwrap().is_empty());
}

#[tokio::test]
async fn failed_scope_pops_keep_handles_available_for_retry() {
    let service = service();
    let stack = create_stack(&service).await;
    let a = push(&service, &stack, "", "retry-A").await;
    let b = push(&service, &stack, "", "retry-B").await;
    assert_eq!(
        service.pop_scope(pop_request(&a)).await.unwrap_err().code(),
        tonic::Code::InvalidArgument
    );
    assert!(service.state.scope_handles.lock().unwrap().contains_key(&a));
    assert!(
        !service.state.scope_handles.lock().unwrap()[&a]
            .pop_active
            .load(Ordering::Acquire)
    );
    let mut overflow = pop_request(&b).into_inner();
    overflow.output = Some(pb::JsonValue {
        json: b"1e400".to_vec(),
    });
    let error = service
        .pop_scope(Request::new(overflow))
        .await
        .expect_err("unrepresentable JSON numbers must reject the pop");
    assert!(error.message().contains("invalid JSON envelope"));
    assert!(service.state.scope_handles.lock().unwrap().contains_key(&b));
    service.pop_scope(pop_request(&b)).await.unwrap();
    service.pop_scope(pop_request(&a)).await.unwrap();
    assert!(service.state.scope_handles.lock().unwrap().is_empty());
}

#[test]
fn concurrent_scope_pop_reservation_rejects_duplicate_without_consuming_handle() {
    let runtime = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let service = Arc::new(service());
    let stack = runtime.block_on(create_stack(&service));
    let handle = runtime.block_on(push(&service, &stack, "", "concurrent-pop"));
    let native_stack = service.state.stack(&stack).unwrap().unwrap();
    let reserved = service.state.scope_handles.lock().unwrap()[&handle]
        .pop_active
        .clone();
    // Force the first native pop to wait after it reserves the public handle.
    let stack_guard = native_stack.write().unwrap();
    let first = {
        let runtime = runtime.clone();
        let service = service.clone();
        let handle = handle.clone();
        std::thread::spawn(move || runtime.block_on(service.pop_scope(pop_request(&handle))))
    };
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while !reserved.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert!(
        reserved.load(Ordering::Acquire),
        "first pop must reserve its handle"
    );
    assert_eq!(
        runtime
            .block_on(service.pop_scope(pop_request(&handle)))
            .unwrap_err()
            .code(),
        tonic::Code::FailedPrecondition
    );
    assert!(
        service
            .state
            .scope_handles
            .lock()
            .unwrap()
            .contains_key(&handle)
    );
    drop(stack_guard);
    first.join().unwrap().unwrap();
    assert!(
        !service
            .state
            .scope_handles
            .lock()
            .unwrap()
            .contains_key(&handle)
    );
}
