// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavior compiled independently against each frozen interface generation.
use abi::{
    Json,
    future::{FfiFuture, FutureExt},
    v1_0::*,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

async fn pending_poll() {
    let mut ready = false;
    std::future::poll_fn(move |cx| {
        if ready {
            std::task::Poll::Ready(())
        } else {
            ready = true;
            let first = cx.waker().clone();
            let second = first.clone();
            second.wake();
            std::task::Poll::Pending
        }
    })
    .await
}
struct Owner {
    active: Arc<AtomicBool>,
    runtime: Arc<RuntimeBox>,
}
impl Component for Owner {
    fn shutdown(&self) -> FfiFuture<Result<()>> {
        let active = self.active.clone();
        let runtime = self.runtime.clone();
        async move {
            pending_poll().await;
            if active.swap(false, Ordering::AcqRel) {
                if let Err(error) = runtime.call_sync("fixture.shutdown".into(), Json::Null) {
                    return Result::Err(error);
                }
            }
            Result::Ok(())
        }
        .into_ffi()
    }
}
struct Cancel(Option<Arc<RuntimeBox>>);
impl Drop for Cancel {
    fn drop(&mut self) {
        if let Some(runtime) = &self.0 {
            runtime
                .call_sync("fixture.cancelled".into(), Json::Null)
                .unwrap();
        }
    }
}
struct Echo {
    active: Arc<AtomicBool>,
    runtime: Arc<RuntimeBox>,
}
impl Callback for Echo {
    fn invoke(
        &self,
        invocation: Invocation,
        next: Option<ContinuationBox>,
    ) -> FfiFuture<Result<Json>> {
        // Capture before constructing the future to cover an unpolled drop.
        let mut cancel = Cancel(Some(self.runtime.clone()));
        let active = self.active.clone();
        async move {
            pending_poll().await;
            if !active.load(Ordering::Acquire) {
                return Result::Err(Error::new("revoked", "closed"));
            }
            assert_eq!(
                invocation.context.scope.snapshot().unwrap(),
                Option::Some(Json::String("scope".into()))
            );
            {
                let mut lease = invocation.context.scope.enter().unwrap();
                lease.restore().unwrap();
            }
            let result = if let Some(next) = next {
                next.call(invocation.value).await
            } else {
                Result::Ok(invocation.value)
            };
            cancel.0.take();
            result
        }
        .into_ffi()
    }
    fn invoke_tool(
        &self,
        invocation: Invocation,
        next: ContinuationBox,
    ) -> FfiFuture<Result<ToolExecutionOutcome>> {
        assert_eq!(invocation.tool_call_id.unwrap().as_str(), "call-id");
        async move {
            pending_poll().await;
            next.call_tool(invocation.value)
                .await
                .map(|result| ToolExecutionOutcome {
                    result,
                    pending_marks: Vec::new(),
                })
        }
        .into_ffi()
    }
    fn invoke_sync(&self, invocation: Invocation) -> Result<Json> {
        Result::Ok(invocation.value)
    }
    fn open_stream(
        &self,
        invocation: Invocation,
        next: ContinuationBox,
    ) -> FfiFuture<Result<StreamBox>> {
        async move {
            pending_poll().await;
            let stream = match next.open_stream(invocation.value).await {
                Ok(stream) => stream,
                Err(error) => return Result::Err(error),
            };
            Result::Ok(StreamBox::new(Forward {
                stream,
                closed: false,
            }))
        }
        .into_ffi()
    }
}
struct Forward {
    stream: StreamBox,
    closed: bool,
}
impl Stream for Forward {
    fn pull(&mut self) -> FfiFuture<Result<Option<Json>>> {
        let future = self.stream.pull();
        async move {
            pending_poll().await;
            future.await
        }
        .into_ffi()
    }
    fn close(&mut self) -> FfiFuture<Result<()>> {
        if self.closed {
            return async { Result::Ok(()) }.into_ffi();
        }
        self.closed = true;
        let future = self.stream.close();
        async move {
            pending_poll().await;
            future.await
        }
        .into_ffi()
    }
}
pub struct Example;
impl Plugin for Example {
    fn allows_multiple_components(&self) -> bool {
        true
    }
    fn validate(&self, _: Json) -> Vec<ConfigDiagnostic> {
        Vec::new()
    }
    fn register(
        &self,
        config: Json,
        runtime: RuntimeBox,
    ) -> FfiFuture<Result<RegisteredComponent>> {
        async move {
            pending_poll().await;
            assert_eq!(
                runtime
                    .call_sync("fixture.register".into(), config.clone())
                    .unwrap(),
                config
            );
            let runtime = Arc::new(runtime);
            let active = Arc::new(AtomicBool::new(true));
            runtime
                .bind_component(ComponentBox::new(Owner {
                    active: active.clone(),
                    runtime: runtime.clone(),
                }))
                .unwrap();
            let registrations = [
                Surface::Subscriber,
                Surface::LlmExecutionIntercept,
                Surface::ToolExecutionIntercept,
                Surface::LlmStreamExecutionIntercept,
            ]
            .into_iter()
            .map(|surface| Registration {
                name: "fixture".into(),
                surface,
                priority: 3,
                break_chain: false,
                options: Json::Null,
                callback: CallbackBox::new(Echo {
                    active: active.clone(),
                    runtime: runtime.clone(),
                }),
            })
            .collect();
            Result::Ok(RegisteredComponent {
                error: Option::None,
                registrations,
                component: ComponentBox::new(Owner { active, runtime }),
            })
        }
        .into_ffi()
    }
}
