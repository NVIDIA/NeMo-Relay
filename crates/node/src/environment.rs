// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use napi::bindgen_prelude::{
    FnArgs, Function, JsObjectValue, Object, PromiseRaw, ToNapiValue, Unknown,
    within_runtime_if_available,
};
use napi::threadsafe_function::{ThreadsafeCallContext, ThreadsafeFunctionCallMode};
use napi::{Env, check_status};

use crate::callback_factory::Factories;
use crate::convert::{
    JsArg, JsBuilder, Tsfn, abort_tsfn, closure_function, detach, js_value, record_callback_error,
    report_queue_status,
};
use crate::error::{Error, Result};

impl From<Error> for JsBuilder {
    fn from(error: Error) -> Self {
        Box::new(move |env| js_value(env, error))
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Liveness {
    KeepLoopAlive,
    LetLoopExit,
}

impl Liveness {
    const fn keeps_loop_alive(self) -> bool {
        matches!(self, Self::KeepLoopAlive)
    }
}

pub(crate) struct EnvState {
    teardown: Arc<AtomicBool>,
    pub(crate) factories: Factories,
    settler: Arc<Tsfn<Settlement, ()>>,
    pending: Arc<AtomicUsize>,
}

thread_local! {
    static STATES: RefCell<Vec<(usize, Rc<EnvState>)>> = const { RefCell::new(Vec::new()) };
}

fn env_key(env: &Env) -> usize {
    env.raw().addr()
}

pub(crate) fn state(env: &Env) -> napi::Result<Rc<EnvState>> {
    let key = env_key(env);
    let known = STATES.with(|slot| {
        slot.borrow()
            .iter()
            .find(|(known, _)| *known == key)
            .map(|(_, state)| Rc::clone(state))
    });
    if let Some(state) = known {
        return Ok(state);
    }
    let state = Rc::new(EnvState::new(env)?);
    env.add_env_cleanup_hook(key, |key| {
        let removed = STATES.with(|slot| {
            let mut states = slot.borrow_mut();
            states
                .iter()
                .position(|(known, _)| *known == key)
                .map(|index| states.swap_remove(index).1)
        });
        if let Some(state) = removed {
            state.teardown.store(true, Ordering::Release);
            if let Err(error) = abort_tsfn(&state.settler) {
                record_callback_error(format!(
                    "nemo_relay: failed to abort the promise settler at environment teardown: {error}"
                ));
            }
        }
    })?;
    STATES.with(|slot| slot.borrow_mut().push((key, Rc::clone(&state))));
    Ok(state)
}

pub(crate) fn teardown_flag(env: &Env) -> napi::Result<Arc<AtomicBool>> {
    Ok(Arc::clone(&state(env)?.teardown))
}

pub(crate) const ENV_SHUTTING_DOWN: &str = "Node environment is shutting down";

pub(crate) fn mark_exiting_on_process_exit(env: &Env) -> napi::Result<()> {
    let flag = teardown_flag(env)?;
    let mark = closure_function(env, "__nemo_relay_mark_env_exiting", move |_ctx| {
        flag.store(true, Ordering::Release);
        Ok(())
    })?;
    let process: Object = env.get_global()?.get_named_property("process")?;
    let once: Function<FnArgs<(String, JsArg)>, Unknown> = process.get_named_property("once")?;
    once.apply(
        process,
        FnArgs::from(("exit".to_owned(), detach(env, &mark))),
    )?;
    Ok(())
}
impl EnvState {
    fn new(env: &Env) -> napi::Result<Self> {
        let factories = Factories::new(env)?;
        let settler = factories
            .noop
            .borrow_back(env)?
            .build_threadsafe_function::<Settlement>()
            .weak::<true>()
            .build_callback(|ctx: ThreadsafeCallContext<Settlement>| {
                ctx.value.deliver(&ctx.env);
                Ok(())
            })?;
        Ok(Self {
            teardown: Arc::new(AtomicBool::new(false)),
            factories,
            settler: Arc::new(settler),
            pending: Arc::new(AtomicUsize::new(0)),
        })
    }
}

fn settler_liveness(
    env: &Env,
    settler: &Tsfn<Settlement, ()>,
    keep_alive: bool,
) -> napi::Result<()> {
    settler.handle.with_read_aborted(|aborted| {
        if aborted {
            return Ok(());
        }
        let raw = settler.handle.get_raw();
        // SAFETY: the handle is live until its release, which the read lock excludes.
        check_status!(unsafe {
            if keep_alive {
                napi::sys::napi_ref_threadsafe_function(env.raw(), raw)
            } else {
                napi::sys::napi_unref_threadsafe_function(env.raw(), raw)
            }
        })
    })
}

struct Deferred(napi::sys::napi_deferred);

// SAFETY: Node hands a deferred out as an opaque token. It is settled once, on the
// JavaScript thread that created it, inside the settler callback.
unsafe impl Send for Deferred {}

struct Settlement {
    deferred: Deferred,
    outcome: std::result::Result<JsBuilder, JsBuilder>,
    settler: Arc<Tsfn<Settlement, ()>>,
    pending: Arc<AtomicUsize>,
    liveness: Liveness,
}

impl Settlement {
    fn deliver(self, env: &Env) {
        let Self {
            deferred,
            outcome,
            settler,
            pending,
            liveness,
        } = self;
        if let Err(error) = settle_deferred(env, deferred, outcome) {
            record_callback_error(format!("nemo_relay: failed to settle a promise: {error}"));
        }
        if liveness.keeps_loop_alive()
            && pending.fetch_sub(1, Ordering::AcqRel) == 1
            && let Err(error) = settler_liveness(env, &settler, false)
        {
            record_callback_error(format!(
                "nemo_relay: failed to release the promise settler: {error}"
            ));
        }
    }
}

fn settle_deferred(
    env: &Env,
    deferred: Deferred,
    outcome: std::result::Result<JsBuilder, JsBuilder>,
) -> napi::Result<()> {
    let (resolve, built) = match outcome {
        Ok(resolve) => (true, resolve(env)),
        Err(reject) => (false, reject(env)),
    };
    // SAFETY: `env` is the live environment of the calling JavaScript thread.
    let (resolve, value) = unsafe {
        match built {
            Ok(value) => (resolve, JsArg::to_napi_value(env.raw(), value)?),
            Err(error) => (false, napi::Error::to_napi_value(env.raw(), error)?),
        }
    };
    // SAFETY: the deferred was created in `env` and is settled exactly once here.
    let status = unsafe {
        if resolve {
            napi::sys::napi_resolve_deferred(env.raw(), deferred.0, value)
        } else {
            napi::sys::napi_reject_deferred(env.raw(), deferred.0, value)
        }
    };
    check_status!(status, "settle promise failed")
}

pub(crate) struct PendingPromise {
    deferred: Option<Deferred>,
    settler: Arc<Tsfn<Settlement, ()>>,
    pending: Arc<AtomicUsize>,
    liveness: Liveness,
}

impl PendingPromise {
    pub(crate) fn resolve_with(mut self, build: JsBuilder) {
        self.send(Ok(build));
    }

    pub(crate) fn reject_with(mut self, build: JsBuilder) {
        self.send(Err(build));
    }

    fn send(&mut self, outcome: std::result::Result<JsBuilder, JsBuilder>) {
        let Some(deferred) = self.deferred.take() else {
            return;
        };
        let status = self.settler.call(
            Settlement {
                deferred,
                outcome,
                settler: Arc::clone(&self.settler),
                pending: Arc::clone(&self.pending),
                liveness: self.liveness,
            },
            ThreadsafeFunctionCallMode::NonBlocking,
        );
        report_queue_status(status, "a promise settlement");
    }
}

impl Drop for PendingPromise {
    fn drop(&mut self) {
        self.send(Err(Error::generic_failure(
            "promise dropped before it settled",
        )
        .into()));
    }
}

pub(crate) fn create_promise<'env, V>(
    env: &'env Env,
    liveness: Liveness,
) -> Result<(PendingPromise, PromiseRaw<'env, V>)> {
    let state = state(env)?;
    let mut deferred = std::ptr::null_mut();
    let mut promise = std::ptr::null_mut();
    // SAFETY: `env` is the live environment of the calling JavaScript thread.
    check_status!(unsafe {
        napi::sys::napi_create_promise(env.raw(), &mut deferred, &mut promise)
    })?;
    let pending = PendingPromise {
        deferred: Some(Deferred(deferred)),
        settler: Arc::clone(&state.settler),
        pending: Arc::clone(&state.pending),
        liveness,
    };
    if liveness.keeps_loop_alive() && state.pending.fetch_add(1, Ordering::AcqRel) == 0 {
        settler_liveness(env, &state.settler, true)?;
    }
    Ok((pending, PromiseRaw::new(env.raw(), promise)))
}

pub(crate) fn spawn_promise<'env, V, T, E, F, R>(
    env: &'env Env,
    future: F,
    resolve: R,
) -> Result<PromiseRaw<'env, V>>
where
    T: Send + 'static,
    E: Into<JsBuilder> + Send + 'static,
    F: Future<Output = std::result::Result<T, E>> + Send + 'static,
    R: FnOnce(&Env, T) -> napi::Result<JsArg> + Send + 'static,
{
    let (pending, promise) = create_promise(env, Liveness::KeepLoopAlive)?;
    within_runtime_if_available(|| {
        tokio::spawn(async move {
            match future.await {
                Ok(value) => pending.resolve_with(Box::new(move |env| resolve(env, value))),
                Err(error) => pending.reject_with(error.into()),
            }
        })
    });
    Ok(promise)
}
