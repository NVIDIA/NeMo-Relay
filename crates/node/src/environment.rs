// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! State kept once per Node environment: cached callback factories, one promise settler,
//! and the teardown flag.

use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use napi::bindgen_prelude::{PromiseRaw, ToNapiValue, within_runtime_if_available};
use napi::threadsafe_function::{ThreadsafeCallContext, ThreadsafeFunctionCallMode};
use napi::{Env, check_status};

use crate::callback_factory::Factories;
use crate::convert::{JsArg, Tsfn, abort_tsfn, js_value, record_callback_error};
use crate::error::{Error, Result};

pub(crate) type Settler = Box<dyn FnOnce(&Env) -> napi::Result<JsArg> + Send>;

pub(crate) trait Rejection: Send + 'static {
    fn settler(self) -> Settler;
}

impl Rejection for Error {
    fn settler(self) -> Settler {
        Box::new(move |env| js_value(env, self))
    }
}

impl Rejection for Arc<Error> {
    fn settler(self) -> Settler {
        Box::new(move |env| js_value(env, self.to_napi()))
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Liveness {
    KeepLoopAlive,
    LetLoopExit,
}

pub(crate) struct EnvState {
    env: napi::sys::napi_env,
    teardown: Arc<AtomicBool>,
    pub(crate) factories: Factories,
    settler: Arc<Tsfn<Settlement, ()>>,
    pending: Arc<AtomicUsize>,
}

thread_local! {
    static STATE: RefCell<Option<Rc<EnvState>>> = const { RefCell::new(None) };
}

pub(crate) fn state(env: &Env) -> napi::Result<Rc<EnvState>> {
    if let Some(state) = STATE.with(|slot| slot.borrow().clone()) {
        debug_assert_eq!(
            state.env,
            env.raw(),
            "environment state used by another environment"
        );
        return Ok(state);
    }
    let state = Rc::new(EnvState::new(env)?);
    // Runs before Node closes the settler. The hook is registered after the threadsafe
    // function, and Node runs cleanup hooks last-in first-out.
    env.add_env_cleanup_hook((), |()| {
        if let Some(state) = STATE.with(|slot| slot.borrow_mut().take()) {
            state.teardown.store(true, Ordering::Release);
            if let Err(error) = abort_tsfn(&state.settler) {
                record_callback_error(format!(
                    "nemo_relay: failed to abort the promise settler at environment teardown: {error}"
                ));
            }
        }
    })?;
    STATE.with(|slot| slot.replace(Some(Rc::clone(&state))));
    Ok(state)
}

pub(crate) fn teardown_flag(env: &Env) -> napi::Result<Arc<AtomicBool>> {
    Ok(Arc::clone(&state(env)?.teardown))
}

impl EnvState {
    fn new(env: &Env) -> napi::Result<Self> {
        let factories = Factories::new(env)?;
        let settler = factories
            .noop
            .borrow_back(env)?
            .build_threadsafe_function::<Settlement>()
            .weak::<true>()
            .build_callback(|ctx: ThreadsafeCallContext<Settlement>| ctx.value.deliver(&ctx.env))?;
        Ok(Self {
            env: env.raw(),
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
    outcome: std::result::Result<Settler, Settler>,
    settler: Arc<Tsfn<Settlement, ()>>,
    pending: Arc<AtomicUsize>,
    liveness: Liveness,
}

impl Settlement {
    fn deliver(self, env: &Env) -> napi::Result<()> {
        let Self {
            deferred,
            outcome,
            settler,
            pending,
            liveness,
        } = self;
        let settled = match outcome {
            Ok(resolve) => resolve(env).map(Ok),
            Err(reject) => reject(env).map(Err),
        };
        let status = match settled {
            // SAFETY: the deferred was created in `env` and is settled exactly once here.
            Ok(Ok(value)) => unsafe {
                let value = JsArg::to_napi_value(env.raw(), value)?;
                napi::sys::napi_resolve_deferred(env.raw(), deferred.0, value)
            },
            Ok(Err(value)) => unsafe {
                let value = JsArg::to_napi_value(env.raw(), value)?;
                napi::sys::napi_reject_deferred(env.raw(), deferred.0, value)
            },
            // SAFETY: as above.
            Err(error) => unsafe {
                let error = napi::Error::to_napi_value(env.raw(), error)?;
                napi::sys::napi_reject_deferred(env.raw(), deferred.0, error)
            },
        };
        check_status!(status, "settle promise failed")?;
        if matches!(liveness, Liveness::KeepLoopAlive)
            && pending.fetch_sub(1, Ordering::AcqRel) == 1
        {
            settler_liveness(env, &settler, false)?;
        }
        Ok(())
    }
}

pub(crate) struct PendingPromise {
    deferred: Option<Deferred>,
    settler: Arc<Tsfn<Settlement, ()>>,
    pending: Arc<AtomicUsize>,
    liveness: Liveness,
}

impl PendingPromise {
    pub(crate) fn resolve_with(mut self, settler: Settler) {
        self.send(Ok(settler));
    }

    pub(crate) fn reject_with(mut self, settler: Settler) {
        self.send(Err(settler));
    }

    fn send(&mut self, outcome: std::result::Result<Settler, Settler>) {
        let Some(deferred) = self.deferred.take() else {
            return;
        };
        // A closing settler means the environment is tearing down and the promise is gone.
        let _ = self.settler.call(
            Settlement {
                deferred,
                outcome,
                settler: Arc::clone(&self.settler),
                pending: Arc::clone(&self.pending),
                liveness: self.liveness,
            },
            ThreadsafeFunctionCallMode::NonBlocking,
        );
    }
}

impl Drop for PendingPromise {
    fn drop(&mut self) {
        self.send(Err(Box::new(|env| {
            js_value(
                env,
                Error::generic_failure("promise dropped before it settled"),
            )
        })));
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
    if matches!(liveness, Liveness::KeepLoopAlive)
        && state.pending.fetch_add(1, Ordering::AcqRel) == 0
    {
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
    E: Rejection,
    F: Future<Output = std::result::Result<T, E>> + Send + 'static,
    R: FnOnce(&Env, T) -> napi::Result<JsArg> + Send + 'static,
{
    let (pending, promise) = create_promise(env, Liveness::KeepLoopAlive)?;
    within_runtime_if_available(|| {
        tokio::spawn(async move {
            match future.await {
                Ok(value) => pending.resolve_with(Box::new(move |env| resolve(env, value))),
                Err(error) => pending.reject_with(error.settler()),
            }
        })
    });
    Ok(promise)
}
