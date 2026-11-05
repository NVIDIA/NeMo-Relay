// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use napi::bindgen_prelude::{
    Array, External, FnArgs, FromNapiRef, FromNapiValue, Function, FunctionRef, JsObjectValue,
    JsValue, JsValuesTupleIntoVec, Object, Unknown,
};
use napi::threadsafe_function::ThreadsafeCallContext;
use napi::{Env, ValueType};
use nemo_relay_runtime::api::runtime::subscriber_dispatcher::{
    PublicationBuffer, capture_nested_publication_buffer,
};
use nemo_relay_runtime::api::runtime::{
    PropagationContext, ScopeStackHandle, capture_propagation_context, current_scope_stack,
};

use crate::convert::{HandledTsfn, JsArg, JsArgs2, detach, js_value};
use crate::environment;

const CALLBACK_FACTORIES_SOURCE: &str = include_str!("callback_factories.js");

type PromiseFactoryArgs = FnArgs<(JsArg, JsArg, JsArg, JsArg, JsArg, JsArg)>;

pub(crate) struct Settlers {
    pub(crate) resolve: JsArg,
    pub(crate) reject: JsArg,
    pub(crate) stream_push: JsArg,
    pub(crate) stream_end: JsArg,
    pub(crate) next: JsArg,
}

#[derive(Clone, Copy)]
pub(crate) enum Wrapper {
    Execution,
    Middleware,
    ConditionalGate,
    Collector,
    Finalizer,
    Scoped,
}

macro_rules! factories {
    ($($field:ident: $args:ty => $ret:ty = $name:literal,)*) => {
        pub(crate) struct Factories {
            $(pub(crate) $field: FunctionRef<$args, $ret>,)*
        }

        impl Factories {
            pub(crate) fn new(env: &Env) -> napi::Result<Self> {
                let factories: Object = env.run_script(CALLBACK_FACTORIES_SOURCE)?;
                Ok(Self {
                    $($field: factories
                        .get_named_property::<Function<$args, $ret>>($name)?
                        .create_ref()?,)*
                })
            }
        }
    };
}

factories! {
    execution: JsArg => JsArg = "execution",
    middleware: JsArg => JsArg = "middleware",
    conditional_gate: JsArg => JsArg = "conditionalGate",
    subscriber: JsArgs2 => JsArg = "subscriber",
    collector: JsArg => JsArg = "collector",
    finalizer: JsArg => JsArg = "finalizer",
    promise: PromiseFactoryArgs => JsArg = "promise",
    scoped: JsArg => JsArg = "scoped",
    stream_next: JsArgs2 => JsArg = "streamNext",
    stream_close: JsArgs2 => JsArg = "streamClose",
    pin_while: JsArgs2 => JsArg = "pinWhile",
    aggregate: JsArgs2 => JsArg = "aggregate",
    error_first: JsArg => JsArg = "errorFirst",
    noop: () => () = "noop",
    publication_callback_active: () => bool = "publicationCallbackActive",
    event_sanitizer_callback_context_id: () => Option<String> = "eventSanitizerCallbackContextId",
    callback_scope: () => JsArg = "callbackScope",
    with_callback_scope: JsArgs2 => JsArg = "withCallbackScope",
    set_callback_scope: JsArg => bool = "setCallbackScope",
}

impl Wrapper {
    fn select(self) -> fn(&Factories) -> &FunctionRef<JsArg, JsArg> {
        match self {
            Self::Execution => |factories| &factories.execution,
            Self::Middleware => |factories| &factories.middleware,
            Self::ConditionalGate => |factories| &factories.conditional_gate,
            Self::Collector => |factories| &factories.collector,
            Self::Finalizer => |factories| &factories.finalizer,
            Self::Scoped => |factories| &factories.scoped,
        }
    }
}

fn factory<'env, Args: JsValuesTupleIntoVec, Return>(
    env: &'env Env,
    select: fn(&Factories) -> &FunctionRef<Args, Return>,
) -> napi::Result<Function<'env, Args, Return>> {
    select(&environment::state(env)?.factories).borrow_back(env)
}

fn call<'env, Args: JsValuesTupleIntoVec>(
    env: &'env Env,
    select: fn(&Factories) -> &FunctionRef<Args, JsArg>,
    args: Args,
) -> napi::Result<Unknown<'env>> {
    let value = factory(env, select)?.call(args)?;
    // SAFETY: `value` was returned by the factory call above, in `env`.
    Ok(unsafe { value.into_unknown(env) })
}

#[derive(Clone)]
pub(crate) struct CallbackScope {
    pub(crate) scope_stack: ScopeStackHandle,
    pub(crate) publication_buffer: Option<PublicationBuffer>,
    pub(crate) propagation: Option<PropagationContext>,
}

impl CallbackScope {
    pub(crate) fn capture() -> Self {
        Self {
            scope_stack: current_scope_stack(),
            publication_buffer: capture_nested_publication_buffer(),
            propagation: capture_propagation_context().ok(),
        }
    }

    pub(crate) fn into_js(self, env: &Env) -> napi::Result<JsArg> {
        js_value(env, External::new(self))
    }
}

pub(crate) fn callback_scope_from(
    env: &Env,
    value: Unknown,
) -> napi::Result<Option<CallbackScope>> {
    match value.get_type()? {
        ValueType::Undefined | ValueType::Null => Ok(None),
        ValueType::External => {
            // SAFETY: `value` belongs to the running JavaScript call in `env`.
            let scope =
                unsafe { External::<CallbackScope>::from_napi_ref(env.raw(), value.raw()) }?;
            Ok(Some(CallbackScope::clone(scope)))
        }
        other => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!("callback scope must be an external value, got {other:?}"),
        )),
    }
}

pub(crate) fn wrap<'env, Args: JsValuesTupleIntoVec, Return>(
    env: &'env Env,
    wrapper: Wrapper,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<Function<'env, Args, Return>> {
    Function::from_unknown(call(env, wrapper.select(), detach(env, func))?)
}

fn callee_handled<'env, Args: JsValuesTupleIntoVec, Return>(
    env: &'env Env,
    func: &Function<'_, Args, Return>,
) -> napi::Result<Function<'env, Args, Return>> {
    Function::from_unknown(call(
        env,
        |factories| &factories.error_first,
        detach(env, func),
    )?)
}

pub(crate) fn handled_tsfn<T, Args, Return>(
    env: &Env,
    function: &Function<'_, Args, Return>,
    callback: impl 'static + FnMut(ThreadsafeCallContext<T>) -> napi::Result<Args>,
) -> napi::Result<HandledTsfn<T, Args, Return>>
where
    T: 'static,
    Args: 'static + JsValuesTupleIntoVec,
    Return: 'static + FromNapiValue,
{
    callee_handled(env, function)?
        .build_threadsafe_function::<T>()
        .callee_handled::<true>()
        .weak::<true>()
        .build_callback(callback)
}

pub(crate) fn wrap_subscriber_callback<'env, Args: JsValuesTupleIntoVec, Return>(
    env: &'env Env,
    func: &Function<JsArg, JsArg>,
    complete: &Function<'_, JsArg, ()>,
) -> napi::Result<Function<'env, Args, Return>> {
    Function::from_unknown(call(
        env,
        |factories| &factories.subscriber,
        FnArgs::from((detach(env, func), detach(env, complete))),
    )?)
}

pub(crate) fn wrap_promise_callback<'env, Args: JsValuesTupleIntoVec, Return>(
    env: &'env Env,
    func: &Function<JsArg, JsArg>,
    settlers: Settlers,
) -> napi::Result<Function<'env, Args, Return>> {
    Function::from_unknown(call(
        env,
        |factories| &factories.promise,
        FnArgs::from((
            detach(env, func),
            settlers.resolve,
            settlers.reject,
            settlers.stream_push,
            settlers.stream_end,
            settlers.next,
        )),
    )?)
}

pub(crate) fn stream_step<'env>(
    env: &'env Env,
    close: bool,
    poll: &Function<JsArg, JsArg>,
    handle: &Object,
) -> napi::Result<Unknown<'env>> {
    let helper: fn(&Factories) -> &FunctionRef<JsArgs2, JsArg> = if close {
        |factories| &factories.stream_close
    } else {
        |factories| &factories.stream_next
    };
    call(
        env,
        helper,
        FnArgs::from((detach(env, poll), detach(env, handle))),
    )
}

pub(crate) fn pin_while<'env>(
    env: &'env Env,
    handle: &Object,
    promise: &impl JsValue<'env>,
) -> napi::Result<Unknown<'env>> {
    call(
        env,
        |factories| &factories.pin_while,
        FnArgs::from((detach(env, handle), detach(env, promise))),
    )
}

pub(crate) fn aggregate_error(
    env: &Env,
    errors: Vec<napi::Error>,
    message: String,
) -> napi::Result<JsArg> {
    let members = Array::from_vec(env, errors)?;
    factory(env, |factories| &factories.aggregate)?.call(FnArgs::from((
        detach(env, &members),
        js_value(env, message)?,
    )))
}

pub(crate) fn publication_callback_active(env: &Env) -> napi::Result<bool> {
    factory(env, |factories| &factories.publication_callback_active)?.call(())
}

pub(crate) fn event_sanitizer_callback_context_id(env: &Env) -> napi::Result<Option<String>> {
    factory(env, |factories| {
        &factories.event_sanitizer_callback_context_id
    })?
    .call(())
}

pub(crate) fn callback_scope(env: &Env) -> napi::Result<Option<CallbackScope>> {
    callback_scope_from(env, call(env, |factories| &factories.callback_scope, ())?)
}

pub(crate) fn with_callback_scope<'env>(
    env: &'env Env,
    scope: CallbackScope,
    callback: &Function<JsArg, JsArg>,
) -> napi::Result<Unknown<'env>> {
    call(
        env,
        |factories| &factories.with_callback_scope,
        FnArgs::from((scope.into_js(env)?, detach(env, callback))),
    )
}

pub(crate) fn set_callback_scope(env: &Env, scope: CallbackScope) -> napi::Result<bool> {
    factory(env, |factories| &factories.set_callback_scope)?.call(scope.into_js(env)?)
}
