// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JavaScript callback wrapper factories, cached once per environment.

use napi::bindgen_prelude::{
    External, FnArgs, FromNapiRef, FromNapiValue, Function, FunctionRef, JsObjectValue, JsValue,
    JsValuesTupleIntoVec, Object, Unknown,
};
use napi::{Env, ValueType};
use nemo_relay::api::runtime::subscriber_dispatcher::{
    PublicationBuffer, capture_nested_publication_buffer,
};
use nemo_relay::api::runtime::{
    PropagationContext, ScopeStackHandle, capture_propagation_context, current_scope_stack,
};

use crate::convert::{JsArg, JsArgs2, detach, js_value};
use crate::environment;

const CALLBACK_FACTORIES_SOURCE: &str = include_str!("callback_factories.js");

type PromiseFactoryArgs = FnArgs<(JsArg, JsArg, JsArg, JsArg, JsArg, JsArg)>;

/// Native functions a promise wrapper calls: `(resolve, reject, streamPush, streamEnd, next)`.
pub(crate) type PromiseSettlers = FnArgs<(JsArg, JsArg, JsArg, JsArg, JsArg)>;

pub(crate) struct Factories {
    execution: FunctionRef<JsArg, JsArg>,
    middleware: FunctionRef<JsArg, JsArg>,
    conditional_gate: FunctionRef<JsArg, JsArg>,
    subscriber: FunctionRef<JsArgs2, JsArg>,
    collector: FunctionRef<JsArg, JsArg>,
    finalizer: FunctionRef<JsArg, JsArg>,
    promise: FunctionRef<PromiseFactoryArgs, JsArg>,
    scoped: FunctionRef<JsArg, JsArg>,
    stream_next: FunctionRef<JsArgs2, JsArg>,
    stream_close: FunctionRef<JsArgs2, JsArg>,
    pin_while: FunctionRef<JsArgs2, JsArg>,
    aggregate: FunctionRef<JsArgs2, JsArg>,
    pub(crate) noop: FunctionRef<(), ()>,
    publication_callback_active: FunctionRef<(), bool>,
    event_sanitizer_callback_context_id: FunctionRef<(), Option<String>>,
    callback_scope: FunctionRef<(), JsArg>,
    with_callback_scope: FunctionRef<JsArgs2, JsArg>,
    set_callback_scope: FunctionRef<JsArg, bool>,
}

impl Factories {
    pub(crate) fn new(env: &Env) -> napi::Result<Self> {
        let factories: Object = env.run_script(CALLBACK_FACTORIES_SOURCE)?;
        fn reference<Args: JsValuesTupleIntoVec, Return>(
            factories: &Object,
            name: &str,
        ) -> napi::Result<FunctionRef<Args, Return>> {
            factories
                .get_named_property::<Function<Args, Return>>(name)?
                .create_ref()
        }
        Ok(Self {
            execution: reference(&factories, "execution")?,
            middleware: reference(&factories, "middleware")?,
            conditional_gate: reference(&factories, "conditionalGate")?,
            subscriber: reference(&factories, "subscriber")?,
            collector: reference(&factories, "collector")?,
            finalizer: reference(&factories, "finalizer")?,
            promise: reference(&factories, "promise")?,
            scoped: reference(&factories, "scoped")?,
            stream_next: reference(&factories, "streamNext")?,
            stream_close: reference(&factories, "streamClose")?,
            pin_while: reference(&factories, "pinWhile")?,
            aggregate: reference(&factories, "aggregate")?,
            noop: reference(&factories, "noop")?,
            publication_callback_active: reference(&factories, "publicationCallbackActive")?,
            event_sanitizer_callback_context_id: reference(
                &factories,
                "eventSanitizerCallbackContextId",
            )?,
            callback_scope: reference(&factories, "callbackScope")?,
            with_callback_scope: reference(&factories, "withCallbackScope")?,
            set_callback_scope: reference(&factories, "setCallbackScope")?,
        })
    }
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

fn wrap<'env, Args: JsValuesTupleIntoVec, Return>(
    env: &'env Env,
    factory: &FunctionRef<JsArg, JsArg>,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<Function<'env, Args, Return>> {
    let wrapper = factory.borrow_back(env)?.call(detach(env, func))?;
    // SAFETY: `wrapper` was returned by the factory call above, in `env`.
    Function::from_unknown(unsafe { wrapper.into_unknown(env) })
}

/// Wrap an execution callback so a failure crosses N-API as an envelope.
///
/// JavaScript validates the return value as JSON, so a BigInt or a non-finite number
/// is reported, not coerced.
pub(crate) fn wrap_execution_callback<'env, Args: JsValuesTupleIntoVec, Return>(
    env: &'env Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<Function<'env, Args, Return>> {
    wrap(env, &environment::state(env)?.factories.execution, func)
}

/// Wrap a middleware callback so a thrown exception crosses N-API as an envelope.
pub(crate) fn wrap_middleware_callback<'env, Args: JsValuesTupleIntoVec, Return>(
    env: &'env Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<Function<'env, Args, Return>> {
    wrap(env, &environment::state(env)?.factories.middleware, func)
}

/// Wrap a conditional registration gate so a throw or a value other than a string or
/// null crosses N-API as an envelope.
pub(crate) fn wrap_conditional_gate_callback<'env, Args: JsValuesTupleIntoVec, Return>(
    env: &'env Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<Function<'env, Args, Return>> {
    wrap(
        env,
        &environment::state(env)?.factories.conditional_gate,
        func,
    )
}

pub(crate) fn wrap_subscriber_callback<'env, Args: JsValuesTupleIntoVec, Return>(
    env: &'env Env,
    func: &Function<JsArg, JsArg>,
    complete: &Function<'_, JsArg, ()>,
) -> napi::Result<Function<'env, Args, Return>> {
    let wrapper = environment::state(env)?
        .factories
        .subscriber
        .borrow_back(env)?
        .call(FnArgs::from((detach(env, func), detach(env, complete))))?;
    // SAFETY: `wrapper` was returned by the factory call above, in `env`.
    Function::from_unknown(unsafe { wrapper.into_unknown(env) })
}

pub(crate) fn wrap_promise_callback<'env, Args: JsValuesTupleIntoVec, Return>(
    env: &'env Env,
    func: &Function<JsArg, JsArg>,
    settlers: PromiseSettlers,
) -> napi::Result<Function<'env, Args, Return>> {
    let (resolve, reject, stream_push, stream_end, next) = settlers.data;
    let wrapper = environment::state(env)?
        .factories
        .promise
        .borrow_back(env)?
        .call(FnArgs::from((
            detach(env, func),
            resolve,
            reject,
            stream_push,
            stream_end,
            next,
        )))?;
    // SAFETY: `wrapper` was returned by the factory call above, in `env`.
    Function::from_unknown(unsafe { wrapper.into_unknown(env) })
}

pub(crate) fn wrap_collector_callback<'env, Args: JsValuesTupleIntoVec, Return>(
    env: &'env Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<Function<'env, Args, Return>> {
    wrap(env, &environment::state(env)?.factories.collector, func)
}

pub(crate) fn wrap_finalizer_callback<'env, Args: JsValuesTupleIntoVec, Return>(
    env: &'env Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<Function<'env, Args, Return>> {
    wrap(env, &environment::state(env)?.factories.finalizer, func)
}

pub(crate) fn wrap_scoped_callback<'env, Args: JsValuesTupleIntoVec, Return>(
    env: &'env Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<Function<'env, Args, Return>> {
    wrap(env, &environment::state(env)?.factories.scoped, func)
}

fn drive_stream<'env>(
    env: &'env Env,
    helper: &FunctionRef<JsArgs2, JsArg>,
    poll: &Function<JsArg, JsArg>,
    handle: &Object,
) -> napi::Result<Unknown<'env>> {
    let promise = helper
        .borrow_back(env)?
        .call(FnArgs::from((detach(env, poll), detach(env, handle))))?;
    // SAFETY: `promise` was returned by the helper call above, in `env`.
    Ok(unsafe { promise.into_unknown(env) })
}

pub(crate) fn stream_next<'env>(
    env: &'env Env,
    poll: &Function<JsArg, JsArg>,
    handle: &Object,
) -> napi::Result<Unknown<'env>> {
    drive_stream(
        env,
        &environment::state(env)?.factories.stream_next,
        poll,
        handle,
    )
}

pub(crate) fn stream_close<'env>(
    env: &'env Env,
    poll: &Function<JsArg, JsArg>,
    handle: &Object,
) -> napi::Result<Unknown<'env>> {
    drive_stream(
        env,
        &environment::state(env)?.factories.stream_close,
        poll,
        handle,
    )
}

pub(crate) fn pin_while<'env>(
    env: &'env Env,
    handle: &Object,
    promise: &impl JsValue<'env>,
) -> napi::Result<Unknown<'env>> {
    let pinned = environment::state(env)?
        .factories
        .pin_while
        .borrow_back(env)?
        .call(FnArgs::from((detach(env, handle), detach(env, promise))))?;
    // SAFETY: `pinned` was returned by the call above, in `env`.
    Ok(unsafe { pinned.into_unknown(env) })
}

pub(crate) fn aggregate_error(
    env: &Env,
    errors: Vec<napi::Error>,
    message: String,
) -> napi::Result<JsArg> {
    let mut members = env.create_array(errors.len() as u32)?;
    for (index, error) in errors.into_iter().enumerate() {
        members.set(index as u32, error)?;
    }
    environment::state(env)?
        .factories
        .aggregate
        .borrow_back(env)?
        .call(FnArgs::from((
            detach(env, &members),
            js_value(env, message)?,
        )))
}

pub(crate) fn publication_callback_active(env: &Env) -> napi::Result<bool> {
    environment::state(env)?
        .factories
        .publication_callback_active
        .borrow_back(env)?
        .call(())
}

pub(crate) fn event_sanitizer_callback_context_id(env: &Env) -> napi::Result<Option<String>> {
    environment::state(env)?
        .factories
        .event_sanitizer_callback_context_id
        .borrow_back(env)?
        .call(())
}

pub(crate) fn callback_scope(env: &Env) -> napi::Result<Option<CallbackScope>> {
    let scope = environment::state(env)?
        .factories
        .callback_scope
        .borrow_back(env)?
        .call(())?;
    // SAFETY: `scope` was returned by the call above, in `env`.
    callback_scope_from(env, unsafe { scope.into_unknown(env) })
}

pub(crate) fn with_callback_scope<'env>(
    env: &'env Env,
    scope: CallbackScope,
    callback: &Function<JsArg, JsArg>,
) -> napi::Result<Unknown<'env>> {
    let value = environment::state(env)?
        .factories
        .with_callback_scope
        .borrow_back(env)?
        .call(FnArgs::from((scope.into_js(env)?, detach(env, callback))))?;
    // SAFETY: `value` was returned by the call above, in `env`.
    Ok(unsafe { value.into_unknown(env) })
}

pub(crate) fn set_callback_scope(env: &Env, scope: CallbackScope) -> napi::Result<bool> {
    environment::state(env)?
        .factories
        .set_callback_scope
        .borrow_back(env)?
        .call(scope.into_js(env)?)
}
