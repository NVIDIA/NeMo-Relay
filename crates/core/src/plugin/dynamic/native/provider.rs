// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private provider calls bound to a live native execution continuation.

use nemo_relay_plugin::LlmProviderRequest;

use super::*;

pub(super) fn owner_is_active(owner: &Option<NativeAsyncNextOwner>) -> bool {
    match owner {
        Some(NativeAsyncNextOwner::Completion(owner)) => owner.upgrade().is_some_and(|owner| {
            !owner.cancelled.load(Ordering::Acquire)
                && owner
                    .sender
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .is_some()
        }),
        Some(NativeAsyncNextOwner::Stream(owner)) => owner.upgrade().is_some_and(|owner| {
            !owner.cancelled.load(Ordering::Acquire) && !owner.settled.load(Ordering::Acquire)
        }),
        None => false,
    }
}

pub(super) unsafe extern "C" fn native_async_next_has_provider(
    next: *const NemoRelayNativeAsyncNext,
) -> bool {
    let Some(next) = (unsafe { (next as *const NativeAsyncNext).as_ref() }) else {
        return false;
    };
    matches!(
        next.inner,
        NativeAsyncNextInner::Llm(_) | NativeAsyncNextInner::LlmStream(_)
    ) && next.provider_dispatcher.is_some()
        && owner_is_active(&next.owner)
}

fn provider_request(
    next: &NativeAsyncNext,
    request_json: *const NemoRelayNativeString,
) -> Result<(LlmProviderDispatcher, LlmProviderRequest), NemoRelayStatus> {
    if matches!(next.inner, NativeAsyncNextInner::Tool(_)) || !owner_is_active(&next.owner) {
        set_native_last_error("provider calls require a live LLM execution continuation");
        return Err(NemoRelayStatus::InvalidArg);
    }
    let Some(dispatcher) = next.provider_dispatcher.clone() else {
        set_native_last_error("private provider dispatch is unavailable for this request");
        return Err(NemoRelayStatus::InvalidArg);
    };
    let value = parse_json_arg(request_json, "provider request")?;
    let request = serde_json::from_value(value).map_err(|_| {
        set_native_last_error("invalid provider request: expected target and content");
        NemoRelayStatus::InvalidJson
    })?;
    Ok((dispatcher, request))
}

pub(super) unsafe extern "C" fn native_async_next_call_provider(
    next: *const NemoRelayNativeAsyncNext,
    request_json: *const NemoRelayNativeString,
    cb: NemoRelayNativeAsyncNextResultCb,
    user_data: *mut c_void,
) -> NemoRelayStatus {
    let Some(next) = (unsafe { (next as *const NativeAsyncNext).as_ref() }) else {
        return NemoRelayStatus::NullPointer;
    };
    let (dispatcher, request) = match provider_request(next, request_json) {
        Ok(input) => input,
        Err(status) => return status,
    };
    spawn_native_unary(
        next,
        Box::pin(async move { (dispatcher.call)(request).await }),
        cb,
        user_data,
    )
}

pub(super) unsafe extern "C" fn native_async_next_stream_provider(
    next: *const NemoRelayNativeAsyncNext,
    request_json: *const NemoRelayNativeString,
    cb: NemoRelayNativeAsyncLlmStreamOpenCb,
    user_data: *mut c_void,
) -> NemoRelayStatus {
    let Some(next) = (unsafe { (next as *const NativeAsyncNext).as_ref() }) else {
        return NemoRelayStatus::NullPointer;
    };
    let (dispatcher, request) = match provider_request(next, request_json) {
        Ok(input) => input,
        Err(status) => return status,
    };
    spawn_native_stream(
        next,
        Box::pin(async move { (dispatcher.stream)(request).await }),
        cb,
        user_data,
        true,
    )
}
