// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::registry::{
    RuntimeRegistrationKind, deregister_conditional_middleware_guardrail,
    list_runtime_registrations, register_conditional_middleware_guardrail,
};
use std::collections::BTreeSet;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// Pause a configured OpenTelemetry subscriber during application work.
pub async fn run_example<F, ApplicationFuture>(
    run_application: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnOnce() -> ApplicationFuture,
    ApplicationFuture: std::future::Future<Output = Result<(), Box<dyn std::error::Error>>>,
{
    let kinds = BTreeSet::from([RuntimeRegistrationKind::Subscriber]);
    let target = list_runtime_registrations(Some(&kinds))?
        .into_iter()
        .find(|registration| {
            registration.local_name == "opentelemetry"
                && registration.owner.plugin_kind.as_deref() == Some("observability")
        })
        .ok_or("OpenTelemetry subscriber was not found")?;

    let paused = Arc::new(AtomicBool::new(true));
    let callback_state = Arc::clone(&paused);
    register_conditional_middleware_guardrail(
        "pause-opentelemetry",
        kinds,
        &target.effective_name,
        Arc::new(move |_kinds, _name| {
            callback_state
                .load(Ordering::Acquire)
                .then(|| "export paused".to_string())
        }),
    )?;

    let application_result = run_application().await;
    deregister_conditional_middleware_guardrail("pause-opentelemetry")?;
    application_result?;
    Ok(())
}
