// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use napi::bindgen_prelude::{JsError, ToNapiValue};
use napi::{Env, Status, check_status};
use nemo_relay::diagnostics::embedded::EmbeddedDiagnosticError;
use nemo_relay::error::FlowError;
use nemo_relay::observability::atof::AtofExporterError;
use nemo_relay::observability::otel::OpenTelemetryError;
use nemo_relay::observability::otel_file::OtlpFileExporterError;
use nemo_relay::plugin::PluginError;
use nemo_relay_adaptive::AdaptiveError;
use nemo_relay_adaptive::acg::AcgError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Napi(#[from] napi::Error),
    #[error(transparent)]
    Flow(#[from] FlowError),
    #[error(transparent)]
    Plugin(#[from] PluginError),
    #[error(transparent)]
    Adaptive(#[from] AdaptiveError),
    #[error(transparent)]
    Acg(#[from] AcgError),
    #[error(transparent)]
    OpenTelemetry(#[from] OpenTelemetryError),
    #[error(transparent)]
    Atof(#[from] AtofExporterError),
    #[error(transparent)]
    OtlpFile(#[from] OtlpFileExporterError),
    #[error("{}", .0.reason())]
    Diagnostic(EmbeddedDiagnosticError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn invalid_argument(reason: impl Into<String>) -> Self {
        Self::Napi(napi::Error::new(Status::InvalidArg, reason.into()))
    }

    pub(crate) fn generic_failure(reason: impl Into<String>) -> Self {
        Self::Napi(napi::Error::from_reason(reason.into()))
    }

    pub(crate) fn to_napi(&self) -> napi::Error {
        match self {
            Self::Napi(error) => error.try_clone().unwrap_or_else(|failure| failure),
            Self::Flow(FlowError::CallbackException {
                source: Some(source),
                ..
            }) => match source.downcast_ref::<napi::Error>() {
                Some(exception) => exception.try_clone().unwrap_or_else(|failure| failure),
                None => chained(self),
            },
            other => chained(other),
        }
    }

    pub(crate) fn throw_into(self, env: &Env) -> napi::Error {
        throw(env, self.to_napi())
    }
}

fn chained(error: &dyn std::error::Error) -> napi::Error {
    let mut converted = napi::Error::new(Status::GenericFailure, error.to_string());
    converted.cause = error.source().map(|source| Box::new(chained(source)));
    converted
}

pub(crate) fn throw(env: &Env, error: napi::Error) -> napi::Error {
    let mut pending = false;
    if let Err(failure) =
        check_status!(unsafe { napi::sys::napi_is_exception_pending(env.raw(), &mut pending) })
    {
        return failure;
    }
    if pending && error.status == Status::PendingException {
        return error;
    }
    let reason = error.reason.clone();
    match env.throw(error) {
        Ok(()) => napi::Error::new(Status::PendingException, reason),
        Err(failure) => failure,
    }
}

impl From<EmbeddedDiagnosticError> for Error {
    fn from(error: EmbeddedDiagnosticError) -> Self {
        Self::Diagnostic(error)
    }
}

impl From<Error> for napi::Error {
    fn from(error: Error) -> Self {
        error.to_napi()
    }
}

impl From<Error> for JsError {
    fn from(error: Error) -> Self {
        Self::from(error.to_napi())
    }
}

impl ToNapiValue for Error {
    unsafe fn to_napi_value(
        env: napi::sys::napi_env,
        val: Self,
    ) -> napi::Result<napi::sys::napi_value> {
        unsafe { napi::Error::to_napi_value(env, val.to_napi()) }
    }
}
