// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use napi::Status;
use napi::bindgen_prelude::{JsError, ToNapiValue};
use nemo_relay_adaptive::AdaptiveError;
use nemo_relay_adaptive::acg::AcgError;
use nemo_relay_diagnostics::embedded::EmbeddedDiagnosticError;
use nemo_relay_observability::atof::AtofExporterError;
use nemo_relay_observability::otel::OpenTelemetryError;
use nemo_relay_observability::otel_file::OtlpFileExporterError;
use nemo_relay_plugin_host::PluginError;
use nemo_relay_types::error::FlowError;

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
    Diagnostic(#[from] EmbeddedDiagnosticError),
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
            Self::Napi(error) => clone_napi(error),
            Self::Flow(FlowError::CallbackException {
                source: Some(source),
                ..
            }) => match source.downcast_ref::<napi::Error>() {
                Some(exception) => clone_napi(exception),
                None => chained(self),
            },
            other => chained(other),
        }
    }

    pub(crate) fn into_napi(self) -> napi::Error {
        match self {
            Self::Napi(error) => error,
            other => other.to_napi(),
        }
    }
}

fn clone_napi(error: &napi::Error) -> napi::Error {
    match error.try_clone() {
        Ok(clone) | Err(clone) => clone,
    }
}

fn chained(error: &dyn std::error::Error) -> napi::Error {
    let mut converted = napi::Error::new(Status::GenericFailure, error.to_string());
    converted.cause = error.source().map(|source| Box::new(chained(source)));
    converted
}

impl From<Error> for JsError {
    fn from(error: Error) -> Self {
        Self::from(error.into_napi())
    }
}

impl ToNapiValue for Error {
    unsafe fn to_napi_value(
        env: napi::sys::napi_env,
        val: Self,
    ) -> napi::Result<napi::sys::napi_value> {
        // SAFETY: the caller upholds the `ToNapiValue` contract, so `env` is the live
        // environment of the calling JavaScript thread.
        unsafe { napi::Error::to_napi_value(env, val.into_napi()) }
    }
}
