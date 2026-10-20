// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure classification for the integrated plugin-operation producer.
//!
//! This adapter never records, formats source errors, infers causes from messages,
//! alters execution behavior, or assigns application health. Producers call it
//! inside lazy recording factories so disabled diagnostics does no extra work.

use nemo_relay_types::diagnostics::{
    Classification, CommonPluginFailureCode, DiagnosticClassifier, FailureClass, FailureCode,
    PluginFailureCode,
};

use crate::plugin::PluginError;

fn failure(code: FailureCode, class: FailureClass) -> Classification {
    // Every pair below is an explicit entry in the shared catalogue. This is a
    // programmer invariant, not validation of untrusted producer input.
    Classification::failure(code, class).expect("classifier uses a supported catalogue pair")
}

fn common_plugin(code: CommonPluginFailureCode, class: FailureClass) -> Classification {
    failure(FailureCode::Plugin(PluginFailureCode::Common(code)), class)
}

/// Plugin-operation mapping before errors are flattened into reports or SDK strings.
#[derive(Debug, Clone, Copy, Default)]
pub struct PluginErrorClassifier;

impl DiagnosticClassifier<PluginError> for PluginErrorClassifier {
    fn classify(&self, error: &PluginError) -> Classification {
        let (code, class) = match error {
            // Legacy producers flatten file-read and validation failures into this
            // variant. Its message cannot establish a more specific cause.
            PluginError::InvalidConfig(_) => (
                CommonPluginFailureCode::ConfigurationInvalid,
                FailureClass::Unknown,
            ),
            PluginError::Conflict(_) => (
                CommonPluginFailureCode::StateConflict,
                FailureClass::Conflict,
            ),
            PluginError::NotFound(_) => (
                CommonPluginFailureCode::ResourceNotFound,
                FailureClass::NotFound,
            ),
            PluginError::Serialization(_) => (
                CommonPluginFailureCode::SerializationFailed,
                FailureClass::Unknown,
            ),
            PluginError::Internal(_) => (CommonPluginFailureCode::Internal, FailureClass::Internal),
            PluginError::RegistrationFailed(_) => (
                CommonPluginFailureCode::RegistrationFailed,
                FailureClass::Unknown,
            ),
        };
        common_plugin(code, class)
    }
}
