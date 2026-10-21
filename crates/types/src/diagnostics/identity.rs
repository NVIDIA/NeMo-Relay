// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source, component, and original attribution contracts.

use super::DiagnosticNumber;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// UTC wall-clock time when the source observed evidence or captured a snapshot.
///
/// This timestamp supports correlation, not elapsed time across machines.
/// Collectors track receipt time with their own clocks.
pub type ObservationTime = DateTime<Utc>;

/// Identity of one Relay process or explicitly owned runtime incarnation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SourceIdentity {
    /// Unique incarnation assigned before logging initialization.
    pub instance_id: Uuid,
    /// Descriptive operating system process ID, never an identity key.
    pub process_id: Option<u32>,
    /// Producer-defined role, such as embedded runtime, gateway, broker, or worker.
    pub process_role: String,
    /// Relay package version producing the evidence.
    pub relay_version: String,
}

/// Identity of one registered component within a diagnostic source.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ComponentIdentity {
    /// Stable logical name, excluding request and session identifiers.
    pub key: String,
    /// Unique registration instance, distinguishing replacement components.
    pub instance_id: Uuid,
    /// Source-local registration generation, increasing when a logical key is replaced.
    pub generation: DiagnosticNumber,
}

/// Bounded attribution for a component refresh or an originating observation.
///
/// Core bounds strings on admission. Nonempty fields also act as an intersection
/// selector in a query; request and session IDs belong in correlation instead.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ComponentOwnership {
    /// Subscriber registration associated with this attribution, when applicable.
    pub subscriber_instance_id: Option<Uuid>,
    /// Plugin registration associated with this attribution, when applicable.
    pub plugin_instance_id: Option<Uuid>,
    /// Bounded identity of the committed configuration.
    pub configuration_id: Option<String>,
    /// Bounded runtime responsibility for an embedded or process owner.
    pub responsibility: Option<String>,
}

/// Original bounded configuration attribution carried by one observation.
///
/// Captured origin metadata can survive a refresh without retaining a historical
/// counter or window store. It is not a separately queryable aggregate identity.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DiagnosticOrigin {
    /// Refresh generation associated with the originating work or observation.
    pub refresh_generation: DiagnosticNumber,
    /// Original bounded owner and configuration attribution.
    pub ownership: ComponentOwnership,
}
