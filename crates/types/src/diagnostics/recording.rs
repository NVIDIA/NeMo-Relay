// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Recording acceptance, independent of retained evidence and output delivery.

use super::DiagnosticCount;

/// Evidence retention is independent of numeric acceptance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceDisposition {
    /// The supplied evidence was retained in full.
    Retained,
    /// Optional detail or correlation was shortened or omitted.
    Truncated,
    /// Counters were updated without retaining an incoming event header.
    Dropped,
}

/// Accepted lifetime and current-summary update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordReceipt {
    /// Source-local accepted-update sequence, not an occurrence count.
    pub sequence: u64,
    /// Code total; absent for summary-only classification overflow.
    pub code_total: Option<DiagnosticCount>,
    /// Independent evidence retention result.
    pub evidence: EvidenceDisposition,
}

/// Accepted collection or intentional disablement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordOutcome {
    /// Numeric state was accepted once.
    Recorded(RecordReceipt),
    /// No observation or loss accounting was produced.
    Disabled,
}
