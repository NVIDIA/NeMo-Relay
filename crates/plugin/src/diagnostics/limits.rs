// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native diagnostic admission limits.

/// Typed diagnostic transport revision.
pub const CONTRACT_VERSION: u32 = 1;
/// Maximum simultaneously live component registrations per activation.
pub const MAX_COMPONENTS: usize = 8;
/// Maximum declared codes per component.
pub const MAX_CODES: usize = 32;
/// Maximum cumulative extension declarations per activation.
pub const MAX_EXTENSIONS: usize = 16;
/// Maximum classes per extension.
pub const MAX_CLASSES: usize = 16;
/// Maximum total text bytes in a native request.
pub const MAX_REQUEST_BYTES: usize = 16 * 1024;
/// Maximum concurrently admitted authoring sessions per host process.
pub const MAX_SESSIONS: usize = 32;
/// Maximum concurrent native data calls per host process.
pub const MAX_CALLS: usize = 4;
