// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-user-machine route lifecycle and lock-bounded broker registry.

pub(crate) mod lifecycle;
pub(crate) mod registry;
pub(crate) mod server;
