// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared daemon control-plane and transport primitives.

pub(crate) mod address;
pub(crate) mod client;
pub(crate) mod control;
pub(crate) mod identity;
pub(crate) mod protocol;
pub(crate) mod routes;
pub(crate) mod state;
pub(crate) mod transport;
pub(crate) mod worker_tls;
