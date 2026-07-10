// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in LLM response cache (exact-match): a feature of the adaptive plugin,
//! configured through [`crate::config::AdaptiveConfig::response_cache`].
//!
//! [`intercept`] holds the execution intercepts and storage rules, [`key`] the
//! cache-key derivation, [`store`] the backends, [`replay`] the streaming
//! replay, and [`mark`] the observability surface.

pub mod config;
pub(crate) mod intercept;
pub(crate) mod key;
pub(crate) mod mark;
pub(crate) mod replay;
/// Public only for the integration-test crate and the CLI doctor's backend
/// health check; not part of the user-facing API.
#[doc(hidden)]
pub mod store;

pub use crate::config::ResponseCacheConfig;
pub use crate::response_cache::config::{BackendConfig, KEY_STRATEGY_EXACT_REQUEST};
pub(crate) use crate::response_cache::intercept::{make_intercept, make_stream_intercept};
pub use crate::response_cache::mark::RESPONSE_CACHE_MARK;
pub(crate) use crate::response_cache::store::build_store;
#[doc(hidden)]
pub use crate::response_cache::store::check_backend_health;
