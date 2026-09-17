// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Retry policy shared by direct gateways and daemon-managed workers for TypeSafe APIs.

use std::time::{Duration, SystemTime};

use http::HeaderMap;
use ring::rand::{SecureRandom, SystemRandom};

pub(crate) const DEFAULT_MAX_RETRIES: u32 = 2;
pub(crate) const DEFAULT_MAX_SERVER_DELAY_MILLIS: u64 = 60_000;
const INITIAL_BACKOFF_MILLIS: u64 = 500;
const MAX_BACKOFF_MILLIS: u64 = 5_000;
const JITTER_RATIO: f64 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TypeSafeRetryPolicy {
    pub(crate) max_retries: u32,
    pub(crate) max_server_delay: Duration,
}

impl Default for TypeSafeRetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: DEFAULT_MAX_RETRIES,
            max_server_delay: Duration::from_millis(DEFAULT_MAX_SERVER_DELAY_MILLIS),
        }
    }
}

impl TypeSafeRetryPolicy {
    pub(crate) fn should_retry_status(self, status: http::StatusCode, retry_count: u32) -> bool {
        retry_count < self.max_retries
            && (status == http::StatusCode::REQUEST_TIMEOUT
                || status == http::StatusCode::TOO_MANY_REQUESTS
                || status.is_server_error())
    }

    pub(crate) fn can_retry_transport(self, retry_count: u32) -> bool {
        retry_count < self.max_retries
    }

    pub(crate) fn delay(self, headers: Option<&HeaderMap>, retry_count: u32) -> Duration {
        if let Some(delay) = headers.and_then(server_retry_delay)
            && delay <= self.max_server_delay
        {
            return delay;
        }
        jittered_backoff(retry_count, random_unit_interval())
    }
}

fn server_retry_delay(headers: &HeaderMap) -> Option<Duration> {
    if let Some(delay) = headers
        .get("retry-after-ms")
        .and_then(|value| value.to_str().ok())
        .and_then(parse_millis)
    {
        return Some(delay);
    }
    let value = headers
        .get(http::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim();
    if let Ok(seconds) = value.parse::<f64>() {
        return finite_nonnegative_duration(seconds * 1_000.0);
    }
    let retry_at = httpdate::parse_http_date(value).ok()?;
    Some(
        retry_at
            .duration_since(SystemTime::now())
            .unwrap_or_default(),
    )
}

fn parse_millis(value: &str) -> Option<Duration> {
    finite_nonnegative_duration(value.trim().parse::<f64>().ok()?)
}

fn finite_nonnegative_duration(millis: f64) -> Option<Duration> {
    (millis.is_finite() && millis >= 0.0).then(|| Duration::from_secs_f64(millis / 1_000.0))
}

fn jittered_backoff(retry_count: u32, unit: f64) -> Duration {
    let base = INITIAL_BACKOFF_MILLIS
        .saturating_mul(1_u64 << retry_count.min(3))
        .min(MAX_BACKOFF_MILLIS) as f64;
    // Match both official SDKs: jitter is randomly subtracted from the exponential delay,
    // producing 75%-100% of the unjittered value at the default ratio.
    let factor = 1.0 - (JITTER_RATIO * unit.clamp(0.0, 1.0));
    Duration::from_millis((base * factor).round() as u64)
}

fn random_unit_interval() -> f64 {
    let mut bytes = [0_u8; 8];
    if SystemRandom::new().fill(&mut bytes).is_err() {
        return 0.5;
    }
    u64::from_le_bytes(bytes) as f64 / u64::MAX as f64
}
