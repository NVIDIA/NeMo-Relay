// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small operational-log helpers shared by daemon processes.

use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

struct State {
    next_emit: Option<tokio::time::Instant>,
    suppressed: u64,
}

pub(crate) struct LogRateLimiter {
    interval: Duration,
    state: Mutex<State>,
}

impl LogRateLimiter {
    pub(crate) fn new(interval: Duration) -> Self {
        Self {
            interval,
            state: Mutex::new(State {
                next_emit: None,
                suppressed: 0,
            }),
        }
    }

    /// Returns the number of failures suppressed since the previous emitted event.
    pub(crate) fn record(&self) -> Option<u64> {
        let now = tokio::time::Instant::now();
        let mut state = lock(&self.state);
        if state.next_emit.is_some_and(|deadline| now < deadline) {
            state.suppressed = state.suppressed.saturating_add(1);
            return None;
        }
        let suppressed = std::mem::take(&mut state.suppressed);
        state.next_emit = Some(now + self.interval);
        Some(suppressed)
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}
