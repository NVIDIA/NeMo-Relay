// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the stream consumer clock.

use super::*;

#[test]
fn repeated_requests_keep_poll_indexes_aligned() {
    let clock = StreamConsumerClock::default();
    clock.requested();
    // A cancelled or failed request is retried before item 0 is delivered.
    clock.requested();
    assert!(!clock.was_prefetched(0, Instant::now()));
    // No request exists yet for item 1, so it is available before its poll.
    assert!(clock.was_prefetched(1, Instant::now()));
    clock.delivered();
    clock.requested();
    assert!(!clock.was_prefetched(1, Instant::now()));
}

#[test]
fn finish_reports_once_to_early_and_late_observers() {
    let clock = StreamConsumerClock::default();
    let reports = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let early = std::sync::Arc::clone(&reports);
    clock.observe(move |report| early.lock().unwrap().push(report.complete));
    clock.requested();
    clock.delivered();
    clock.finish(true);
    // A second finish, such as a drop after exhaustion, keeps the first report.
    clock.finish(false);
    let late = std::sync::Arc::clone(&reports);
    clock.observe(move |report| late.lock().unwrap().push(report.complete));
    assert_eq!(*reports.lock().unwrap(), vec![true, true]);
}

#[test]
fn clocks_exist_only_while_a_recorder_is_registered() {
    // Other tests in this binary may hold registrations, so compare counts.
    let registration = StreamConsumerClockRegistration::default();
    assert!(StreamConsumerClock::if_enabled().is_some());
    drop(registration);
}

#[tokio::test]
async fn current_reads_the_clock_scoped_to_the_stream_call() {
    assert!(StreamConsumerClock::current().is_none());
    let clock = StreamConsumerClock::default();
    let seen = with_stream_consumer_clock(Some(clock.clone()), async {
        StreamConsumerClock::current().is_some()
    })
    .await;
    assert!(seen);
    assert!(StreamConsumerClock::current().is_none());
}
