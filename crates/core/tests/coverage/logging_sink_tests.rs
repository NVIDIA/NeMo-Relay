// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{DROP_REPORT_INTERVAL_MILLIS, DropNoticeRateLimiter};

#[test]
fn drop_notice_rate_limiter_reports_immediately_then_once_per_interval() {
    let rate_limiter = DropNoticeRateLimiter::new();
    let interval = DROP_REPORT_INTERVAL_MILLIS;
    let first_timestamp = 10 * interval;

    assert!(rate_limiter.should_report(first_timestamp));
    assert!(!rate_limiter.should_report(first_timestamp + interval - 1));
    assert!(rate_limiter.should_report(first_timestamp + interval));
}
