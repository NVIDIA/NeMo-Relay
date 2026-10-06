// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn conditional_gate_result_wait_is_bounded() {
    let (_tx, rx) = std::sync::mpsc::sync_channel(1);
    let error = recv_conditional_gate_result(rx, std::time::Duration::from_millis(1))
        .expect_err("an unresponsive callback must time out");

    assert!(error.reason.contains("conditional gate callback timed out"));
}
