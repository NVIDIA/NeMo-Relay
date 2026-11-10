// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn callback_result_wait_is_bounded() {
    let (_tx, rx) = std::sync::mpsc::sync_channel(1);
    let error = recv_callback_result(&rx, Some(Duration::from_millis(1)), "test callback")
        .expect_err("an unresponsive callback must time out");

    assert!(
        error
            .to_string()
            .contains("test callback: callback timed out")
    );
}
