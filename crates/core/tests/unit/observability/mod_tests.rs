// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{relay_span_id, relay_trace_id};
use uuid::Uuid;

#[test]
fn relay_id_conversions_preserve_zero_bytes() {
    let uuid = Uuid::nil();

    assert_eq!(relay_trace_id(uuid).to_bytes(), [0; 16]);
    assert_eq!(relay_span_id(uuid).to_bytes(), [0; 8]);
}
