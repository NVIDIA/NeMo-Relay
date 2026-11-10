// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{JAVASCRIPT_MAX_SAFE_INTEGER, NodeFrom, NodeResourceMetricInteger};

#[test]
fn preserves_integer_values_around_the_javascript_safe_integer_boundary() {
    let below_boundary = JAVASCRIPT_MAX_SAFE_INTEGER - 1;

    assert_eq!(
        NodeResourceMetricInteger::node_from(below_boundary),
        NodeResourceMetricInteger::Number(below_boundary as f64)
    );
    assert_eq!(
        NodeResourceMetricInteger::node_from(JAVASCRIPT_MAX_SAFE_INTEGER),
        NodeResourceMetricInteger::Number(JAVASCRIPT_MAX_SAFE_INTEGER as f64)
    );

    let above_boundary = JAVASCRIPT_MAX_SAFE_INTEGER + 1;
    assert_eq!(
        NodeResourceMetricInteger::node_from(above_boundary),
        NodeResourceMetricInteger::BigInt(above_boundary)
    );
}
