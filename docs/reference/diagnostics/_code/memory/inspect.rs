// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::diagnostics::DiagnosticQuery;

let query = DiagnosticQuery {
    max_components: Some(0),
    max_events: Some(1),
    ..Default::default()
};
let page = reader.inspect_memory(&query)?;
let encoded: &[u8] = page.as_bytes();
// Inspect the encoded page inside the authorized application boundary.
drop(page); // Return the query workspace.
