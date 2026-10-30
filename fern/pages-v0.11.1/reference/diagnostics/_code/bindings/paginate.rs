// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

let mut query = DiagnosticQuery {
    max_components: Some(1),
    max_events: Some(1),
    ..Default::default()
};
let mut complete = false;
for _ in 0..32 {
    let page = reader.snapshot(&query)?;
    let snapshot: DiagnosticSnapshot = serde_json::from_str(page.as_str())?;
    // Inspect snapshot.components and snapshot.events within the authorized boundary.
    query.continuation = snapshot.continuation;
    if query.continuation.is_none() {
        complete = true;
        break;
    }
} // Each page's adapter permit is returned on drop.
if !complete {
    return Err("The caller's page budget was exhausted".into());
}
