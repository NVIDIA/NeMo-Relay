// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const query = { maxComponents: 1, maxEvents: 1 };
let complete = false;
for (let remaining = 32; remaining > 0; --remaining) {
  const page = reader.snapshot(query);
  // Inspect page.components and page.events within the authorized boundary.
  if (page.continuation === null) {
    complete = true;
    break;
  }
  query.continuation = page.continuation;
}
if (!complete) throw new Error("The caller's page budget was exhausted");
