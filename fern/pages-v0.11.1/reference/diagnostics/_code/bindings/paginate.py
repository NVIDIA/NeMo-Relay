# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_relay import diagnostics

reader = diagnostics.reader()
query = {"max_components": 1, "max_events": 1}
for _ in range(32):
    page = reader.snapshot(query)
    # Inspect page["components"] and page["events"] within the authorized boundary.
    if page["continuation"] is None:
        break
    query["continuation"] = page["continuation"]
else:
    raise RuntimeError("The caller's page budget was exhausted")
