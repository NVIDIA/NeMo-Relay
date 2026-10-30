# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_relay import diagnostics

reader = diagnostics.reader()
try:
    reader.snapshot({"unknown": True})
except diagnostics.DiagnosticsError as error:
    if error.reason != "invalid_query":
        raise
