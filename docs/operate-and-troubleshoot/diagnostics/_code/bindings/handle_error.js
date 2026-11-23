// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

try {
  reader.snapshot({ unknown: true });
} catch (error) {
  if (!(error instanceof diagnostics.DiagnosticsError) || error.reason !== 'invalid_query') {
    throw error;
  }
}
