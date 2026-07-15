// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) fn hook_status() -> Result<String, String> {
    Ok("hooks: injected during run".into())
}
