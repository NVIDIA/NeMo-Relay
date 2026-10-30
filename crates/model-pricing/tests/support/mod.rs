// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) fn pricing_test_mutex() -> &'static std::sync::Mutex<()> {
    static MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
    &MUTEX
}
