// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independently compiled checked native contract fixture.
abi::export_factory!(
    kind = "fixture",
    create = || panic!("incompatible factory must not run")
);
