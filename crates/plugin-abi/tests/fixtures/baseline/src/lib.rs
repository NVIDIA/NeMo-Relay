// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independently compiled checked native contract fixture.
#[path = "../../behavior_plugin.rs"]
mod behavior;
abi::export_factory!(
    kind = "fixture",
    create = || Ok(abi::v1_0::PluginBox::new(behavior::Example))
);
