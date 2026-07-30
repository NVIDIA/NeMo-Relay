// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

//! First-party in-process Switchyard libsy routing plugin for NeMo Relay.

pub mod component;
mod stream_translation;
mod translation;

pub use component::{
    AlgorithmConfig, ProtocolDefaults, SWITCHYARD_PLUGIN_KIND, SwitchyardConfig, TargetBinding,
    WireProtocol, deregister_switchyard_component, register_switchyard_component,
};
