// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synthetic future interface. Never add these fields to v1.0.
use abi_stable::{StableAbi,library::RootModule,sabi_types::VersionStrings};
use crate::v1_0;
#[repr(C)]
#[derive(StableAbi)]
#[sabi(kind(Prefix(prefix_ref = Factory_Ref)))]
pub struct Factory {
 pub v1_0: v1_0::Factory_Ref,
 #[sabi(last_prefix_field)]
 pub revision: extern "C" fn()->u32,
}
impl RootModule for Factory_Ref {
 abi_stable::declare_root_module_statics!{Factory_Ref}
 const BASE_NAME: &'static str="nemo_relay_plugin";
 const NAME: &'static str="nemo.relay.native.factory.v1.1";
 const VERSION_STRINGS:VersionStrings=VersionStrings::new("1.0.0");
}
