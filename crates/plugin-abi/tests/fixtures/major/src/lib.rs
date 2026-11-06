// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independently compiled native compatibility fixture.
#![allow(non_camel_case_types, non_local_definitions)]
use std::sync::OnceLock;
use abi::{Bootstrap,Bootstrap_Ref,PluginDescriptor,NativeApiVersion,Json};
use abi::v1_0::*;
use abi::abi_stable::{export_root_module,library::LibHeader,prefix_type::PrefixTypeTrait,sabi_trait::prelude::TD_Opaque,sabi_types::RRef,std_types::{ROption,RStr,RVec}};
use abi::async_ffi::{FfiFuture,FutureExt};
struct Example;
impl Plugin for Example {
 fn allows_multiple_components(&self)->bool{true}
 fn validate(&self,_config:Json)->RVec<ConfigDiagnostic>{RVec::new()}
 fn register(&self,_config:Json,_runtime:RuntimeBox)->FfiFuture<Result<RegisteredComponent>>{async{Result::RErr(Error::new("fixture","registration is not exercised by layout fixtures"))}.into_ffi()}
}
extern "C" fn create()->Result<PluginBox>{Result::ROk(Plugin_TO::from_value(Example,TD_Opaque))}
extern "C" fn descriptor()->PluginDescriptor{PluginDescriptor{native_api:NativeApiVersion::new(2,0),plugin_kind:"fixture".into()}}
fn factory_header()->&'static LibHeader{
 static HEADER:OnceLock<LibHeader>=OnceLock::new();
 HEADER.get_or_init(||LibHeader::from_module(Factory{create}.leak_into_prefix()))
}

extern "C" fn query(name:RStr<'_>)->ROption<RRef<'static,LibHeader>>{

 if name.as_str()==FACTORY_MODULE {ROption::RSome(RRef::new(factory_header()))} else {ROption::RNone}
}
#[export_root_module]
pub fn module()->Bootstrap_Ref{Bootstrap{descriptor,query_module:query}.leak_into_prefix()}
