// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script for the NeMo Relay worker gRPC protocol crate.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto = "proto/nemo/relay/worker/v1/plugin_worker.proto";
    let include = "proto";
    let mut prost = prost_build::Config::new();
    prost.protoc_executable(protoc_bin_vendored::protoc_bin_path()?);

    tonic_prost_build::configure().compile_with_config(prost, &[proto], &[include])?;
    Ok(())
}
