// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script for the NeMo Relay worker gRPC protocol crate.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile the existing contract separately so its API stays available while
    // the replacement descriptors and consumers are introduced independently.
    {
        let mut legacy = prost_build::Config::new();
        legacy.protoc_executable(protoc_bin_vendored::protoc_bin_path()?);
        legacy.boxed(".nemo.relay.worker.v1.LlmInvocation.execution_codec_context");
        tonic_prost_build::configure().compile_with_config(
            legacy,
            &["proto/nemo/relay/worker/v1/plugin_worker.proto"],
            &["proto"],
        )?;
    }
    // Track the complete import graph, including supporting DTO schemas.
    println!("cargo:rerun-if-changed=proto");
    let protos = [
        "proto/nvidia/nemo/relay/worker/v1/plugin_worker_service.proto",
        "proto/nvidia/nemo/relay/worker/v1/relay_host_runtime_service.proto",
        "proto/nvidia/nemo/relay/worker/v1/relay_diagnostics_service.proto",
    ]
    .map(std::path::Path::new);
    let dependency = validation_dependency()?;
    let includes = [std::path::Path::new("proto"), dependency.as_path()];
    let mut prost = prost_build::Config::new();
    prost.enable_type_names();
    prost.protoc_executable(protoc_bin_vendored::protoc_bin_path()?);
    // Keep optional annotations
    // and large alternatives indirect without changing their protobuf wire shape.
    for field in [
        "CategoryProfile.annotated_request",
        "CategoryProfile.annotated_response",
        "LlmRequestInvocation.annotated_request",
        "LlmRequestInterceptOutcome.annotated_request",
        "InvokeResponse.result.event_sanitize",
        "InvokeResponse.result.llm_request_outcome",
        "DecodeLlmCodecRequestResponse.result.value",
        "DecodeLlmCodecResponseResponse.result.value",
        "ReadDiagnosticPageResponse.result.snapshot",
        "GetDiagnosticDiscoveryResponse.result.discovery",
        "InspectDiagnosticMemoryResponse.result.memory",
    ] {
        prost.boxed(format!(".nvidia.nemo.relay.worker.v1.{field}"));
    }
    prost.file_descriptor_set_path(
        std::path::PathBuf::from(std::env::var("OUT_DIR")?).join("worker_descriptor.bin"),
    );

    tonic_prost_build::configure().compile_with_config(prost, &protos, &includes)?;
    Ok(())
}

/// Export only the locked dependency into Cargo's build directory.
fn validation_dependency() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-env-changed=BUF");
    println!("cargo:rerun-if-changed=buf.lock");
    let lock = include_str!("buf.lock");
    let dependency = lock
        .split_once("name: buf.build/bufbuild/protovalidate")
        .map(|(_, dependency)| dependency)
        .ok_or("buf.lock must pin Protovalidate")?;
    let commit = dependency
        .lines()
        .find_map(|line| line.trim().strip_prefix("commit: "))
        .filter(|commit| commit.len() == 32 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .ok_or("buf.lock must pin an immutable Protovalidate commit")?;
    let output = std::path::PathBuf::from(std::env::var("OUT_DIR")?)
        .join("proto-deps")
        .join(commit);
    if !output.join("buf/validate/validate.proto").is_file() {
        let buf = std::env::var_os("BUF").unwrap_or_else(|| "buf".into());
        let status = std::process::Command::new(buf)
            .args(["export", &format!("buf.build/bufbuild/protovalidate:{commit}"),
                "--path", "buf/validate/validate.proto", "--output"])
            .arg(&output).status()
            .map_err(|error| format!("Buf is required to build nemo-relay-worker-proto; install Buf or set BUF: {error}"))?;
        if !status.success() {
            return Err(
                "Buf could not export the Protovalidate dependency pinned in buf.lock".into(),
            );
        }
    }
    Ok(output)
}
