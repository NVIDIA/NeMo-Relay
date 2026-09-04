// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod client;
mod config;
mod metadata;
mod orchestrate;
mod provider;
mod resources;

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use config::{LoadOptions, ProcessSpec, TargetHeader, TargetSpec};

#[derive(Debug, Parser)]
#[command(about = "Hyper-based transport benchmark for the NeMo Relay daemon")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the deterministic OpenAI/Anthropic streaming provider.
    Provider(ProviderArgs),
    /// Benchmark already-running direct and Relay topology endpoints.
    Load(LoadArgs),
    /// Run the short, direct-provider HTTP/1.1 and HTTP/2 CI check.
    Smoke(SmokeArgs),
}

#[derive(Debug, Args)]
struct ProviderArgs {
    #[arg(long, default_value = "127.0.0.1:48100")]
    bind: SocketAddr,

    /// Write the selected URL after the listener is bound.
    #[arg(long)]
    ready_file: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct LoadArgs {
    #[arg(
        long,
        default_value = "scripts/latency_benchmark/config/daemon-transport-full.toml"
    )]
    config: PathBuf,

    /// URL of the deterministic provider, used as the direct baseline.
    #[arg(long)]
    direct_url: String,

    /// Additional public topology endpoint as NAME=URL. Names are daemon-pass-through and daemon-worker.
    #[arg(long = "target")]
    targets: Vec<TargetSpec>,

    /// Relay binary used to securely orchestrate a directly measured worker-only target.
    #[arg(long)]
    worker_binary: Option<PathBuf>,

    /// Read a target-specific request header from the environment as TARGET:HEADER=ENV_NAME.
    #[arg(long = "header-env")]
    headers: Vec<TargetHeader>,

    /// Process sampled for CPU/RSS metadata as NAME=PID. The load driver is always sampled.
    #[arg(long = "pid")]
    processes: Vec<ProcessSpec>,

    /// Relay binary recorded in metadata as PROFILE=PATH. Values are hashed but never executed.
    #[arg(long = "binary-metadata")]
    binaries: Vec<metadata::BinarySpec>,

    #[arg(long, default_value = "target/benchmark-results/daemon-transport.json")]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct SmokeArgs {
    #[arg(
        long,
        default_value = "target/benchmark-results/daemon-transport-smoke.json"
    )]
    output: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Provider(args) => provider::run(args.bind, args.ready_file).await,
        Command::Load(args) => {
            let mut options = LoadOptions::from_file(
                &args.config,
                args.direct_url,
                args.targets,
                args.headers,
                args.processes,
                args.binaries,
                args.output,
            )?;
            let harness = match args.worker_binary.as_deref() {
                Some(binary) => {
                    let harness = orchestrate::WorkerHarness::start(
                        binary,
                        &options
                            .targets
                            .iter()
                            .find(|target| target.topology == config::Topology::Direct)
                            .expect("LoadOptions always contains a direct target")
                            .url,
                    )
                    .await?;
                    harness.add_to(&mut options)?;
                    Some(harness)
                }
                None => None,
            };
            let result = client::run(options).await;
            let cleanup = match harness {
                Some(harness) => harness.shutdown().await,
                None => Ok(()),
            };
            match (result, cleanup) {
                (Err(error), _) => Err(error),
                (Ok(()), Err(error)) => Err(error),
                (Ok(()), Ok(())) => Ok(()),
            }
        }
        Command::Smoke(args) => client::run_smoke(args.output).await,
    }
}
