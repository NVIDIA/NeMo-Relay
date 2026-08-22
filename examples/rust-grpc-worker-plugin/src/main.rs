// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_rust_grpc_worker_plugin_example::DocumentationWorker;
use nemo_relay_worker::{Result, serve_plugin};

#[tokio::main]
async fn main() -> Result<()> {
    serve_plugin(DocumentationWorker).await
}
