// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process entrypoint for the NeMo Relay coding-agent gateway.

use std::process::ExitCode;

fn main() -> ExitCode {
    nemo_relay_cli::run_cli()
}
