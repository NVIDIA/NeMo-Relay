// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub fn print_diagnostics(
    exporter: &nemo_relay::exporter::atof::AtofExporter,
) -> Result<(), Box<dyn std::error::Error>> {
    exporter.force_flush()?;
    for diagnostic in exporter.runtime_diagnostics().entries() {
        println!(
            "{} {} {}",
            diagnostic.code, diagnostic.count, diagnostic.message
        );
    }
    Ok(())
}
