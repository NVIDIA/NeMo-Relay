// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

match reader.snapshot_json(r#"{"unknown":true}"#) {
    Err(error) if error.reason() == "invalid_query" => {}
    Err(error) => return Err(error.into()),
    Ok(_) => return Err("An unknown query field was accepted".into()),
}
