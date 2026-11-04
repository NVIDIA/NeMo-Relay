// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{atomic_private_write, create_private_dir_all};

#[test]
fn absolute_temp_directory_is_accepted() {
    let temporary = tempfile::tempdir().expect("temporary directory should be created");
    let output = temporary.path().join("atof");

    create_private_dir_all(&output).expect("absolute Windows output directory should open");
    atomic_private_write(&output, &output.join("trajectory.json"), b"{}")
        .expect("absolute Windows output file should write atomically");

    assert_eq!(
        std::fs::read(output.join("trajectory.json")).unwrap(),
        b"{}"
    );
}
