// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(unix)]
#[path = "confined_fs/unix.rs"]
mod platform;
#[cfg(windows)]
#[path = "confined_fs/windows.rs"]
mod platform;

#[cfg(not(any(unix, windows)))]
compile_error!("private observability files require Unix or Windows filesystem primitives");

pub(in crate::observability) use platform::ConfinedDir;
