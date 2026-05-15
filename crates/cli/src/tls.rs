// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) fn install_rustls_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}
