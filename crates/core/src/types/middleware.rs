// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub struct Intercept<F> {
    pub priority: i32,
    pub break_chain: bool,
    pub callable: F,
}

pub struct ExecutionIntercept<F> {
    pub priority: i32,
    pub callable: F,
}

pub struct GuardrailEntry<F> {
    pub priority: i32,
    pub guardrail: F,
}
