// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub fn log_event(event: &nemo_relay::Event) {
    println!("{}", event.to_json_string().expect("serialize Relay event"));
}
