# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_relay import diagnostics

reader = diagnostics.reader()
page = reader.inspect_memory({"max_components": 0, "max_events": 1})
