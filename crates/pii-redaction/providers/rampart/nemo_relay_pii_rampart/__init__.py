# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rampart inference provider for the NeMo Relay PII component."""

from .detector import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, RampartDetector, RampartSettings

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "RampartDetector",
    "RampartSettings",
]
