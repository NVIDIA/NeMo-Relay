# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def normalize_ticket_id(value):
    raise NotImplementedError


def retry_delays(attempts, base, cap):
    raise NotImplementedError


def deployment_plan(graph, blocked=()):
    raise NotImplementedError
