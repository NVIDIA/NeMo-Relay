# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private limits for bounded Python diagnostics query normalization."""

import json

from nemo_relay._native import _diagnostics_query_contract

_CORE_CONTRACT = json.loads(_diagnostics_query_contract())

# These traversal caps bound the pre-serialization object walk in this adapter.
MAX_QUERY_BYTES = _CORE_CONTRACT["max_query_bytes"]
MAX_QUERY_NODES = 64
MAX_QUERY_DEPTH = 4
MAX_QUERY_OBJECT_FIELDS = 20
MAX_QUERY_FIELD_NAME_BYTES = 128
NUMERIC_QUERY_FIELDS = frozenset(_CORE_CONTRACT["numeric_query_fields"])
