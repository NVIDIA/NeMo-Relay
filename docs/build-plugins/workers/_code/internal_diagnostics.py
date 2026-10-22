# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_relay_plugin import (
    DiagnosticCodeDefinition,
    DiagnosticDeliveryError,
    DiagnosticObservation,
)
from nemo_relay_plugin import (
    DiagnosticObservationOutcome as Outcome,
)
from nemo_relay_plugin import (
    DiagnosticObservationSeverity as Severity,
)


async def record_internal_failure(runtime):
    diagnostics = await runtime.diagnostics()
    try:
        component = await diagnostics.component(
            "example.internal",
            [
                DiagnosticCodeDefinition(builtin_name="plugin.internal"),
            ],
        )
        component.try_record_lazy(
            lambda: DiagnosticObservation(
                code=1,
                count=1,
                failure_class="internal",
                outcome=Outcome.DIAGNOSTIC_OBSERVATION_OUTCOME_FAILURE,
                severity=Severity.DIAGNOSTIC_OBSERVATION_SEVERITY_ERROR,
                detail="The internal operation failed",
            )
        )
    except DiagnosticDeliveryError:
        # Optional evidence delivery must preserve the operation's original result.
        pass
