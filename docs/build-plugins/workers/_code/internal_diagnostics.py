# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_relay_plugin import (
    DiagnosticCodeDefinition,
    DiagnosticComponent,
    DiagnosticExtensionDefinition,
    DiagnosticObservation,
    PluginRuntime,
    WorkerDiagnostics,
)
from nemo_relay_plugin import (
    DiagnosticObservationOutcome as Outcome,
)
from nemo_relay_plugin import (
    DiagnosticObservationSeverity as Severity,
)


async def register_lookup_diagnostics(
    runtime: PluginRuntime,
) -> tuple[WorkerDiagnostics, DiagnosticComponent]:
    diagnostics = await runtime.diagnostics()
    component = await diagnostics.component(
        "example.lookup",
        [
            DiagnosticCodeDefinition(
                extension=DiagnosticExtensionDefinition(
                    name="lookup_failed",
                    description="The lookup backend was unavailable",
                    outcome=Outcome.DIAGNOSTIC_OBSERVATION_OUTCOME_FAILURE,
                    allowed_failure_classes=["unavailable"],
                )
            ),
            DiagnosticCodeDefinition(
                extension=DiagnosticExtensionDefinition(
                    name="lookup_completed",
                    description="A lookup completed",
                    outcome=Outcome.DIAGNOSTIC_OBSERVATION_OUTCOME_SUCCESS,
                    allowed_failure_classes=[],
                )
            ),
        ],
    )
    return diagnostics, component


def record_lookup_failure(component: DiagnosticComponent, request_id: str) -> None:
    component.try_record_lazy(
        lambda: DiagnosticObservation(
            code=1,
            count=1,
            outcome=Outcome.DIAGNOSTIC_OBSERVATION_OUTCOME_FAILURE,
            severity=Severity.DIAGNOSTIC_OBSERVATION_SEVERITY_ERROR,
            failure_class="unavailable",
            detail="The lookup backend was unavailable",
            request_id=request_id,
        )
    )


def record_lookup_success(component: DiagnosticComponent, request_id: str) -> None:
    component.try_record_lazy(
        lambda: DiagnosticObservation(
            code=2,
            count=1,
            outcome=Outcome.DIAGNOSTIC_OBSERVATION_OUTCOME_SUCCESS,
            severity=Severity.DIAGNOSTIC_OBSERVATION_SEVERITY_INFO,
            detail="A lookup completed",
            request_id=request_id,
        )
    )
