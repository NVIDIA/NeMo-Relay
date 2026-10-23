# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native shell execution, session state, and bounded process-tree ownership."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path

WINDOWS = os.name == "nt"
NAME = "powershell" if WINDOWS else "bash"


def quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'" if WINDOWS else shlex.quote(str(value))


def command(*arguments: str | Path) -> str:
    return ("& " if WINDOWS else "") + " ".join(quote(value) for value in arguments)


def python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if WINDOWS else "bin/python")


def prepend(directory: Path) -> str:
    if WINDOWS:
        return "$env:PATH = " + quote(str(directory) + os.pathsep) + " + $env:PATH"
    return "export PATH=" + quote(directory) + ':"$PATH"'


def session(content: str, cwd: Path, state: Path, previous: Path | None, diagnostic: bool) -> str:
    if WINDOWS:
        prefix = (
            f"$saved = Get-Content -Raw -LiteralPath {quote(previous)} | ConvertFrom-Json -AsHashtable\n"
            "foreach ($key in $saved.Keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key]) }\n"
            if previous
            else ""
        )
        save = (
            "[Environment]::GetEnvironmentVariables('Process') | ConvertTo-Json | "
            f"Set-Content -LiteralPath {quote(state)} -Encoding utf8\n"
        )
        return (
            prefix
            + f"Set-Location -LiteralPath {quote(cwd)}\n"
            + ("$PSNativeCommandUseErrorActionPreference = $false\n" if diagnostic else "")
            + content
            + "\n"
            + ("$doctorStatus = $LASTEXITCODE\n" if diagnostic else "")
            + save
            + ("exit $doctorStatus\n" if diagnostic else "")
        )
    prefix = "source " + quote(previous) + "\n" if previous else ""
    return (
        prefix
        + "cd "
        + quote(cwd)
        + "\n"
        + ("set +e\n" if diagnostic else "")
        + content
        + "\n"
        + ("quickstart_doctor_status=$?\n" if diagnostic else "")
        + "export -p > "
        + quote(state)
        + "\n"
        + ('exit "$quickstart_doctor_status"\n' if diagnostic else "")
    )


def execute(command: str, cwd: Path, environment: dict[str, str], stdout: Path, stderr: Path, timeout: int) -> None:
    with tempfile.TemporaryDirectory(prefix="relay-shell-") if WINDOWS else nullcontext() as temporary:
        if WINDOWS:
            # Relay's existing controller starts a worker outside the host's
            # restrictive job and owns every descendant through kill-on-close.
            # Killing the controller on timeout therefore also kills detached agents.
            folder = Path(temporary or "")
            script = folder / "command.ps1"
            script.write_text(
                "$ErrorActionPreference = 'Stop'\n$PSNativeCommandUseErrorActionPreference = $true\n"
                "[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)\n"
                "$OutputEncoding = [Console]::OutputEncoding\n" + command,
                encoding="utf-8",
            )
            payload = folder / "request.json"
            environment = {**environment, "RUNNER_TEMP": environment.get("RUNNER_TEMP", tempfile.gettempdir())}
            payload.write_text(
                json.dumps(
                    {
                        "Executable": shutil.which("pwsh"),
                        "Arguments": ["-NoProfile", "-NonInteractive", "-File", str(script)],
                        "Environment": environment,
                    }
                ),
                encoding="utf-8",
            )
            wrapper = Path(__file__).resolve().parents[2] / ".github/scripts/run-windows-tests.ps1"
            arguments = ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(wrapper), "-PayloadPath", str(payload)]
        else:
            arguments = ["bash", "-euo", "pipefail", "-c", command]
        with stdout.open("wb") as out, stderr.open("wb") as err:
            process = subprocess.Popen(
                arguments,
                cwd=cwd,
                env=environment,
                stdout=out,
                stderr=err,
                start_new_session=not WINDOWS,
            )
            try:
                code = process.wait(timeout=timeout)
            finally:
                if WINDOWS:
                    if process.poll() is None:
                        process.kill()
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
    if code:
        raise ValueError(f"Command exited {code}; see {stderr}")
