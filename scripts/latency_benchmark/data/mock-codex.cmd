@REM SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
@REM SPDX-License-Identifier: Apache-2.0
@echo off
if "%1"=="--version" (
  echo codex-cli 0.143.0
  exit /b 0
)
> "%BENCHMARK_GATEWAY_FILE%" echo %NEMO_RELAY_GATEWAY_URL%
:wait
if exist "%BENCHMARK_STOP_FILE%" exit /b 0
ping 127.0.0.1 -n 2 >nul
goto wait
