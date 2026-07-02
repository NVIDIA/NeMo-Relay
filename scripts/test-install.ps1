# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Installer = Join-Path $RepoRoot 'install.ps1'
$PowerShell = (Get-Process -Id $PID).Path
$TestRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("nemo-relay-install-test-" + [guid]::NewGuid().ToString('N'))
$InstallDir = Join-Path $TestRoot 'custom-bin'
$OriginalUserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$OriginalProcessPath = $env:Path
$OriginalVersion = $env:NEMO_RELAY_VERSION
$TestsRun = 0

function Fail([string]$Message) {
    throw "FAIL: $Message"
}

function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) {
        Fail $Message
    }
}

function Assert-Contains([string]$Text, [string]$Expected) {
    Assert-True ($Text.Contains($Expected)) "expected '$Expected' in: $Text"
}

function Assert-PathContains([string]$PathValue, [string]$ExpectedDirectory) {
    $expected = [System.IO.Path]::GetFullPath($ExpectedDirectory).TrimEnd('\')
    foreach ($entry in ($PathValue -split ';')) {
        if ($entry -and [string]::Equals($entry.Trim().TrimEnd('\'), $expected, [System.StringComparison]::OrdinalIgnoreCase)) {
            return
        }
    }
    Fail "expected PATH to contain $expected"
}

function Assert-NoTemporaryFiles([string]$Directory) {
    if (Test-Path -LiteralPath $Directory) {
        $temporaryFiles = Get-ChildItem -LiteralPath $Directory -Filter '.nemo-relay.*' -Force
        Assert-True ($temporaryFiles.Count -eq 0) "temporary installer files were not cleaned up in $Directory"
    }
}

function Get-ExpectedTarget {
    $architecture = $env:PROCESSOR_ARCHITEW6432
    if ([string]::IsNullOrWhiteSpace($architecture)) {
        $architecture = $env:PROCESSOR_ARCHITECTURE
    }
    switch ($architecture.ToUpperInvariant()) {
        'AMD64' { return 'x86_64-pc-windows-msvc' }
        'X86_64' { return 'x86_64-pc-windows-msvc' }
        'ARM64' { return 'aarch64-pc-windows-msvc' }
        default { Fail "unexpected test architecture $architecture" }
    }
}

function Invoke-Installer {
    param(
        [string]$Directory,
        [switch]$CaptureProcessPath,
        [string[]]$Arguments = @()
    )

    if ($CaptureProcessPath) {
        $previousInstallerPath = $env:NEMO_RELAY_TEST_INSTALLER
        $previousInstallDir = $env:NEMO_RELAY_TEST_INSTALL_DIR
        $env:NEMO_RELAY_TEST_INSTALLER = $Installer
        $env:NEMO_RELAY_TEST_INSTALL_DIR = $Directory
        try {
            $command = '& $env:NEMO_RELAY_TEST_INSTALLER -InstallDir $env:NEMO_RELAY_TEST_INSTALL_DIR; Write-Output "__NEMO_RELAY_PATH__$env:Path"'
            $script:RunOutput = (& $PowerShell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command $command 2>&1 | Out-String)
        }
        finally {
            $env:NEMO_RELAY_TEST_INSTALLER = $previousInstallerPath
            $env:NEMO_RELAY_TEST_INSTALL_DIR = $previousInstallDir
        }
    }
    else {
        $script:RunOutput = (& $PowerShell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $Installer @Arguments 2>&1 | Out-String)
    }
    $script:RunStatus = $LASTEXITCODE
}

function Assert-Success {
    Assert-True ($RunStatus -eq 0) "expected success, got ${RunStatus}: $RunOutput"
}

function Assert-Failure {
    Assert-True ($RunStatus -ne 0) "expected failure: $RunOutput"
}

try {
    New-Item -ItemType Directory -Force -Path $TestRoot | Out-Null

    $TestsRun++
    Invoke-Installer -Arguments @('-Help')
    Assert-Success
    Assert-Contains $RunOutput 'Usage:'

    $TestsRun++
    $env:NEMO_RELAY_VERSION = 'not-a-version'
    Invoke-Installer -Arguments @('-InstallDir', $InstallDir)
    Assert-Failure
    Assert-Contains $RunOutput 'unsupported version'

    $TestsRun++
    $env:NEMO_RELAY_VERSION = ''
    Invoke-Installer -Directory $InstallDir -CaptureProcessPath
    Assert-Success
    Assert-Contains $RunOutput ("for $(Get-ExpectedTarget)...")
    Assert-True (Test-Path -LiteralPath (Join-Path $InstallDir 'nemo-relay.exe') -PathType Leaf) 'latest install did not create nemo-relay.exe'
    $latestVersion = (& (Join-Path $InstallDir 'nemo-relay.exe') --version | Out-String)
    Assert-Contains $latestVersion 'nemo-relay '
    Assert-PathContains ([Environment]::GetEnvironmentVariable('Path', 'User')) $InstallDir
    $processPath = ($RunOutput -split '__NEMO_RELAY_PATH__', 2)[1]
    Assert-True (-not [string]::IsNullOrWhiteSpace($processPath)) 'installer did not report its updated process PATH'
    Assert-PathContains $processPath $InstallDir
    Assert-NoTemporaryFiles $InstallDir

    $TestsRun++
    $env:NEMO_RELAY_VERSION = '0.3.0'
    Invoke-Installer -Directory $InstallDir -CaptureProcessPath
    Assert-Success
    $pinnedVersion = (& (Join-Path $InstallDir 'nemo-relay.exe') --version | Out-String)
    Assert-Contains $pinnedVersion 'nemo-relay 0.3.0'
    Assert-NoTemporaryFiles $InstallDir

    $TestsRun++
    $env:NEMO_RELAY_VERSION = '999.999.999'
    Invoke-Installer -Arguments @('-InstallDir', $InstallDir)
    Assert-Failure
    Assert-Contains $RunOutput 'could not download https://github.com/NVIDIA/NeMo-Relay/releases/download/999.999.999/'
    $preservedVersion = (& (Join-Path $InstallDir 'nemo-relay.exe') --version | Out-String)
    Assert-Contains $preservedVersion 'nemo-relay 0.3.0'
    Assert-NoTemporaryFiles $InstallDir

    Write-Output "PASS: $TestsRun PowerShell installer scenarios"
}
finally {
    [Environment]::SetEnvironmentVariable('Path', $OriginalUserPath, 'User')
    $env:Path = $OriginalProcessPath
    $env:NEMO_RELAY_VERSION = $OriginalVersion
    Remove-Item -LiteralPath $TestRoot -Recurse -Force -ErrorAction SilentlyContinue
}
