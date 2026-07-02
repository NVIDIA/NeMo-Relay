# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

[CmdletBinding()]
param(
    [string]$InstallDir,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'

$Repository = 'NVIDIA/NeMo-Relay'
$GitHubUrl = "https://github.com/$Repository"
$GitHubApiUrl = "https://api.github.com/repos/$Repository"

function Show-Usage {
    @'
Install the NeMo Relay CLI from GitHub Releases.

Usage:
  irm https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/main/install.ps1 | iex
  .\install.ps1 [-InstallDir DIR] [-Help]

Environment:
  NEMO_RELAY_VERSION   Release to install, for example 0.5.0 or v0.5.0.
                       Defaults to the latest stable release.

Options:
  -InstallDir DIR      Destination directory (default: %LOCALAPPDATA%\nemo-relay\bin).
  -Help                Show this help text.

Examples:
  irm https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/main/install.ps1 | iex
  $env:NEMO_RELAY_VERSION = '0.5.0'; irm https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/main/install.ps1 | iex
  .\install.ps1 -InstallDir "$HOME\bin"
'@ | Write-Output
}

function Fail([string]$Message) {
    throw "nemo-relay installer: $Message"
}

function Get-ReleaseVersion {
    $version = $env:NEMO_RELAY_VERSION
    if ([string]::IsNullOrWhiteSpace($version)) {
        Write-Host 'Finding the latest stable NeMo Relay release...'
        try {
            $release = Invoke-RestMethod -Uri "$GitHubApiUrl/releases/latest" -Headers @{
                Accept = 'application/vnd.github+json'
                'User-Agent' = 'nemo-relay-install-script'
            } -TimeoutSec 300
        }
        catch {
            Fail 'could not resolve the latest stable release'
        }
        $version = $release.tag_name
        if ([string]::IsNullOrWhiteSpace($version)) {
            Fail 'latest release response did not contain a tag name'
        }
    }

    $version = $version -replace '^v', ''
    if ($version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+(-(alpha|beta|rc)\.[0-9]+)?$') {
        Fail "unsupported version '$version'; expected 0.5.0 or a prerelease such as 0.5.0-rc.1"
    }
    return $version
}

function Get-Target {
    $architecture = $env:PROCESSOR_ARCHITEW6432
    if ([string]::IsNullOrWhiteSpace($architecture)) {
        $architecture = $env:PROCESSOR_ARCHITECTURE
    }
    if ([string]::IsNullOrWhiteSpace($architecture)) {
        Fail "unsupported Windows architecture '$architecture'. Supported platforms: Windows x86_64 and Windows ARM64. For other platforms, use 'cargo install nemo-relay-cli'."
    }

    switch ($architecture.ToUpperInvariant()) {
        'AMD64' { return 'x86_64-pc-windows-msvc' }
        'X86_64' { return 'x86_64-pc-windows-msvc' }
        'ARM64' { return 'aarch64-pc-windows-msvc' }
        default { Fail "unsupported Windows architecture '$architecture'. Supported platforms: Windows x86_64 and Windows ARM64. For other platforms, use 'cargo install nemo-relay-cli'." }
    }
}

function Download-File([string]$Uri, [string]$Path) {
    try {
        Invoke-WebRequest -Uri $Uri -OutFile $Path -UseBasicParsing -TimeoutSec 300
    }
    catch {
        Fail "could not download $Uri"
    }
}

function Add-ToPath([string]$Value, [string]$Directory) {
    foreach ($entry in ($Value -split ';')) {
        if ($entry -and [string]::Equals($entry.Trim().TrimEnd('\'), $Directory, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $Value
        }
    }
    if ([string]::IsNullOrEmpty($Value)) {
        return $Directory
    }
    return "$Value;$Directory"
}

function Add-InstallDirectoryToPath([string]$Directory) {
    $directory = [System.IO.Path]::GetFullPath($Directory).TrimEnd('\')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $updatedUserPath = Add-ToPath $userPath $directory
    if ($updatedUserPath -ne $userPath) {
        [Environment]::SetEnvironmentVariable('Path', $updatedUserPath, 'User')
    }
    $env:Path = Add-ToPath $env:Path $directory
}

if ($Help) {
    Show-Usage
    exit 0
}

try {
    if ([string]::IsNullOrWhiteSpace($InstallDir)) {
        if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
            Fail 'LOCALAPPDATA must be set to choose the default install directory'
        }
        $InstallDir = Join-Path $env:LOCALAPPDATA 'nemo-relay\bin'
    }
    if ([string]::IsNullOrWhiteSpace($InstallDir)) {
        Fail 'install directory must not be empty'
    }

    $version = Get-ReleaseVersion
    $target = Get-Target
    $asset = "nemo-relay-cli-$target-$version.exe"
    $assetUrl = "$GitHubUrl/releases/download/$version/$asset"
    $checksumUrl = "$assetUrl.sha256"

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    if (-not (Test-Path -LiteralPath $InstallDir -PathType Container)) {
        Fail "install path is not a directory: $InstallDir"
    }
    $InstallDir = [System.IO.Path]::GetFullPath($InstallDir)
    $downloadFile = Join-Path $InstallDir ".nemo-relay.download.$([guid]::NewGuid().ToString('N'))"
    $checksumFile = Join-Path $InstallDir ".nemo-relay.checksum.$([guid]::NewGuid().ToString('N'))"
    $backupFile = Join-Path $InstallDir ".nemo-relay.backup.$([guid]::NewGuid().ToString('N'))"
    $destination = Join-Path $InstallDir 'nemo-relay.exe'

    try {
        Write-Output "Downloading NeMo Relay CLI $version for $target..."
        Download-File $assetUrl $downloadFile
        Download-File $checksumUrl $checksumFile

        $expectedChecksum = ((Get-Content -LiteralPath $checksumFile -TotalCount 1).Trim() -split '\s+')[0].ToLowerInvariant()
        if ($expectedChecksum -notmatch '^[0-9a-f]{64}$') {
            Fail "invalid checksum file for $asset"
        }
        $actualChecksum = (Get-FileHash -LiteralPath $downloadFile -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualChecksum -ne $expectedChecksum) {
            Fail "checksum verification failed for $asset"
        }

        if (Test-Path -LiteralPath $destination) {
            [System.IO.File]::Replace($downloadFile, $destination, $backupFile)
        }
        else {
            [System.IO.File]::Move($downloadFile, $destination)
        }
    }
    finally {
        Remove-Item -LiteralPath $downloadFile, $checksumFile, $backupFile -Force -ErrorAction SilentlyContinue
    }

    Add-InstallDirectoryToPath $InstallDir
    Write-Output "Installed NeMo Relay CLI $version to $destination"
    Write-Output "Added $InstallDir to the Windows user PATH. Newly opened shells inherit this change."
}
catch {
    Write-Error $_
    exit 1
}
