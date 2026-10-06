# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Keep native lifecycle tests outside restrictive jobs created by the CI host.
# The controller owns a cleanup job that permits explicit worker breakaway.
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$PayloadPath,
    [switch]$Worker,
    [Parameter(Position = 0)]
    [string]$Executable,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CommandArguments
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

# Copy native pipes directly because PowerShell file redirection buffers output.
function Invoke-TestExecutable([hashtable]$Payload) {
    $start = [Diagnostics.ProcessStartInfo]::new($Payload.Executable)
    $start.UseShellExecute = $false
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.WorkingDirectory = $Payload.Directory
    foreach ($argument in $Payload.Arguments) { $start.ArgumentList.Add($argument) }
    $process = $null
    $stdout = $null
    $stderr = $null
    try {
        # Disable file buffering so the controller can read each write immediately.
        $stdout = [IO.FileStream]::new($Payload.Stdout, 'Create', 'Write', 'ReadWrite', 1, $true)
        $stderr = [IO.FileStream]::new($Payload.Stderr, 'Create', 'Write', 'ReadWrite', 1, $true)
        $process = [Diagnostics.Process]::Start($start)
        $copyOut = $process.StandardOutput.BaseStream.CopyToAsync($stdout)
        $copyErr = $process.StandardError.BaseStream.CopyToAsync($stderr)
        $process.WaitForExit()
        $null = $copyOut.GetAwaiter().GetResult()
        $null = $copyErr.GetAwaiter().GetResult()
        return $process.ExitCode
    } finally {
        if ($null -ne $process) { $process.Dispose() }
        if ($null -ne $stdout) { $stdout.Dispose() }
        if ($null -ne $stderr) { $stderr.Dispose() }
    }
}

Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

public sealed class RelayTestJob : IDisposable {
    [StructLayout(LayoutKind.Sequential)]
    struct BasicLimits {
        public long ProcessTime, JobTime;
        public uint Flags;
        public UIntPtr MinimumWorkingSet, MaximumWorkingSet;
        public uint ActiveProcesses;
        public UIntPtr Affinity;
        public uint Priority, Scheduling;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct IoCounters {
        public ulong ReadOperations, WriteOperations, OtherOperations;
        public ulong ReadBytes, WriteBytes, OtherBytes;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct ExtendedLimits {
        public BasicLimits Basic;
        public IoCounters Io;
        public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern IntPtr CreateJobObjectW(IntPtr attributes, string name);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern IntPtr OpenJobObjectW(uint access, bool inherit, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetInformationJobObject(IntPtr job, int kind, ref ExtendedLimits limits, uint size);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool inJob);
    [DllImport("kernel32.dll")]
    static extern IntPtr GetCurrentProcess();
    [DllImport("kernel32.dll")]
    static extern bool CloseHandle(IntPtr handle);

    IntPtr handle;

    public RelayTestJob(string name) {
        handle = CreateJobObjectW(IntPtr.Zero, name);
        if (handle == IntPtr.Zero) throw new Win32Exception();
        var limits = new ExtendedLimits();
        // Kill ordinary descendants when the controller exits. Detachment is explicit.
        limits.Basic.Flags = 0x2000 | 0x0800;
        if (!SetInformationJobObject(handle, 9, ref limits, (uint)Marshal.SizeOf<ExtendedLimits>())) {
            var error = new Win32Exception();
            Dispose();
            throw error;
        }
    }

    public static void Join(string name) {
        bool inJob;
        if (!IsProcessInJob(GetCurrentProcess(), IntPtr.Zero, out inJob)) throw new Win32Exception();
        if (inJob) throw new InvalidOperationException("The Windows test bootstrap is still in a host job.");
        var job = OpenJobObjectW(1, false, name);
        if (job == IntPtr.Zero) throw new Win32Exception();
        try {
            if (!AssignProcessToJobObject(job, GetCurrentProcess())) throw new Win32Exception();
        } finally {
            CloseHandle(job);
        }
    }

    public void Dispose() {
        if (handle != IntPtr.Zero) {
            CloseHandle(handle);
            handle = IntPtr.Zero;
        }
    }
}
'@

if ($Worker) {
    $payload = Get-Content -Raw -LiteralPath $PayloadPath | ConvertFrom-Json -AsHashtable
    $exitCode = 1
    try {
        [RelayTestJob]::Join($payload.JobName)
        [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
        $OutputEncoding = [Console]::OutputEncoding
        foreach ($name in $payload.Environment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $payload.Environment[$name])
        }
        $exitCode = Invoke-TestExecutable $payload
    } catch {
        $_ | Out-String | Add-Content -LiteralPath $payload.Stderr
    } finally {
        [IO.File]::WriteAllText($payload.Result, [string]$exitCode)
    }
    exit $exitCode
}

if (!$Executable) {
    throw "A test executable is required."
}

# Keep each decoder between polls so split UTF-8 characters remain intact.
function Write-NewTestOutput([hashtable]$Log) {
    if ($null -eq $Log.Stream) {
        if (!(Test-Path -LiteralPath $Log.Path)) { return }
        $Log.Stream = [IO.File]::Open($Log.Path, 'Open', 'Read', 'ReadWrite')
        $Log.Decoder = [Text.UTF8Encoding]::new($false).GetDecoder()
    }
    $bytes = [byte[]]::new(4096)
    $characters = [char[]]::new([Text.Encoding]::UTF8.GetMaxCharCount($bytes.Length))
    while (($count = $Log.Stream.Read($bytes, 0, $bytes.Length)) -gt 0) {
        $written = $Log.Decoder.GetChars($bytes, 0, $count, $characters, 0, $false)
        $Log.Writer.Write($characters, 0, $written)
    }
    $Log.Writer.Flush()
}

$directory = Join-Path $env:RUNNER_TEMP ("relay-native-tests-" + [Guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $directory
$job = $null
$exitCode = 1
$logs = @(
    @{ Path = (Join-Path $directory "stdout.log"); Writer = [Console]::Out; Stream = $null },
    @{ Path = (Join-Path $directory "stderr.log"); Writer = [Console]::Error; Stream = $null }
)
try {
    $jobName = "Local\RelayNativeTests-" + [Guid]::NewGuid()
    $job = [RelayTestJob]::new($jobName)
    $environment = @{}
    # Pass build and test settings without copying CI service credentials.
    $names = @(
        "PATH", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP",
        "SystemRoot", "SystemDrive", "COMSPEC", "CI", "GITHUB_ACTIONS",
        "RUNNER_OS", "RUNNER_ARCH", "RUNNER_TEMP", "GITHUB_WORKSPACE",
        "CARGO_HOME", "CARGO_BUILD_TARGET", "CARGO_TARGET_DIR", "CARGO_TERM_COLOR",
        "CARGO_ENCODED_RUSTFLAGS", "CARGO_INCREMENTAL", "CARGO_UNSTABLE_SPARSE_REGISTRY",
        "CARGO_REGISTRIES_CRATES_IO_PROTOCOL", "RUSTUP_HOME", "RUSTUP_TOOLCHAIN",
        "RUSTFLAGS", "RUSTDOCFLAGS", "RUST_BACKTRACE", "LLVM_PROFILE_FILE",
        "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_PYTHON_DOWNLOADS",
        "UV_PYTHON_PREFERENCE", "UV_PYTHON", "pythonLocation", "Python_ROOT_DIR",
        "NEMO_RELAY_CI_WORKSPACE", "NEMO_RELAY_CI_WORKSPACE_TMP",
        "NEMO_RELAY_RUN_REDIS_TESTS", "NEMO_RELAY_RUN_S3_TESTS"
    )
    foreach ($entry in [Environment]::GetEnvironmentVariables().GetEnumerator()) {
        if ($entry.Key -in $names -or $entry.Key -like "CARGO_PROFILE_*") {
            $environment[$entry.Key] = $entry.Value
        }
    }
    $payload = @{
        JobName = $jobName
        Directory = (Get-Location).Path
        Executable = (Get-Command $Executable -CommandType Application).Source
        Arguments = @($CommandArguments)
        Environment = $environment
        Stdout = (Join-Path $directory "stdout.log")
        Stderr = (Join-Path $directory "stderr.log")
        Result = (Join-Path $directory "exit-code")
    }
    $payloadPath = Join-Path $directory "payload.json"
    $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $payloadPath
    $command = '"{0}" -NoProfile -NonInteractive -File "{1}" -Worker -PayloadPath "{2}"' -f (
        (Join-Path $PSHOME "pwsh.exe"), $PSCommandPath, $payloadPath
    )
    # WMI starts the bootstrap outside the calling process's job hierarchy.
    $created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $command }
    if ($created.ReturnValue -ne 0) {
        throw "Windows test bootstrap failed with status $($created.ReturnValue)."
    }
    $process = [Diagnostics.Process]::GetProcessById($created.ProcessId)
    try {
        while (!$process.WaitForExit(1000)) {
            foreach ($log in $logs) { Write-NewTestOutput $log }
        }
    } finally {
        $process.Dispose()
    }
    if (!(Test-Path -LiteralPath $payload.Result)) {
        throw "Windows test bootstrap exited without a test result."
    }
    $exitCode = [int](Get-Content -Raw -LiteralPath $payload.Result)
} finally {
    if ($null -ne $job) {
        $job.Dispose()
    }
    try {
        foreach ($log in $logs) { Write-NewTestOutput $log }
    } finally {
        foreach ($log in $logs) {
            if ($null -ne $log.Stream) { $log.Stream.Dispose() }
        }
        Remove-Item -Recurse -Force -LiteralPath $directory
    }
}
exit $exitCode
