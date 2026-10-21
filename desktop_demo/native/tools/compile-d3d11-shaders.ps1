param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [string]$Root = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)),
    [string]$WindowsSdkVersion = "10.0.26100.0",
    [switch]$VerifyDeterminism
)

$ErrorActionPreference = "Stop"
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$Root = [System.IO.Path]::GetFullPath($Root)

function Resolve-Fxc {
    param([string]$RequestedVersion)

    $kitsBin = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
    if ($RequestedVersion) {
        $requested = $RequestedVersion.TrimEnd('\', '/')
        $candidate = Join-Path $kitsBin "$requested\x64\fxc.exe"
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
        throw "The pinned Windows SDK fxc.exe was not found: $candidate"
    }
    $versions = @()
    if (Test-Path $kitsBin) {
        $versions += Get-ChildItem $kitsBin -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^\d+\.\d+\.\d+\.\d+$' } |
            Sort-Object { [version]$_.Name } -Descending |
            ForEach-Object { $_.Name }
    }

    foreach ($version in $versions) {
        $candidate = Join-Path $kitsBin "$version\x64\fxc.exe"
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    throw "Windows SDK fxc.exe was not found. Install the Windows 10/11 SDK x64 shader compiler."
}

function Get-BytesSha256 {
    param([byte[]]$Bytes)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha.ComputeHash($Bytes))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Get-StringSha256 {
    param([string]$Text)
    return Get-BytesSha256 $utf8NoBom.GetBytes($Text)
}

function Write-TextIfChanged {
    param([string]$Path, [string]$Content)
    if ((Test-Path $Path) -and [System.IO.File]::ReadAllText($Path) -ceq $Content) {
        return
    }
    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($Path)) | Out-Null
    [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

function Invoke-FxcShader {
    param(
        [string]$Fxc,
        [string]$Source,
        [string]$Entry,
        [string]$Target,
        [string]$Output,
        [switch]$AllowWarnings
    )
    $arguments = @(
        '/nologo',
        '/Ges',
        '/O3',
        '/Qstrip_debug',
        '/Qstrip_reflect',
        "/E$Entry",
        "/T$Target",
        "/Fo$Output",
        $Source
    )
    if (-not $AllowWarnings) {
        $arguments = @('/WX') + $arguments
    }
    # PowerShell can promote native stderr warnings to terminating
    # NativeCommandError records when the parent build uses Stop. Capture the
    # compiler diagnostics and decide failure solely from fxc's exit code.
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $compilerOutput = & $Fxc @arguments 2>&1
    $compilerExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($compilerExitCode -ne 0) {
        throw "fxc failed for $Entry ($Target):`n$($compilerOutput -join "`n")"
    }
    if (-not (Test-Path $Output)) {
        throw "fxc did not create $Output for $Entry ($Target)."
    }
}

$fxc = Resolve-Fxc $WindowsSdkVersion
$generatedDirectory = Join-Path $Root "obj\$Configuration\generated"
$shaderDirectory = Join-Path $Root "obj\$Configuration\shaders"
[System.IO.Directory]::CreateDirectory($generatedDirectory) | Out-Null
[System.IO.Directory]::CreateDirectory($shaderDirectory) | Out-Null

$presenterSource = Join-Path $Root "src\d3d11_presenter.hlsl"
foreach ($required in @($presenterSource)) {
    if (-not (Test-Path $required)) {
        throw "Required shader source was not found: $required"
    }
}

$entries = @(
    [pscustomobject]@{ Symbol='VsMain'; Entry='VsMain'; Target='vs_5_0'; Source=$presenterSource },
    [pscustomobject]@{ Symbol='PsMain'; Entry='PsMain'; Target='ps_5_0'; Source=$presenterSource },
    [pscustomobject]@{ Symbol='PsWorkingInputMagicMain'; Entry='PsWorkingInputMagicMain'; Target='ps_5_0'; Source=$presenterSource },
    [pscustomobject]@{ Symbol='PsMagicSharpEncodedMain'; Entry='PsMagicSharpEncodedMain'; Target='ps_5_0'; Source=$presenterSource },
    [pscustomobject]@{ Symbol='PsCopySameSizeMain'; Entry='PsCopySameSizeMain'; Target='ps_5_0'; Source=$presenterSource },
    [pscustomobject]@{ Symbol='PsBilinearResampleMain'; Entry='PsBilinearResampleMain'; Target='ps_5_0'; Source=$presenterSource },
    [pscustomobject]@{ Symbol='PsMagicResampleMain'; Entry='PsMagicResampleMain'; Target='ps_5_0'; Source=$presenterSource },
    [pscustomobject]@{ Symbol='PsPresentSdrMain'; Entry='PsPresentSdrMain'; Target='ps_5_0'; Source=$presenterSource },
    [pscustomobject]@{ Symbol='PsPresentDirectSdrMain'; Entry='PsPresentDirectSdrMain'; Target='ps_5_0'; Source=$presenterSource },
    [pscustomobject]@{ Symbol='CsPresentDirectSdrMain'; Entry='CsPresentDirectSdrMain'; Target='cs_5_0'; Source=$presenterSource },
    [pscustomobject]@{ Symbol='CsPrepareWorkingInput'; Entry='CsPrepareWorkingInput'; Target='cs_5_0'; Source=$presenterSource }
)

$compiled = @()
foreach ($shader in $entries) {
    $output = Join-Path $shaderDirectory "$($shader.Symbol).cso"
    Invoke-FxcShader $fxc $shader.Source $shader.Entry $shader.Target $output
    $bytes = [System.IO.File]::ReadAllBytes($output)
    $hash = Get-BytesSha256 $bytes
    if ($VerifyDeterminism) {
        $verificationOutput = Join-Path $shaderDirectory "$($shader.Symbol).verify.cso"
        Invoke-FxcShader $fxc $shader.Source $shader.Entry $shader.Target $verificationOutput
        $verificationBytes = [System.IO.File]::ReadAllBytes($verificationOutput)
        $verificationHash = Get-BytesSha256 $verificationBytes
        Remove-Item -LiteralPath $verificationOutput -Force
        if ($verificationHash -cne $hash) {
            throw "Non-deterministic fxc output for $($shader.Symbol): $hash != $verificationHash"
        }
    }
    $sourceFullPath = [System.IO.Path]::GetFullPath($shader.Source)
    $rootPrefix = $Root.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    $relativeSource = if ($sourceFullPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        $sourceFullPath.Substring($rootPrefix.Length).Replace('\', '/')
    }
    else {
        $sourceFullPath.Replace('\', '/')
    }
    $compiled += [pscustomobject]@{
        Symbol = $shader.Symbol
        Entry = $shader.Entry
        Target = $shader.Target
        Source = $relativeSource
        SourceSha256 = Get-BytesSha256 ([System.IO.File]::ReadAllBytes($shader.Source))
        Bytes = $bytes
        Sha256 = $hash
    }
}

$sourceSetDescription = ($compiled | ForEach-Object {
    "$($_.Symbol)|$($_.Entry)|$($_.Target)|$($_.Source)|$($_.SourceSha256)|$($_.Sha256)"
}) -join "`n"
$sourceSetHash = Get-StringSha256 $sourceSetDescription
$fxcVersion = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($fxc).FileVersion

$header = @"
#pragma once

#include <cstddef>
#include <cstdint>

namespace aess::generated::d3d11
{
    struct ShaderBytecode
    {
        const std::uint8_t* data;
        std::size_t size;
        const char* sha256;
    };

    inline constexpr char kSourceSetSha256[] = "$sourceSetHash";
"@
foreach ($shader in $compiled) {
    $header += "    extern const ShaderBytecode k$($shader.Symbol);`n"
}
$header += "}`n"

$implementation = @"
// Generated by tools/compile-d3d11-shaders.ps1. Do not edit.
#include "AESSProductionShaders.h"

namespace aess::generated::d3d11
{
    namespace
    {
"@
foreach ($shader in $compiled) {
    $implementation += "        alignas(16) constexpr std::uint8_t k$($shader.Symbol)Bytes[] = {`n"
    for ($offset = 0; $offset -lt $shader.Bytes.Length; $offset += 12) {
        $count = [System.Math]::Min(12, $shader.Bytes.Length - $offset)
        $values = for ($index = 0; $index -lt $count; ++$index) {
            '0x{0:x2}' -f $shader.Bytes[$offset + $index]
        }
        $implementation += "            $($values -join ', '),`n"
    }
    $implementation += "        };`n"
}
$implementation += "    }`n`n"
foreach ($shader in $compiled) {
    $implementation += "    const ShaderBytecode k$($shader.Symbol){ k$($shader.Symbol)Bytes, sizeof(k$($shader.Symbol)Bytes), `"$($shader.Sha256)`" };`n"
}
$implementation += "}`n"

$headerPath = Join-Path $generatedDirectory "AESSProductionShaders.h"
$implementationPath = Join-Path $generatedDirectory "AESSProductionShaders.cpp"
Write-TextIfChanged $headerPath $header
Write-TextIfChanged $implementationPath $implementation

$manifest = [ordered]@{
    schemaVersion = 1
    compiler = [ordered]@{ path = $fxc; version = $fxcVersion; flags = '/Ges /O3 /WX /Qstrip_debug /Qstrip_reflect' }
    sourceSetSha256 = $sourceSetHash
    shaders = @($compiled | ForEach-Object {
        [ordered]@{
            symbol = $_.Symbol
            entry = $_.Entry
            target = $_.Target
            source = $_.Source
            sourceSha256 = $_.SourceSha256
            bytecodeBytes = $_.Bytes.Length
            bytecodeSha256 = $_.Sha256
        }
    })
}
$manifestText = ($manifest | ConvertTo-Json -Depth 5) + "`n"
Write-TextIfChanged (Join-Path $generatedDirectory "AESSProductionShaders.json") $manifestText


Write-Host "Generated production presentation shaders only."
