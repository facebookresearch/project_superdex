# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Builds the MSI installer for the SolidWorks add-in.
# Requires the .NET SDK; the WiX v6 toolset is restored automatically via NuGet.

param (
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WixProject = Join-Path $ProjectRoot "installer\SuperDexCadExporter.wixproj"

function Invoke-Native {
    # Local $ErrorActionPreference keeps native stderr from becoming a terminating
    # error, so callers can branch on the exit code instead.
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @()
    )

    $ErrorActionPreference = 'Continue'
    $PSNativeCommandUseErrorActionPreference = $false

    $output = & $FilePath @Arguments 2>$null
    [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Output   = ($output | Out-String).Trim()
    }
}

Push-Location $ProjectRoot
try {
    $CommitVersion = 'unknown'

    if (Get-Command 'hg' -ErrorAction SilentlyContinue) {
        $log = Invoke-Native 'hg' @('log', '-r', '.', '--template', '{node|short}')
        if ($log.ExitCode -eq 0 -and $log.Output) {
            $status = Invoke-Native 'hg' @('status')
            $dirty = if ($status.ExitCode -eq 0 -and $status.Output) { '-dirty' } else { '' }
            $CommitVersion = "$($log.Output)$dirty"
        }
    }

    if ($CommitVersion -eq 'unknown' -and (Get-Command 'git' -ErrorAction SilentlyContinue)) {
        $describe = Invoke-Native 'git' @('describe', '--tags', '--long', '--dirty', '--always')
        if ($describe.ExitCode -eq 0 -and $describe.Output) {
            $CommitVersion = $describe.Output
        }
    }
} finally {
    Pop-Location
}

Write-Host "Building installer (commit $CommitVersion)..."
dotnet clean $WixProject; dotnet build $WixProject -c $Configuration -p:CommitVersion=$CommitVersion -t:Rebuild

if ($LASTEXITCODE -ne 0) {
    throw "Installer build failed."
}

Write-Host "Installer written to $(Join-Path $ProjectRoot 'installer\output')"
