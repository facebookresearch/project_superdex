# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

param (
    [string]$filename
)

if ((Test-Path (Join-Path (Get-Location) '.hg')) -or (Get-Command 'sl' -ErrorAction SilentlyContinue)) {
    $hash = (sl log -r . --template '{node|short}') 2>$null
    if ($LASTEXITCODE -eq 0 -and $hash) {
        $dirty = if ((sl status) 2>$null) { '-dirty' } else { '' }
        $CommitVersion = "$hash$dirty"
    } else {
        $CommitVersion = 'unknown'
    }
} else {
    $CommitVersion = git describe --tags --long --dirty --always
}

$FileContent = 'using System.Reflection;

[assembly: AssemblyInformationalVersion("{0}")]' -f $CommitVersion
$FileContent | Out-File $filename
