param()
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectDir
try {
    python -m PyInstaller --noconfirm --clean --workpath build/pyinstaller --distpath dist PyAgent.spec
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed' }
    $exeFile = Join-Path $projectDir 'dist\PyAgent.exe'
    $digest = (Get-FileHash -LiteralPath $exeFile -Algorithm SHA256).Hash.ToLowerInvariant()
    Set-Content -LiteralPath (Join-Path $projectDir 'dist\SHA256.txt') -Value "$digest  PyAgent.exe" -Encoding ASCII
    Write-Host "Executable: $projectDir\dist\PyAgent.exe"
} finally {
    Pop-Location
}
