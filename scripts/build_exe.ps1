param([switch]$InstallDependencies, [switch]$CheckOnly, [switch]$Clean)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectDir
try {
    $pythonExe = $null
    $pythonPrefix = @()
    $candidates = @((Join-Path $projectDir 'build\venv\Scripts\python.exe'),
                    (Join-Path $projectDir '.venv\Scripts\python.exe'), 'python', 'py')
    foreach ($candidate in $candidates) {
        if (-not (Get-Command $candidate -ErrorAction SilentlyContinue)) { continue }
        $candidatePrefix = @()
        if ($candidate -eq 'py') { $candidatePrefix = @('-3') }
        try {
            & $candidate @candidatePrefix -c "import sys,struct; sys.exit(0 if sys.version_info >= (3,10) and struct.calcsize('P') == 8 else 1)" 2>$null
        } catch { continue }
        if ($LASTEXITCODE -eq 0) {
            $pythonExe = $candidate
            $pythonPrefix = $candidatePrefix
            break
        }
    }
    if (-not $pythonExe) { throw 'Install 64-bit Python 3.10 or later, then run build_exe.bat again.' }
    Write-Host "Python: $pythonExe" -ForegroundColor Cyan
    & $pythonExe @pythonPrefix -c "import importlib.util,sys; sys.exit(not all(importlib.util.find_spec(m) is not None for m in ('PyQt5','requests','PyInstaller','PIL')))"
    if ($LASTEXITCODE -ne 0) {
        if (-not $InstallDependencies) { throw 'Missing dependencies. Double-click build_exe.bat to install them automatically.' }
        $buildVenv = Join-Path $projectDir 'build\venv'
        if ($pythonExe -ne (Join-Path $buildVenv 'Scripts\python.exe')) {
            Write-Host 'Creating an isolated build environment...'
            & $pythonExe @pythonPrefix -m venv $buildVenv
            if ($LASTEXITCODE -ne 0) { throw 'Failed to create the build environment.' }
        }
        $pythonExe = Join-Path $buildVenv 'Scripts\python.exe'
        $pythonPrefix = @()
        & $pythonExe -m pip install --disable-pip-version-check -r requirements.txt 'pyinstaller>=6.22.1,<7' 'Pillow>=10'
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check the connection and retry.' }
    }
    if ($CheckOnly) { Write-Host 'Python and build dependencies are ready.' -ForegroundColor Green; exit 0 }
    $env:PYTHONUTF8 = '1'
    $buildFlags = @('--noconfirm', '--workpath', 'build/pyinstaller', '--distpath', 'dist')
    if ($Clean) { $buildFlags += '--clean' }
    & $pythonExe @pythonPrefix -m PyInstaller @buildFlags PyAgent.spec
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed' }
    & $pythonExe @pythonPrefix -c "from pathlib import Path; import hashlib; p=Path('dist/PyAgent.exe'); Path('dist/SHA256.txt').write_text(hashlib.sha256(p.read_bytes()).hexdigest() + '  PyAgent.exe\n',encoding='ascii')"
    if ($LASTEXITCODE -ne 0) { throw 'EXE was built, but writing SHA256.txt failed.' }
    Write-Host "Executable: $projectDir\dist\PyAgent.exe"
} catch {
    Write-Host "Build failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Close any running PyAgent.exe before rebuilding. Existing config and sessions are preserved.'
    exit 1
} finally {
    Pop-Location
}
