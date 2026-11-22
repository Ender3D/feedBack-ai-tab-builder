$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Activate local Python 3.12 CUDA .venv if not already active
if (-not $env:VIRTUAL_ENV -and (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    . ".\.venv\Scripts\Activate.ps1"
}

$v = python -c "import ai_tab_gui; print(ai_tab_gui.APP_VERSION)"
Write-Host "Building FeedBack-AI-Tab-Builder v$v..." -ForegroundColor Cyan

# Generate Windows .exe version metadata dynamically
$verArgs = @()
if (Get-Command create-version-file -ErrorAction SilentlyContinue) {
    @"
Version: $v.0
CompanyName: Ender3D
FileDescription: fee[dB]ack AI Audio-to-Tab Builder
InternalName: FeedBack-AI-Tab-Builder
OriginalFilename: FeedBack-AI-Tab-Builder-v$v.exe
ProductName: FeedBack AI Audio-to-Tab Builder
"@ | Set-Content -Encoding UTF8 "version.yml"
    create-version-file version.yml --outfile file_version_info.txt
    $verArgs = @("--version-file", "file_version_info.txt")
}

# Clean previous build outputs
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

pyinstaller --noconfirm --onedir --windowed `
    @verArgs `
    --name "FeedBack-AI-Tab-Builder-v$v" `
    --paths feedback_lib `
    --add-data "feedback_lib;feedback_lib" `
    --collect-all demucs `
    --collect-all basic_pitch `
    --collect-all librosa `
    --collect-all onnxruntime `
    --hidden-import song `
    --hidden-import yaml `
    ai_tab_gui.py

# Automatically pack release .7z archive if 7-Zip is installed
$7z = if (Get-Command 7z -ErrorAction SilentlyContinue) { "7z" } elseif (Test-Path "C:\Program Files\7-Zip\7z.exe") { "C:\Program Files\7-Zip\7z.exe" } else { $null }
if ($7z) {
    Write-Host "Packing release archive with 7-Zip..." -ForegroundColor Cyan
    & $7z a -t7z -mx=5 "dist\FeedBack-AI-Tab-Builder-v$v.7z" ".\dist\FeedBack-AI-Tab-Builder-v$v\*" | Out-Null
    Write-Host "`n[Done] Release archive ready: dist\FeedBack-AI-Tab-Builder-v$v.7z" -ForegroundColor Green
} else {
    Write-Host "`n[Done] Output folder ready: dist\FeedBack-AI-Tab-Builder-v$v\" -ForegroundColor Green
}
