# mtp-accept.ps1 - MTP acceptance tripwire: serve mode, N prose turns, first discarded.
# Acceptance is a model+prompt property (hardware-independent), so the same prompt
# must give the same % on any device/build; a collapse means the verify==decode
# invariant broke (this is the gate that caught the f16-ring draft bug).
# NOTE: keep this file pure ASCII - PS 5.1 reads a BOM-less .ps1 as ANSI and
# mangles UTF-8 punctuation into a parse error.
# Usage: .\mtp-accept.ps1 -Model <target.gguf> -Head <assistant.gguf> -Tag e4b
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Head,
    [Parameter(Mandatory=$true)][string]$Tag,
    [string]$Bin = "$(Split-Path -Parent $PSScriptRoot)\build\Release\run-cuda-i8.exe",
    [string]$Cat = "$(Split-Path -Parent $PSScriptRoot)\build\Release\socket_cat.exe"
)
$ErrorActionPreference = "Continue"
$OUT = "$env:TEMP\settle"; New-Item -ItemType Directory -Force $OUT | Out-Null
$SOCK = "$env:TEMP\lg-mtp.sock"; $ERRF = "$OUT\mtp-$Tag.err"
Remove-Item $SOCK, $ERRF -ErrorAction SilentlyContinue
$Q = "$OUT\mtp-q.txt"
[IO.File]::WriteAllLines($Q, @("Explain in detail how a refrigerator works, covering the compressor, the refrigerant cycle, and why the inside gets cold while the back gets warm."))

$p = Start-Process -FilePath $Bin -ArgumentList @("-m",$Model,"-mtp",$Head,"-s",$SOCK) -RedirectStandardError $ERRF -RedirectStandardOutput "$OUT\mtp-$Tag.stdout" -PassThru -NoNewWindow
try {
    $deadline = (Get-Date).AddSeconds(180)
    while (-not (Test-Path $SOCK) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
    if (-not (Test-Path $SOCK)) { throw "server socket never appeared" }
    Start-Sleep -Milliseconds 500
    for ($i = 0; $i -lt 4; $i++) { cmd /c "`"$Cat`" `"$SOCK`" < `"$Q`" > `"$OUT\mtp-$Tag-$i.out`" 2>nul" }
} finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; Start-Sleep -Milliseconds 300
}
Write-Host "=== mtp-$Tag acceptance (discard turn 1 - draft graph warmup) ==="
Select-String -Path $ERRF -Pattern "accepted" | ForEach-Object { $_.Line }
Write-Host "=== mtp-$Tag decode tok/s ==="
Select-String -Path $ERRF -Pattern "turn:" | ForEach-Object { $_.Line }
