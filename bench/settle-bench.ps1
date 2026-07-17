# settle-bench.ps1 — canonical decode+prefill bench (Windows/A5000), one model per call.
# Produces the numbers in docs/benchmarks.md. Serve mode: 6x 929-token prefill
# turns + 5x max-length decode turns, one connection each, first of each
# discarded; rates from the per-turn `turn:` stderr stat.
# Usage: .\settle-bench.ps1 -Model <path-to.gguf> -Tag e4b
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Tag,
    [string]$Bin = "$(Split-Path -Parent $PSScriptRoot)\build\Release\run-cuda-i8.exe",
    [string]$Cat = "$(Split-Path -Parent $PSScriptRoot)\build\Release\socket_cat.exe"
)
$ErrorActionPreference = "Continue"
$PF   = "$PSScriptRoot\line929s.txt"
$OUT  = "$env:TEMP\settle"
New-Item -ItemType Directory -Force $OUT | Out-Null
$SOCK = "$env:TEMP\lg-settle.sock"
$ERRF = "$OUT\lg-$Tag.err"
Remove-Item $SOCK, $ERRF -ErrorAction SilentlyContinue

$DECF = "$OUT\decode-q.txt"
[IO.File]::WriteAllLines($DECF, @("Explain in detail how a refrigerator works, covering the compressor, the refrigerant cycle, and why the inside gets cold while the back gets warm."))

$p = Start-Process -FilePath $Bin -ArgumentList @("-m", $Model, "-s", $SOCK) -RedirectStandardError $ERRF -RedirectStandardOutput "$OUT\lg-$Tag.stdout" -PassThru -NoNewWindow
try {
    $deadline = (Get-Date).AddSeconds(120)
    while (-not (Test-Path $SOCK) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
    if (-not (Test-Path $SOCK)) { throw "server socket never appeared" }
    Start-Sleep -Milliseconds 500
    for ($i = 0; $i -lt 6; $i++) { cmd /c "`"$Cat`" `"$SOCK`" < `"$PF`" > `"$OUT\lg-$Tag-pf$i.out`" 2>nul" }
    for ($i = 0; $i -lt 5; $i++) { cmd /c "`"$Cat`" `"$SOCK`" < `"$DECF`" > `"$OUT\lg-$Tag-dec$i.out`" 2>nul" }
} finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 300
}
Write-Host "=== lg-$Tag turn lines (6 prefill then 5 decode; discard first of each) ==="
Select-String -Path $ERRF -Pattern "turn:" | ForEach-Object { $_.Line }
