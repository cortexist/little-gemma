# Warm serve-mode prefill benchmark (A5000). Matches the methodology of
# llama.cpp.diagrams/tensorsharp-llama.cpp-benchmark.md: socket serve path,
# N identical turns, first turn discarded (repack/clock warmup), best-of-rest.
# Usage: .\bench-serve.ps1 [-WideChunk 0] [-Turns 6] [-Words 900] [-Bin path]
# $Bin/$CAT default to this repo's Release build (resolved from the script's own
# location). $Model has no in-repo default (the gguf lives in a sibling checkout);
# pass -Model or set LG_BENCH_MODEL.
param(
    [int]$WideChunk = 0,
    [int]$Turns = 6,
    [int]$Words = 900,
    [string]$Bin = "$(Split-Path -Parent $PSScriptRoot)\build\Release\run-cuda-i8.exe",
    [string]$Model = $(if ($env:LG_BENCH_MODEL) { $env:LG_BENCH_MODEL } else { "gemma-4-12B-it-Q4_K_M.gguf" })
)
$ErrorActionPreference = "Continue"
$CAT = "$(Split-Path -Parent $PSScriptRoot)\build\Release\socket_cat.exe"
$SOCK = "$env:TEMP\lg-bench.sock"
$ERRF = "$env:TEMP\lg-bench.err"
$OUTF = "$env:TEMP\lg-bench.out"
$PROMPTF = "$env:TEMP\lg-bench-prompts.txt"

if ($WideChunk -gt 0) { $env:LG_WIDE_CHUNK = "$WideChunk" } else { Remove-Item Env:LG_WIDE_CHUNK -ErrorAction SilentlyContinue }
Remove-Item $SOCK, $ERRF, $OUTF -ErrorAction SilentlyContinue

# One turn per line; long filler + short-answer question so decode stays short.
$line = (("word " * $Words).Trim()) + " Ignore all the words above. Reply with exactly one word: what is the capital of France?"
$lines = @()
for ($i = 0; $i -lt $Turns; $i++) { $lines += $line }
[IO.File]::WriteAllLines($PROMPTF, $lines)

$p = Start-Process -FilePath $Bin -ArgumentList @("-m", $Model, "-s", $SOCK) -RedirectStandardError $ERRF -RedirectStandardOutput "$env:TEMP\lg-bench.stdout" -PassThru -NoNewWindow
try {
    $deadline = (Get-Date).AddSeconds(60)
    while (-not (Test-Path $SOCK) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
    if (-not (Test-Path $SOCK)) { throw "server socket never appeared" }
    Start-Sleep -Milliseconds 500
    # One connection per turn: pos restarts each time (same-shape turns, like
    # llama-bench pp), process stays warm. A single line goes per connection.
    $ONE = "$env:TEMP\lg-bench-one.txt"
    [IO.File]::WriteAllLines($ONE, @($line))
    Remove-Item $OUTF -ErrorAction SilentlyContinue
    for ($i = 0; $i -lt $Turns; $i++) {
        cmd /c "`"$CAT`" `"$SOCK`" < `"$ONE`" >> `"$OUTF`" 2>nul"
    }
} finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 300
}

$errText = [IO.File]::ReadAllText($ERRF)
$stats = @()
foreach ($m in [regex]::Matches($errText, "turn: (\d+) in ([\d.]+)s \(([\d.]+) tok/s\)")) {
    $stats += [pscustomobject]@{ tokens = [int]$m.Groups[1].Value; s = [double]$m.Groups[2].Value; tps = [double]$m.Groups[3].Value }
}
if (-not $stats) { Write-Host "NO TURNS - server stderr:"; Get-Content $ERRF | Select-Object -Last 20; exit 1 }
$warm = $stats | Select-Object -Skip 1
$best = ($warm | Measure-Object -Property tps -Maximum).Maximum
$avg  = [math]::Round(($warm | Measure-Object -Property tps -Average).Average, 1)
Write-Host ("WIDE_CHUNK={0} turns={1} tokens={2} first={3} warm best={4} avg={5} tok/s" -f $WideChunk, $stats.Count, $stats[0].tokens, $stats[0].tps, $best, $avg)
Write-Host ("all: " + (($stats | ForEach-Object { $_.tps }) -join " "))
$reply = (Get-Content $OUTF -Raw -ErrorAction SilentlyContinue)
if ($reply -match "Paris") { Write-Host "cohesion: Paris OK" } else { Write-Host "cohesion: FAIL (no Paris in replies)"; Write-Host ($reply.Substring(0, [Math]::Min(300, $reply.Length))) }
