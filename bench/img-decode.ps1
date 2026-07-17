# img-decode.ps1 - does an image in context change the DECODE RATE?
# Text control and image turn in the SAME server session (identical weights,
# clocks, thermals). An image span is just KV rows, so the prediction is: no
# change beyond the depth effect, which is ~flat since the 2026-07-17 KV-split
# fix. The image's real cost is TTFT (encoder + span prefill), not decode rate.
# Keep this file pure ASCII (PS 5.1 reads a BOM-less .ps1 as ANSI and mangles
# UTF-8 punctuation into a parse error).
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Mmproj,
    [Parameter(Mandatory=$true)][string]$Image,
    [Parameter(Mandatory=$true)][string]$Tag,
    [string]$Bin   = "$(Split-Path -Parent $PSScriptRoot)\build\Release\run-cuda-i8.exe",
    [string]$Cat   = "$(Split-Path -Parent $PSScriptRoot)\build\Release\socket_cat.exe",
    [string]$Mmcat = "C:\Users\Zero\Cortexist\little-gemma-tools\build\Release\mmcat.exe"
)
$ErrorActionPreference = "Continue"
$OUT = "$env:TEMP\settle"; New-Item -ItemType Directory -Force $OUT | Out-Null
$SOCK = "$env:TEMP\lg-img.sock"; $ERRF = "$OUT\img-$Tag.err"
Remove-Item $SOCK, $ERRF -ErrorAction SilentlyContinue

$Q = "Describe this in detail, covering every object, colour, texture and spatial relationship you can see, and then explain what is probably happening."
$QF = "$OUT\img-q.txt"; [IO.File]::WriteAllLines($QF, @($Q))

$p = Start-Process -FilePath $Bin -ArgumentList @("-m",$Model,"-mm",$Mmproj,"-s",$SOCK) -RedirectStandardError $ERRF -RedirectStandardOutput "$OUT\img-$Tag.stdout" -PassThru -NoNewWindow
try {
    $deadline = (Get-Date).AddSeconds(180)
    while (-not (Test-Path $SOCK) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
    if (-not (Test-Path $SOCK)) { throw "server socket never appeared" }
    Start-Sleep -Milliseconds 500
    Write-Host "### TEXT control (no image)"
    for ($i = 0; $i -lt 3; $i++) { cmd /c "`"$Cat`" `"$SOCK`" < `"$QF`" > `"$OUT\img-$Tag-text$i.out`" 2>nul" }
    Write-Host "### IMAGE turns (span + same question)"
    for ($i = 0; $i -lt 3; $i++) { & $Mmcat $SOCK $Image $Q > "$OUT\img-$Tag-img$i.out" 2>"$OUT\img-$Tag-img$i.err" }
} finally {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; Start-Sleep -Milliseconds 300
}
Write-Host "=== turn lines: first 3 = text control, last 3 = image (discard each first) ==="
Select-String -Path $ERRF -Pattern "turn:" | ForEach-Object { $_.Line }
Write-Host "=== mmcat frame report ==="
Get-Content "$OUT\img-$Tag-img2.err" -ErrorAction SilentlyContinue | Select-Object -First 2
Write-Host "=== image reply sanity (did it see the picture?) ==="
$r = Get-Content "$OUT\img-$Tag-img2.out" -Raw -ErrorAction SilentlyContinue
if ($r) { $r.Substring(0, [Math]::Min(240, $r.Length)) }
