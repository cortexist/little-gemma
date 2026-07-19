param($Model, $Head, $Label, $Prompt)
$exe  = "C:\Users\Zero\Cortexist\little-gemma\build\Release\run-cuda-i8.exe"
$scat = "C:\Users\Zero\Cortexist\little-gemma\build\Release\socket_cat.exe"
$sock = "C:\Users\Zero\lgb.sock"
$log  = "C:\Users\Zero\lgb.log"
$nul  = "C:\Users\Zero\lgb.out"
Remove-Item $sock,$log,$nul -ErrorAction SilentlyContinue
$a = @("-m",$Model,"-mtp",$Head,"-s",$sock)
$srv = Start-Process -FilePath $exe -ArgumentList $a -RedirectStandardError $log -RedirectStandardOutput $nul -PassThru -NoNewWindow
for ($i=0; $i -lt 180; $i++) {
  if ((Test-Path $log) -and (Select-String -Path $log -Pattern "listening on" -Quiet)) { break }
  Start-Sleep -Seconds 1
}
Start-Sleep -Seconds 2
$Prompt | & $scat $sock | Out-Null   # turn 1: warms MTP draft graph
$Prompt | & $scat $sock | Out-Null   # turn 2: measured
Start-Sleep -Seconds 1
Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 500
Write-Output "===== $Label ====="
Select-String -Path $log -Pattern "turn:|mtp:" | Select-Object -Last 2 | ForEach-Object { $_.Line }
