# Сжатие виртуального диска Docker. ЗАПУСКАТЬ ОТ АДМИНИСТРАТОРА.
#     правой кнопкой по файлу -> "Выполнить с помощью PowerShell" из админской консоли,
#     либо в админском PowerShell:  powershell -ExecutionPolicy Bypass -File journal\compact_docker_disk.ps1
#
# ЗАЧЕМ. Docker Desktop держит образы и кеш сборки внутри одного файла-диска
# %LOCALAPPDATA%\Docker\wsl\disk\docker_data.vhdx. Когда внутри него что-то удаляют, файл
# НЕ уменьшается сам: Windows продолжает видеть его прежнего размера. 22.09 из него вычистили
# 28.6 ГБ кеша сборки и два мёртвых образа, но на диске C: это не отразилось никак.
# Сжатие возвращает освободившееся место системе.
#
# ПОЧЕМУ НУЖНЫ ПРАВА. diskpart подключает и сжимает виртуальный диск, а это операция уровня
# системы. Без прав администратора команда падает с "The requested operation requires elevation".
#
# ЧТО ЭТО ЛОМАЕТ. Ничего. Образы и контейнеры остаются на месте, удаляется только пустота
# внутри файла. Docker Desktop нужно будет запустить заново - он сам поднимет свою WSL-машину.
#
# ЧЕГО ЭТО НЕ ДЕЛАЕТ. Не трогает ваши дистрибутивы WSL (kali-linux, Ubuntu) и ничего не удаляет.

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$admin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Host "Нужны права администратора: запустите PowerShell от имени администратора и повторите." -ForegroundColor Yellow
    exit 1
}

$vhd = Join-Path $env:LOCALAPPDATA "Docker\wsl\disk\docker_data.vhdx"
if (-not (Test-Path $vhd)) {
    Write-Host "Не нашёл $vhd - возможно, у вас другая версия Docker Desktop." -ForegroundColor Yellow
    exit 1
}

$before = (Get-Item $vhd).Length / 1GB
$freeBefore = (Get-PSDrive C).Free / 1GB
Write-Host ("Диск Docker: {0:N1} ГБ. Свободно на C: {1:N1} ГБ." -f $before, $freeBefore)

Write-Host "Останавливаю WSL (Docker Desktop потом нужно будет запустить заново)..."
wsl --shutdown
Start-Sleep -Seconds 10

$plan = @"
select vdisk file="$vhd"
attach vdisk readonly
compact vdisk
detach vdisk
exit
"@
$script = Join-Path $env:TEMP "compact_docker_vhdx.txt"
Set-Content -Path $script -Value $plan -Encoding ASCII
Write-Host "Сжимаю (несколько минут, прогресс показывает diskpart)..."
diskpart /s $script
Remove-Item $script -ErrorAction SilentlyContinue

$after = (Get-Item $vhd).Length / 1GB
$freeAfter = (Get-PSDrive C).Free / 1GB
Write-Host ("Диск Docker: {0:N1} ГБ -> {1:N1} ГБ. Свободно на C: {2:N1} ГБ -> {3:N1} ГБ (освобождено {4:N1} ГБ)." `
    -f $before, $after, $freeBefore, $freeAfter, ($freeAfter - $freeBefore)) -ForegroundColor Green
Write-Host "Теперь можно запустить Docker Desktop и поднять прототип: docker compose up -d"
