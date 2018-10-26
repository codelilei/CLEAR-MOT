color A
@echo off&setlocal EnableDelayedExpansion

D:
cd D:\Projects\CLEAR_MOT\Release

::echo %~dp0
::dir /s /b %~dp0\*.mp4
for /f %%i in ('dir /s /b %~dp0\*.mp4') do (
    echo %%i
    rem set /p skip=press n or N to skip:
    rem if not !skip! == n if not !skip! == N (
    choice /T 5 /D n /M "skip this?"
    if !errorlevel! == 2 (
        CLEAR_MOT.exe %%i 1
    )
    rem set skip=y
)

pause