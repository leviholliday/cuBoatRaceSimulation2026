@echo off
title Canoe Hull Search - Setup
echo.
echo               o                o
echo              /^|\___           /^|\___
echo       _______/_\____\_________/_\____\_______
echo       \                                     /
echo        \__________  C U B O A T  __________/
echo   ~~~~~~~  ~~~~ ~~~~~~  ~~ ~~~~~~  ~~~~ ~~~~~~
echo     ~~~~~ ~~~~~  ~~~~~~~ ~~~  ~~~~~ ~~~~~~~ ~~
echo.

where wsl >nul 2>&1
if errorlevel 1 (
  echo This version of Windows doesn't seem to support WSL.
  echo Please tell Levi and he'll help a different way.
  pause
  exit /b 1
)

net session >nul 2>&1
if errorlevel 1 (
  echo Asking Windows for permission -- click "Yes" on the popup...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

echo Installing WSL and Ubuntu. This can take a few minutes -- don't close
echo this window until it says "Done" below.
echo.
wsl --install -d Ubuntu

echo.
echo ================================================================
echo   Done! Two more things, then you're finished with this part:
echo.
echo     1. RESTART your computer now.
echo     2. After it restarts, open "Ubuntu" from your Start Menu.
echo        The first time it opens it will ask you to make up a
echo        username and password -- anything works, it's just for
echo        this Linux install and unrelated to your Windows login.
echo.
echo   Then go back to cuboat.netlify.app/guest for step 2.
echo ================================================================
echo.
pause
