@echo off
title Compile Deepfake Detector
cd /d "%~dp0"

set "JAVAC_EXE=%~dp0..\jdk-17\bin\javac.exe"
if not exist "%JAVAC_EXE%" set "JAVAC_EXE=javac"

echo Compiling Java source files...
if not exist "bin" mkdir "bin"

"%JAVAC_EXE%" -cp "lib\onnxruntime-1.17.1.jar" -d "bin" src\main\java\com\deepfake\*.java

if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Compilation finished with 0 errors!
) else (
    echo [ERROR] Compilation failed!
)
pause
