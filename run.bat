@echo off
title AI-Based Deepfake Image Detection System
cd /d "%~dp0"

set "JAVA_EXE=%~dp0..\jdk-17\bin\java.exe"
if not exist "%JAVA_EXE%" set "JAVA_EXE=java"

echo ========================================================
echo   Starting AI Deepfake Image Detection System...
echo ========================================================

start "" "%JAVA_EXE%" -cp "bin;lib\onnxruntime-1.17.1.jar" com.deepfake.Main
