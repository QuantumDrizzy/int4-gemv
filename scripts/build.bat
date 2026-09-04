@echo off
REM Build both binaries. nvcc needs cl.exe on PATH, and the VS generator does not
REM pick up the CUDA toolset on this machine, so vcvars64 is loaded explicitly.
if "%VCVARS%"=="" set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
  echo vcvars64.bat not found - set VCVARS to your Visual Studio install
  exit /b 1
)
call %VCVARS% >nul
if not exist bench mkdir bench
nvcc -O3 -arch=sm_120 -Isrc -o int4_gemv.exe src\int4_gemv.cu || exit /b 1
nvcc -O3 -arch=sm_120 -Isrc -o diag_aspect.exe src\diag_aspect.cu || exit /b 1
echo built int4_gemv.exe and diag_aspect.exe
