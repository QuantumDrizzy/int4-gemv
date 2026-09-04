@echo off
REM Build the kernel. nvcc needs cl.exe on PATH, and the VS generator does not
REM pick up the CUDA toolset on this machine, so vcvars64 is loaded explicitly.
set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
  echo vcvars64.bat not found - set VCVARS to your Visual Studio install
  exit /b 1
)
call %VCVARS% >nul
if not exist bench mkdir bench
nvcc -O3 -arch=sm_120 -o int4_gemv.exe src\int4_gemv.cu
