@echo off
setlocal enabledelayedexpansion

set CUDA_PATH=%CUDA_PATH_V10_1%/
rem echo %CUDA_PATH%
rem set CUDA_PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/
set OPTIX_PATH=%OPTIX_PATH_V7_2%/
rem set OPTIX_PATH=C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.1.0/
rem echo %OPTIX_PATH%
set CCBIN="C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.28.29333/bin/Hostx64/x64"
set OUTPUT_PTX_PATH=assets/ptx/
set OUTPUT_PTX_EXT=.ptx
set sm_arch=60
dir /b /s "assets\kernels\*.cu"  > compile_cuda_list_tmp.txt

for /f "tokens=*" %%A in (compile_cuda_list_tmp.txt) do  (
call :func "%%~fA" %%~nA
rem goto endfor
)
:endfor
del compile_cuda_list_tmp.txt
goto end

:func
set filename=%2
set fullfilename=%1%
set out_ptx=%OUTPUT_PTX_PATH%%filename%%OUTPUT_PTX_EXT%

echo Compiling %fullfilename% to %out_ptx%
"%CUDA_PATH%bin/nvcc.exe" %fullfilename% -ptx -o "%out_ptx%" -ccbin %CCBIN% -m64 -D_USE_MATH_DEFINES -DNOMINMAX -arch=compute_%sm_arch% -code=sm_%sm_arch% --use_fast_math --compiler-options /D_USE_MATH_DEFINES -rdc true -DNVCC "-I%OPTIX_PATH%include" "-I%OPTIX_PATH%include"

:end
