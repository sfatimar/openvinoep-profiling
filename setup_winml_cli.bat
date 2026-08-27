@echo off
rem ===========================================================================
rem  setup_winml_cli.bat
rem
rem  Create a Python virtual environment named `winml_cli` and install the
rem  Microsoft WinML CLI (https://github.com/microsoft/winml-cli) into it.
rem
rem  Usage (from this directory):
rem      setup_winml_cli.bat              Create the env and install winml-cli
rem      setup_winml_cli.bat --force      Delete and recreate the env
rem      setup_winml_cli.bat --upgrade    Upgrade winml-cli in an existing env
rem      setup_winml_cli.bat --help       Show this help
rem
rem  Requirements:
rem    - uv               https://docs.astral.sh/uv/
rem    - Windows 11 24H2+ required for NPU enumeration; CPU/GPU work on older builds
rem
rem  Notes:
rem    - winml-cli requires Python 3.11 specifically, and x64 (AMD64) Python only:
rem      its PyTorch and Windows ML runtime dependencies publish no win_arm64
rem      wheels. The interpreter is pinned explicitly rather than relying on
rem      whichever 3.11 happens to be first on PATH.
rem    - The env is named `winml_cli`, NOT `.venv`, so a bare `uv run winml`
rem      will not find it -- uv auto-discovers only `.venv`. See the notes printed
rem      at the end, or just use resnet_perf_example.py which handles this.
rem ===========================================================================

setlocal EnableDelayedExpansion

set "VENV_DIR=winml_cli"
set "PYTHON_SPEC=cpython-3.11-windows-x86_64-none"
set "PACKAGE=winml-cli"
set "FORCE=0"
set "UPGRADE=0"

rem --- argument parsing ------------------------------------------------------

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--force" (
    set "FORCE=1"
    shift
    goto parse_args
)
if /I "%~1"=="--upgrade" (
    set "UPGRADE=1"
    shift
    goto parse_args
)
if /I "%~1"=="--help" goto show_help
if /I "%~1"=="-h"     goto show_help
if /I "%~1"=="/?"     goto show_help
echo error: unknown argument "%~1"  ^(try --help^)
exit /b 2

:show_help
echo.
echo Create a Python virtual environment named "%VENV_DIR%" and install the
echo Microsoft WinML CLI into it.
echo.
echo Usage:
echo     setup_winml_cli.bat              Create the env and install winml-cli
echo     setup_winml_cli.bat --force      Delete and recreate the env
echo     setup_winml_cli.bat --upgrade    Upgrade winml-cli in an existing env
echo     setup_winml_cli.bat --help       Show this help
echo.
echo Requirements: uv, and Windows 11 24H2 or newer for NPU enumeration.
echo.
exit /b 0

:args_done

rem Run relative to this script, not the caller's working directory.
pushd "%~dp0" || (echo error: could not enter script directory & exit /b 1)

rem --- preflight -------------------------------------------------------------

echo.
echo ==^> Checking prerequisites

where uv >nul 2>&1
if errorlevel 1 (
    echo.
    echo error: uv not found on PATH.
    echo   Install it with:
    echo       powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo   or see https://docs.astral.sh/uv/getting-started/installation/
    popd
    exit /b 1
)

for /f "tokens=*" %%v in ('uv --version 2^>nul') do set "UV_VERSION=%%v"
echo uv: !UV_VERSION!

if /I not "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
    echo warning: PROCESSOR_ARCHITECTURE is %PROCESSOR_ARCHITECTURE%; winml-cli needs x64 Python.
    echo          Pinning %PYTHON_SPEC% anyway -- on Windows on Arm the x64
    echo          interpreter will run under emulation.
)

rem --- interpreter -----------------------------------------------------------

echo.
echo ==^> Ensuring Python 3.11 ^(x64^) is available

uv python find "%PYTHON_SPEC%" >nul 2>&1
if errorlevel 1 (
    echo not present; downloading via uv
    uv python install "%PYTHON_SPEC%"
    if errorlevel 1 (
        echo error: failed to install %PYTHON_SPEC%
        popd
        exit /b 1
    )
) else (
    for /f "tokens=*" %%p in ('uv python find "%PYTHON_SPEC%" 2^>nul') do set "FOUND_PYTHON=%%p"
    echo found: !FOUND_PYTHON!
)

rem --- virtual environment ---------------------------------------------------

if "%FORCE%"=="1" if exist "%VENV_DIR%\" (
    echo.
    echo ==^> Removing existing %VENV_DIR% ^(--force^)
    rmdir /s /q "%VENV_DIR%"
    if exist "%VENV_DIR%\" (
        echo error: could not remove %VENV_DIR% -- is a shell or editor holding it open?
        popd
        exit /b 1
    )
)

if exist "%VENV_DIR%\" (
    echo.
    echo ==^> Reusing existing environment: %VENV_DIR%
) else (
    echo.
    echo ==^> Creating virtual environment: %VENV_DIR%
    uv venv "%VENV_DIR%" --python "%PYTHON_SPEC%"
    if errorlevel 1 (
        echo error: failed to create the virtual environment
        popd
        exit /b 1
    )
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo error: could not locate the venv interpreter at %VENV_PYTHON%
    popd
    exit /b 1
)

for /f "tokens=*" %%i in ('"%VENV_PYTHON%" -c "import sys,platform; print(sys.version.split()[0], platform.machine())" 2^>nul') do set "PY_INFO=%%i"
echo interpreter: !PY_INFO!

rem --- install ---------------------------------------------------------------

echo.
if "%UPGRADE%"=="1" (
    echo ==^> Upgrading %PACKAGE%
    uv pip install --python "%VENV_PYTHON%" --upgrade "%PACKAGE%"
) else (
    echo ==^> Installing %PACKAGE%
    uv pip install --python "%VENV_PYTHON%" "%PACKAGE%"
)
if errorlevel 1 (
    echo error: %PACKAGE% installation failed
    popd
    exit /b 1
)

rem --- verify ----------------------------------------------------------------

set "WINML=%VENV_DIR%\Scripts\winml.exe"
if not exist "%WINML%" (
    echo error: %PACKAGE% installed but no winml.exe found at %WINML%
    popd
    exit /b 1
)

echo.
echo ==^> Verifying installation
"%WINML%" --version 2>nul
if errorlevel 1 echo warning: "winml --version" not supported by this build; continuing

echo.
echo ==^> Enumerating devices and execution providers
rem Informational only. A non-zero exit here usually means the Windows ML runtime
rem is missing or the OS build predates NPU support -- the install is still valid.
"%WINML%" sys --list-device --list-ep
if errorlevel 1 (
    echo warning: "winml sys" returned non-zero. The CLI is installed, but device
    echo          enumeration failed. Common causes: Windows older than 11 24H2,
    echo          or a missing Windows ML runtime.
)

rem --- done ------------------------------------------------------------------

echo.
echo ===========================================================================
echo Done. winml-cli is installed in .\%VENV_DIR%
echo.
echo Run the ResNet example ^(handles the non-standard venv name for you^):
echo.
echo     python resnet_perf_example.py
echo.
echo Or drive the CLI directly:
echo.
echo     .\%VENV_DIR%\Scripts\winml.exe perf -m resnet_out\model.onnx --device auto --iterations 50 --monitor
echo.
echo To use the documented "uv run winml ..." form with this env, uv must be told
echo about it -- uv only auto-discovers a venv named ".venv":
echo.
echo     rem cmd.exe
echo     .\%VENV_DIR%\Scripts\activate.bat
echo     uv run --active winml perf -m resnet_out\model.onnx --device auto --iterations 50 --monitor
echo.
echo     rem PowerShell
echo     .\%VENV_DIR%\Scripts\Activate.ps1
echo     uv run --active winml perf -m resnet_out\model.onnx --device auto --iterations 50 --monitor
echo ===========================================================================
echo.

popd
endlocal
exit /b 0
