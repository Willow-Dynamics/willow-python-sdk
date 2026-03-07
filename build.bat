@echo off
setlocal

:: ==============================================================================
:: WILLOW 5 RUNTIME SDK - AUTOMATED DEPLOYMENT SCRIPT
:: ==============================================================================
:: This script performs a clean build, enforces correct Git remotes,
:: uploads artifacts to PyPI, and pushes version tags to GitHub.
:: ==============================================================================

:: --- CONFIGURATION CONSTANTS ---
set "REPO_URL=https://github.com/Willow-Dynamics/willow-python-sdk.git"
set "PACKAGE_NAME=willow-runtime"

echo ========================================================
echo    WILLOW SDK DEPLOYMENT SEQEUENCE
echo ========================================================
echo.

:: --- STEP 1: VERSION INPUT & VALIDATION ---
echo [CHECK] Have you updated the version number in 'pyproject.toml'?
echo.
set /p VERSION="Enter the Target Version (e.g., 5.2.0): "

if "%VERSION%"=="" (
    echo [ERROR] Version cannot be empty. Aborting.
    goto :FAIL
)

echo.
echo [TARGET] Deploying Version: %VERSION%
echo [TARGET] Repository: %REPO_URL%
echo.
pause

:: --- STEP 2: CLEANUP OLD ARTIFACTS ---
echo.
echo [1/6] Cleaning previous build artifacts...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"
if exist "%PACKAGE_NAME%.egg-info" rmdir /s /q "%PACKAGE_NAME%.egg-info"
if exist "willow_runtime.egg-info" rmdir /s /q "willow_runtime.egg-info"
echo       Clean complete.

:: --- STEP 3: ENFORCE GIT SAFETY (ISOLATION) ---
echo.
echo [2/6] Enforcing Git Remote Isolation...
:: We remove and re-add origin to ensure 100% certainty where this code is going.
git remote remove origin 2>NUL
git remote add origin %REPO_URL%
echo       Origin forced to: %REPO_URL%
git remote -v

:: --- STEP 4: BUILD PACKAGE ---
echo.
echo [3/6] Building Python Wheel and Source Tarball...
python -m build
if %errorlevel% neq 0 goto :FAIL
echo       Build successful.

:: --- STEP 5: PYPI UPLOAD ---
echo.
echo [4/6] Uploading to PyPI...
echo.
echo       You will need your PyPI API Token (pypi-AgEI...).
echo       If you want to paste it once and hide it, paste it below.
echo       (Leave empty to use Twine's interactive prompt).
echo.
set /p PYPI_TOKEN="Token (Optional - Press Enter to skip): "

if not "%PYPI_TOKEN%"=="" (
    set TWINE_USERNAME=__token__
    set TWINE_PASSWORD=%PYPI_TOKEN%
    python -m twine upload dist/*
    :: Clear token from memory immediately
    set TWINE_PASSWORD=
    set PYPI_TOKEN=
) else (
    python -m twine upload dist/*
)

if %errorlevel% neq 0 goto :FAIL
echo       PyPI Upload successful.

:: --- STEP 6: GIT COMMIT & PUSH ---
echo.
echo [5/6] Committing Source Code...
git add .
git commit -m "Release v%VERSION%"

echo.
echo [6/6] Pushing to GitHub...
:: Push code
git push -u origin main
if %errorlevel% neq 0 goto :FAIL

:: Tag and Push Tag
echo       Tagging Release v%VERSION%...
git tag v%VERSION%
git push origin v%VERSION%
if %errorlevel% neq 0 goto :FAIL

echo.
echo ========================================================
echo    DEPLOYMENT COMPLETE: v%VERSION%
echo ========================================================
echo    1. PyPI: https://pypi.org/project/%PACKAGE_NAME%/
echo    2. GitHub: %REPO_URL%
echo ========================================================
pause
exit /b 0

:FAIL
echo.
echo [CRITICAL FAILURE] Deployment stopped due to errors.
echo Check the logs above.
pause
exit /b 1