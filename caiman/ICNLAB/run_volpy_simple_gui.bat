@REM @echo off
@REM call C:\Users\ICNLab\anaconda3\condabin\conda.bat activate caiman
@REM python "%~dp0test_single_trial_RAM_DISK_5.4_simple_GUI.py"
@echo off


:: Step 1: Try to activate using standard PATH (if they have Conda initialized)
call conda activate caiman 2>nul
if not errorlevel 1 goto :run_script


:: Step 2: If not in PATH, search common default installation locations
set "FOUND_CONDA=0"
for %%D in (
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\AppData\Local\anaconda3"
    "%USERPROFILE%\AppData\Local\miniconda3"
    "C:\ProgramData\anaconda3"
    "C:\ProgramData\miniconda3"
) do (
    if exist "%%~D\condabin\conda.bat" (
        call "%%~D\condabin\conda.bat" activate caiman
        set "FOUND_CONDA=1"
        goto :run_script
    )
)


:: Step 3: What to do if Conda is completely missing
if "%FOUND_CONDA%"=="0" (
    echo [ERROR] Could not find Conda on this PC!
    echo Please ensure Miniconda or Anaconda is installed.
    pause
    exit /b
)


:run_script
:: Step 4: Run your python file using the directory of this batch file
python "%~dp0volpy_simple_GUI.py"


:: Optional: Keep the window open so they can see any errors before it closes
pause
