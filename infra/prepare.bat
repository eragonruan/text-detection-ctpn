@echo off
setlocal EnableDelayedExpansion

:: Set root directory
set root_dir=%~dp0\..

:: Install Cython
call python --version || exit /b 1
call pip install --upgrade Cython || exit /b 1

:: Install other necessary packages
call pip install --upgrade tf_slim || exit /b 1

:: Build Cython dependencies
cd !root_dir!\utils\bbox
python setup.py build
cd build\lib*
copy /y *.pyd ..\..
cd ..\..
rmdir /s /q build

:: Download trained model
cd !root_dir!\infra
if not exist checkpoints_mlt.zip (
    python download_gdrive_file.py 1HcZuB_MHqsKhKEKpfF1pEU85CYy4OlWO checkpoints_mlt.zip
)

:: Extract trained model using:
:: - tar, available on Windows 10 build 17063 or later, or
:: - 7zip, if it was found in path (for example via `choco install 7zip`)
where tar 2>nul
if !errorlevel! neq 0 (
    where 7z 2>nul
    if !errorlevel! neq 0 (
        echo Error, no utility found to extract the archive
        exit /b 1
    ) else (
        7z x -y checkpoints_mlt.zip -o!root_dir!
    )
) else (
    echo Extracting, this may take some time
    tar -x -f checkpoints_mlt.zip -C !root_dir!
)

endlocal
