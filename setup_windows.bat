@echo off
REM Windows 快速設置腳本
REM 用於設置球衣號碼檢測訓練環境

echo ========================================
echo 球衣號碼檢測模型訓練 - 環境設置
echo ========================================
echo.

REM 檢查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 未找到 Python，請先安裝 Python 3.8-3.11
    pause
    exit /b 1
)

echo [1/4] 創建虛擬環境...
if exist venv (
    echo 虛擬環境已存在，跳過創建
) else (
    python -m venv venv
    echo ✓ 虛擬環境創建成功
)

echo.
echo [2/4] 激活虛擬環境...
call venv\Scripts\activate.bat

echo.
echo [3/4] 升級 pip...
python -m pip install --upgrade pip

echo.
echo [4/4] 安裝依賴...
echo 注意：PyTorch 需要根據 CUDA 版本單獨安裝
echo 請參考 README.md 中的安裝步驟
pip install -r requirements.txt

echo.
echo ========================================
echo 環境設置完成！
echo ========================================
echo.
echo 下一步：
echo 1. 設置 ROBOFLOW_API_KEY 環境變數
echo 2. 安裝 PyTorch (CUDA 版本)
echo 3. 運行 python download_datasets.py 下載資料集
echo.
echo 詳細說明請參考 README.md
echo.
pause

