# 球衣號碼檢測模型訓練專案

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2F12.1-green)
![License](https://img.shields.io/badge/license-MIT-blue)

這個專案用於訓練一個專門檢測排球運動員球衣號碼的 YOLOv8 模型。資料集來自 Roboflow Universe 的多個公開資料集。

## 🏷️ Topics

`yolov8` `jersey-detection` `computer-vision` `deep-learning` `pytorch` `object-detection` `volleyball` `roboflow` `yolo` `machine-learning`

## 📋 目錄

- [系統需求](#系統需求)
- [安裝步驟](#安裝步驟)
- [資料集下載](#資料集下載)
- [資料集組織](#資料集組織)
- [模型訓練](#模型訓練)
- [整合到專案](#整合到專案)
- [疑難排解](#疑難排解)

## 🔧 系統需求

### Windows 系統（推薦用於 GPU 訓練）

- **作業系統**: Windows 10/11
- **Python**: 3.8-3.11（建議 3.10）
- **GPU**: NVIDIA GPU，支援 CUDA（建議 6GB+ VRAM）
- **CUDA**: 11.8 或 12.1
- **記憶體**: 至少 16GB RAM
- **磁碟空間**: 至少 20GB 可用空間（用於資料集和模型）

### 檢查 CUDA 版本

在命令提示字元中運行：

```cmd
nvidia-smi
```

這會顯示您的 CUDA 版本和 GPU 資訊。

## 📦 安裝步驟

### 1. 克隆或下載專案

```cmd
cd C:\Users\YourName\Documents
git clone <repository-url>
cd jersey_detection
```

或直接下載並解壓縮專案文件夾。

### 2. 創建虛擬環境

#### 選項 A: 使用 Conda（推薦，特別是用於 GPU 訓練）

**安裝 Conda**（如果尚未安裝）:
- 下載並安裝 [Anaconda](https://www.anaconda.com/products/distribution) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

**創建 Conda 環境**:

```cmd
conda create -n jersey_detection python=3.10
conda activate jersey_detection
```

**安裝 PyTorch（CUDA 版本）**:

使用 Conda 安裝 PyTorch 通常更方便，因為它會自動處理 CUDA 和 cuDNN：

#### CUDA 11.8:

```cmd
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### CUDA 12.1:

```cmd
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### CPU 版本（不推薦，訓練很慢）:

```cmd
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

**安裝其他依賴**:

```cmd
pip install -r requirements.txt
```

#### 選項 B: 使用 Python venv

**創建虛擬環境**:

```cmd
python -m venv venv
venv\Scripts\activate
```

**安裝 PyTorch（CUDA 版本）**:

**重要**: 根據您的 CUDA 版本選擇對應的 PyTorch 安裝命令。

#### CUDA 11.8:

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 12.1:

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CPU 版本（不推薦，訓練很慢）:

```cmd
pip install torch torchvision torchaudio
```

**安裝其他依賴**:

```cmd
pip install -r requirements.txt
```

### 3. 驗證安裝

```cmd
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

如果看到 `CUDA available: True`，表示安裝成功！

**注意**: 
- **推薦使用 Conda**，特別是對於 GPU 訓練，因為 Conda 可以更好地管理 CUDA 和 cuDNN 依賴
- 如果使用 venv，請確保已正確安裝 NVIDIA 驅動程式和 CUDA 工具包
- 如果遇到 CUDA 相關問題，建議使用 Conda 重新安裝

## 📥 資料集下載

### 步驟 1: 獲取 Roboflow API Key

1. 前往 [Roboflow](https://roboflow.com/)
2. 註冊/登入帳號
3. 前往 [Account Settings](https://app.roboflow.com/) → API
4. 複製您的 API Key

### 步驟 2: 設置環境變數

#### Windows (命令提示字元):

```cmd
set ROBOFLOW_API_KEY=your_api_key_here
```

#### Windows (PowerShell):

```powershell
$env:ROBOFLOW_API_KEY="your_api_key_here"
```

#### 永久設置（可選）:

1. 右鍵「此電腦」→「內容」
2. 「進階系統設定」→「環境變數」
3. 新增系統變數：
   - 變數名稱: `ROBOFLOW_API_KEY`
   - 變數值: 您的 API Key

### 步驟 3: 下載資料集

```cmd
python download_datasets.py
```

這會下載以下 4 個資料集：

1. `volleyai-actions/jersey-number-detection-s01j4`
2. `workspace67/jersey-fxmll`
3. `teste-5efoz/player-number-detect`
4. `hgjhj/jersey-number-detection-br3ld`

下載的資料集會保存在 `datasets/raw/` 目錄下。

**預期時間**: 根據網路速度，可能需要 10-30 分鐘。

**問題排查**:
- 如果出現 API Key 錯誤，確認環境變數已正確設置
- 如果下載失敗，檢查網路連接
- 如果磁碟空間不足，清理一些空間後重試

## 🔄 資料集組織

下載完成後，需要合併和組織資料集：

```cmd
python organize_datasets.py
```

此腳本會：

1. 分析所有資料集的類別
2. 統一類別映射（將不同資料集的類別ID映射到統一的ID）
3. 合併訓練/驗證/測試集
4. 生成 `data.yaml` 配置文件

合併後的資料集保存在 `datasets/processed/merged/` 目錄下。

**輸出結構**:
```
datasets/processed/merged/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

## 🚀 模型訓練

### 基本訓練

```cmd
python train_model.py
```

### 訓練配置

預設配置（可在 `train_model.py` 中修改）：

- **模型**: YOLOv8m (Medium) - 推薦，平衡準確度和速度
- **Epochs**: 100
- **圖片大小**: 自動優化（RTX 5060: 640x640 for YOLOv8m）
- **Batch size**: 自動優化（RTX 5060: 16 for YOLOv8m，避免 OOM）
- **早停耐心值**: 20 epochs
- **優化器**: AdamW
- **學習率**: 0.001
- **混合精度訓練 (AMP)**: 啟用（提高訓練速度）
- **Workers**: 自動優化（Windows 推薦 4，避免多工錯誤）

### 選擇不同的模型

在 `train_model.py` 中修改 `MODEL_NAME`（目前預設為 `yolov8m.pt`）：

- `yolov8n.pt` - Nano（最快，參數最少，適合快速測試）
- `yolov8s.pt` - Small（平衡速度和準確度）
- `yolov8m.pt` - Medium（**預設**，推薦，較好的準確度）
- `yolov8l.pt` - Large（高準確度）
- `yolov8x.pt` - XLarge（最高準確度，但訓練最慢）

### 訓練過程

訓練開始後，您會看到：

- 每個 epoch 的進度條
- 訓練損失（loss）
- 驗證指標（mAP, precision, recall）
- 訓練曲線圖

訓練結果會保存在 `runs/jersey_detection/` 目錄下。

**預期時間**（使用自動優化配置）:
- GPU (RTX 5060): 約 2-3 小時（100 epochs, YOLOv8m）
- GPU (RTX 3060): 約 3-5 小時（100 epochs, YOLOv8m）
- GPU (RTX 4090): 約 1.5-2.5 小時（100 epochs, YOLOv8m）
- CPU: 可能需要 15+ 小時（不推薦）

**注意**: 訓練腳本會自動檢測您的 GPU 並優化參數。對於 RTX 5060，使用 YOLOv8m 模型會自動使用 batch size 16（優化以避免 OOM）和圖片尺寸 640x640 以充分利用 GPU 性能。Windows 系統會自動使用較少的 workers (4) 以避免多工錯誤。

### 訓練結果

訓練完成後，您會找到：

- **最佳模型**: `runs/jersey_detection/weights/best.pt`
- **最新模型**: `runs/jersey_detection/weights/last.pt`
- **訓練曲線**: `runs/jersey_detection/results.png`
- **驗證結果**: `runs/jersey_detection/val_batch0_labels.jpg`
- **混淆矩陣**: `runs/jersey_detection/confusion_matrix.png`

## 🔗 整合到專案

訓練完成後，將模型整合到您的 [volleyball-analysis](https://github.com/itsYoga/volleyball-analysis) 專案：

### 方法 1: 使用整合腳本

```cmd
python integrate_model.py
```

此腳本會自動：
1. 尋找最佳模型
2. 複製到專案目錄
3. 創建使用範例代碼

### 方法 2: 手動複製

```cmd
copy runs\jersey_detection\weights\best.pt ..\volleyball-analysis\models\jersey_detection_yv8.pt
```

### 使用範例

參考 `integration_example.py` 中的範例代碼，在您的專案中使用模型：

```python
from integration_example import JerseyNumberDetector

# 初始化檢測器
detector = JerseyNumberDetector()

# 檢測球衣號碼
detections = detector.detect(image)
```

## 🐛 疑難排解

### 問題 1: CUDA 不可用

**症狀**: 訓練時顯示 "CUDA available: False"

**解決方案**:
1. 確認已安裝 CUDA 版本的 PyTorch
2. 確認 NVIDIA 驅動程式已安裝（運行 `nvidia-smi`）
3. 確認 CUDA 版本匹配（PyTorch 和系統 CUDA）
4. 重新安裝對應版本的 PyTorch

### 問題 2: 記憶體不足 (Out of Memory)

**症狀**: 訓練時出現 "CUDA out of memory"

**解決方案**:
1. 減少 batch size（在 `train_model.py` 中修改 `batch_size`）
2. 使用較小的模型（如 `yolov8n.pt`）
3. 減少圖片大小（修改 `imgsz`）
4. 關閉其他使用 GPU 的應用程式

### 問題 3: 下載資料集失敗

**症狀**: "下載失敗" 或 API 錯誤

**解決方案**:
1. 確認 `ROBOFLOW_API_KEY` 環境變數已設置
2. 檢查網路連接
3. 確認 Roboflow API 配額未用完
4. 重新運行下載腳本

### 問題 4: 資料集為空

**症狀**: "錯誤: 資料集為空"

**解決方案**:
1. 確認已運行 `download_datasets.py`
2. 確認已運行 `organize_datasets.py`
3. 檢查 `datasets/processed/merged/` 目錄下是否有文件

### 問題 5: 訓練速度很慢

**解決方案**:
1. 確認使用 GPU（檢查 `CUDA available: True`）
2. 增加 batch size（如果記憶體允許）
3. 減少 workers 數量（如果數據載入是瓶頸）
4. 使用較小的模型進行快速測試

### 問題 6: 模型準確度不佳

**解決方案**:
1. 訓練更多 epochs（增加 `epochs` 參數）
2. 使用更大的模型（如 `yolov8m.pt` 或 `yolov8l.pt`）
3. 檢查資料集品質和標註是否正確
4. 調整學習率和其他超參數
5. 使用資料增強（已預設啟用）

## 📊 訓練監控

### 使用 TensorBoard（可選）

安裝 TensorBoard：

```cmd
pip install tensorboard
```

啟動 TensorBoard：

```cmd
tensorboard --logdir runs/jersey_detection
```

然後在瀏覽器中打開 `http://localhost:6006`

### 查看訓練日誌

訓練日誌保存在 `runs/jersey_detection/` 目錄下，包括：

- `results.csv` - 每個 epoch 的指標
- `train_batch*.jpg` - 訓練批次視覺化
- `val_batch*.jpg` - 驗證批次視覺化

## 📝 注意事項

1. **環境選擇**: 推薦使用 Conda 環境，特別是對於 GPU 訓練，因為 Conda 可以更好地管理 CUDA 依賴
2. **備份**: 訓練前確保有足夠的磁碟空間，建議至少 20GB
3. **中斷恢復**: 如果訓練中斷，可以從檢查點恢復（如果設置了 `resume=True`）
4. **模型選擇**: 根據您的需求選擇模型大小，較大的模型通常更準確但訓練更慢
5. **GPU 記憶體**: 訓練腳本會自動優化參數，但如果遇到記憶體問題，可以手動減少 batch size
6. **資料集品質**: 確保資料集標註正確，這會直接影響模型性能
7. **RTX 5060 優化**: 訓練腳本已針對 RTX 5060 進行優化，會自動使用最佳參數配置

## 🎯 下一步

訓練完成後：

1. 評估模型性能（查看 `results.png`）
2. 測試模型在實際圖片上的表現
3. 整合到您的排球分析專案中
4. 根據實際使用情況調整和優化

## 📚 參考資料

- [YOLOv8 文檔](https://docs.ultralytics.com/)
- [Roboflow Universe](https://universe.roboflow.com/)
- [PyTorch 安裝指南](https://pytorch.org/get-started/locally/)
- [排球分析專案](https://github.com/itsYoga/volleyball-analysis)

## 📄 授權

本專案使用 MIT 授權。

## 🤝 貢獻

歡迎提交 Issue 或 Pull Request！

---

**祝訓練順利！** 🎉

如有問題，請查看 [疑難排解](#疑難排解) 部分或開 Issue。
