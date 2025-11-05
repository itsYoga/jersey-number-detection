# RTX 5060 GPU 優化配置說明

## RTX 5060 規格
- **VRAM**: 8GB GDDR6
- **CUDA Cores**: 2048
- **記憶體頻寬**: 256 GB/s
- **建議 CUDA 版本**: 11.8 或 12.1

## 已優化的參數

### 自動優化功能
訓練腳本會自動根據您的 GPU 規格優化以下參數：

1. **Batch Size**
   - YOLOv8n: 64
   - YOLOv8s: 48
   - YOLOv8m: 32
   - YOLOv8l: 24
   - YOLOv8x: 16

2. **Image Size**
   - 較小模型 (n/s): 832x832
   - 較大模型 (m/l/x): 640x640

3. **Workers**
   - 設置為 12（充分利用多核心 CPU 加速數據載入）

4. **混合精度訓練 (AMP)**
   - 已啟用，可提高約 30-50% 訓練速度並節省記憶體

5. **CUDA 優化**
   - cuDNN Benchmark: 啟用
   - TF32: 啟用（如果 GPU 支援）

## 手動調整（如果需要）

如果遇到記憶體不足錯誤，可以在 `train_model.py` 中調整：

```python
# 減少 batch size
batch_size = 32  # 從 64 減少到 32

# 減少圖片大小
img_size = 640  # 從 832 減少到 640

# 減少 workers
workers = 8  # 從 12 減少到 8

# 禁用混合精度（如果出現問題）
amp = False
```

## 監控 GPU 使用

訓練時可以使用以下命令監控 GPU：

```cmd
nvidia-smi -l 1
```

這會每秒更新一次 GPU 使用情況。

## 預期性能

使用 RTX 5060 訓練 YOLOv8n：
- **訓練速度**: 約 50-80 張圖片/秒
- **訓練時間**: 100 epochs 約 1.5-2.5 小時（取決於資料集大小）
- **GPU 利用率**: 應該達到 95%+ 
- **記憶體使用**: 約 6-7GB VRAM

## 進一步優化建議

1. **使用更大的模型**: 如果記憶體允許，可以嘗試 YOLOv8s 或 YOLOv8m 以獲得更好的準確度

2. **啟用數據緩存**: 如果 RAM 足夠（32GB+），可以在訓練參數中設置 `cache='ram'`

3. **調整學習率**: 對於較大的 batch size，可能需要調整學習率：
   ```python
   lr0=0.001 * (batch_size / 32)  # 線性縮放
   ```

4. **使用多 GPU**: 如果有多個 GPU，可以設置 `device=[0, 1]` 使用多 GPU 訓練

## 疑難排解

### 記憶體不足 (OOM)
- 減少 batch size
- 減少圖片大小
- 關閉數據緩存

### GPU 利用率低
- 增加 workers 數量
- 檢查數據載入是否為瓶頸
- 確認啟用了混合精度訓練

### 訓練速度慢
- 確認已安裝 CUDA 版本的 PyTorch
- 檢查驅動程式是否最新
- 確認 cuDNN 已正確安裝

