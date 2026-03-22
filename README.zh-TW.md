[English](README.md) | **中文**

# Animal Eye Metrology

影像分割 + 度量衡管線：偵測影像中的動物、定位眼睛，並以三層精度測量雙眼間距與動物間眼距。

![架構圖](docs/architecture.png)

## 技術架構

| 元件 | 技術 | 用途 |
|------|------|------|
| API 框架 | FastAPI | REST API，附 Swagger UI |
| 實例分割 | YOLO11n-seg (ultralytics) | 偵測動物輪廓 + 邊界框 |
| 眼部關鍵點偵測 | RTMPose-m + AP-10K (rtmlib/ONNX) | 定位左右眼關鍵點 |
| 相對深度估計 | Depth Anything V2 (DA V2) | 逐像素相對深度，用於透視修正 |
| 度量深度估計 | Apple Depth Pro | 逐像素度量深度（公尺）+ 焦距 |
| 視覺化 | OpenCV | 標註結果影像 |
| 容器化 | Docker + docker-compose | 一鍵部署 |

## 三層測量架構

### 第一層：像素距離

影像 2D 平面上兩個眼睛座標之間的直線歐幾里得距離 — 就像拿尺量螢幕上兩個點有多遠。不考慮深度、不考慮相機，純粹的平面幾何。

- **公式**：`d = sqrt((x2-x1)^2 + (y2-y1)^2)`
- **單位**：像素
- **限制**：無物理意義 — 遠處的動物在圖片上看起來較小，像素距離被壓縮

### 第二層：深度修正像素距離

仍然是像素距離，但使用 Depth Anything V2 的相對深度圖做透視修正。解決的問題是：遠處的動物在圖片上看起來比較小，像素距離被壓縮。這層把壓縮補回來。

- **公式**：`corrected = pixel_dist * avg_depth / min(depth_a, depth_b)`
- **單位**：像素（經透視修正）
- **優勢**：比例關係比第一層更接近真實
- **備註**：僅相對深度 — 沒有絕對尺度，因此仍非物理量測
- **API**：使用 `depth_pro=fast` 取得此層（約 2 秒，不需安裝 Depth Pro）

### 第三層：度量距離

使用 Apple Depth Pro（ICLR 2025）取得度量深度（公尺）與估算的焦距，透過 pinhole camera model 將 2D 像素座標轉換為 3D 實際座標：

```
X = (x_pixel - cx) * Z / focal_length
Y = (y_pixel - cy) * Z / focal_length
d = sqrt((X2-X1)^2 + (Y2-Y1)^2 + (Z2-Z1)^2)
```

- **單位**：公尺
- **優勢**：唯一具有物理意義的距離 — 無需相機校正即可獲得真實 3D 歐幾里得距離
- **備註**：度量深度存在估計誤差 — 非實驗室級精度

#### 合理性檢查

將度量 IOD（雙眼間距）與已知生物學範圍交叉驗證：

| 動物 | 預期 IOD |
|------|----------|
| 貓 | 5-6 公分 |
| 狗 | 6-10 公分 |
| 長頸鹿 | 18-22 公分 |
| 綿羊 | 6-8 公分 |

結果：**PASS**（在範圍內）、**WARNING**（在 50% 容許範圍內）、**FAIL**（不合理）

## 快速開始

### 本地安裝

```bash
git clone <repo-url>
cd animal-metrology
pip install -r requirements.txt

# 安裝度量深度（選用 — 模型約 1.8GB）：
pip install git+https://github.com/apple/ml-depth-pro.git
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('apple/DepthPro', 'depth_pro.pt', local_dir='weights/')"

cp .env.example .env
python -m scripts.run_demo    # 處理測試影像
python -m app.main            # 啟動 API 伺服器
```

### Docker

```bash
docker compose up --build
```

API 位於 `http://localhost:8000`

## API 文件

Swagger UI：`http://localhost:8000/docs`

### 端點

| 方法 | 端點 | 說明 |
|------|------|------|
| GET | `/health` | 健康檢查 |
| GET | `/api/v1/coco/animals` | 瀏覽包含多隻動物的 COCO 影像 |
| POST | `/api/v1/analyze/{image_id}` | 對 COCO 影像執行管線 |
| POST | `/api/v1/analyze/upload` | 對上傳影像執行管線 |

### 查詢參數

| 參數 | 值 | 預設 | 說明 |
|------|------|------|------|
| `steps` | `segment`、`eyes`、`full` | `full` | 管線深度 |
| `visualize` | `true`、`false` | `true` | 是否產生標註影像 |
| `depth_pro` | `none`、`fast`、`metric` | `metric` | 深度估計模式 |

#### 深度模式說明

| 模式 | 模型 | 輸出（累計） | 速度 |
|------|------|-------------|------|
| `none` | — | `pixel_distance` | 即時 |
| `fast` | Depth Anything V2 | + `depth_corrected_px` | ~2 秒 |
| `metric` | Apple Depth Pro | + `metric_distance_m` + 合理性檢查 | ~30-60 秒 |

> 每個模式包含前一模式的所有欄位。`metric` 回傳完整三層結果。
> 若未安裝 Depth Pro 但選擇了 `metric`，系統會自動降級為 `fast` 模式並記錄警告。

### 範例

```bash
# 完整管線，三層距離（預設，約 30-60 秒）
curl -X POST "http://localhost:8000/api/v1/analyze/287545"

# 完整管線，僅像素距離（最快）
curl -X POST "http://localhost:8000/api/v1/analyze/287545?depth_pro=none"

# 完整管線，深度修正像素距離（約 2 秒）
curl -X POST "http://localhost:8000/api/v1/analyze/287545?depth_pro=fast"

# 僅分割，僅 JSON
curl -X POST "http://localhost:8000/api/v1/analyze/287545?steps=segment&visualize=false"

# 眼部偵測加視覺化，不使用深度
curl -X POST "http://localhost:8000/api/v1/analyze/287545?steps=eyes&depth_pro=none"

# 上傳圖片（不需要 COCO 資料集）
curl -X POST "http://localhost:8000/api/v1/analyze/upload" \
  -F "file=@my_photo.jpg" -G -d "depth_pro=fast"
```

## 輸出範例

![輸出範例 — 2 隻綿羊 (547383)](docs/examples/analyze_547383.jpg)

標註影像呈現管線三層輸出的完整結果：

- **彩色遮罩 + 輪廓**：實例分割結果（每隻動物分配不同顏色）
- **眼部標記 (L/R)**：偵測到的左右眼關鍵點，以圓圈框標示
- **實線**：Intra-animal 雙眼距離（同一隻動物的兩眼間距）
- **白色虛線**：Inter-animal 距離（不同動物右眼之間的距離）

#### 距離標籤解讀

**Intra-animal 標籤**（實線上）— 依深度模式累計顯示：

```
# depth_pro=none
48.9 px

# depth_pro=fast（DA V2）
48.9 px | corr: 52.3 px

# depth_pro=metric（Depth Pro）— 包含所有層
48.9 px | corr: 52.3 px | 0.10m | ! sheep IOD
```
- `48.9 px` — 第一層：像素距離
- `corr: 52.3 px` — 第二層：深度修正後的像素距離（已補償透視壓縮）
- `0.10m` — 第三層：Depth Pro 估算的度量距離（公尺）
- `! sheep IOD` — 合理性檢查結果

**IOD** = Inter-Ocular Distance（雙眼間距）。每個物種有已知的生物學 IOD 範圍，用來驗證度量結果是否合理。

**合理性檢查圖示**：
- `v`（綠色）= **PASS** — 度量距離在預期 IOD 範圍內
- `!`（黃色）= **WARNING** — 超出範圍但在 50% 容許範圍內
- `x`（紅色）= **FAIL** — 明顯不合理，可能是眼部偵測或深度估計有誤

**Inter-animal 標籤**（虛線上）— 依深度模式累計顯示：

```
# depth_pro=none
#0-#1 R-eye: 175.6 px

# depth_pro=fast
#0-#1 R-eye: 170.5 px | corr: 184.7 px

# depth_pro=metric
#0-#1 R-eye: 175.6 px | corr: 184.7 px | 2.50m
```
- `#0-#1` — 動物配對 ID
- `R-eye` — 以各動物的右眼為測量點
- `175.6 px` — 像素距離
- `corr: 184.7 px` — 深度修正後的像素距離
- `2.50m` — 度量距離（無合理性檢查 — 動物間距離沒有生物學參考值）

## 測試影像

| 影像 ID | 動物 | 說明 |
|---------|------|------|
| 287545 | 2 隻長頸鹿 | 大型動物，不同深度 |
| 402473 | 2 隻貓 | 小型動物，相近深度 |
| 547383 | 2 隻綿羊 | 中型動物，不同深度 |

## 模型選擇理由

### YOLO11n-seg
- **選用原因**：COCO 預訓練，一次完成偵測 + 分割，nano 版本適合 CPU
- **指標**：mAP@50 = 73.4（COCO val2017）
- **評估方式**：IoU（Intersection over Union）衡量分割遮罩品質

### RTMPose-m + AP-10K
- **選用原因**：AP-10K 是標準動物姿態基準（17 關鍵點、23 物種）
- **指標**：AP@0.5 = 72.2（AP-10K）
- **評估方式**：OKS（Object Keypoint Similarity）衡量關鍵點準確度
- **部署方式**：透過 rtmlib 以 ONNX 格式載入 — 避免沉重的 mmpose/mmcv 依賴

### Depth Anything V2
- **選用原因**：高品質相對深度圖，用於透視修正（第二層）
- **優勢**：輕量、快速，不需要焦距 — 只需要相對順序
- **設計**：用於第二層，補償像素距離的透視壓縮

### Apple Depth Pro
- **選用原因**：輸出度量深度（公尺）+ 估計焦距 — 無需相機校正
- **論文**："Depth Pro: Sharp Monocular Metric Depth in Less Than a Second"（ICLR 2025）
- **評估方式**：AbsRel、RMSE（標準基準：NYU、KITTI）
- **設計**：驅動第三層（度量距離）；選用元件 — 若不可用，管線會優雅降級為僅像素模式

## 驗證方法

1. **視覺化**：標註影像包含眼部標記、距離線與三層標籤
2. **合理性檢查**：度量 IOD 與已知動物生物學交叉驗證
3. **單元測試**：27 個測試涵蓋距離計算、度量投影、合理性檢查邏輯
4. **展示腳本**：`python -m scripts.run_demo` 端對端處理所有測試影像

## 限制

- **度量深度準確度**：Depth Pro 為零樣本推論 — 誤差隨場景變化。合理性檢查有助標記異常值。
- **眼部偵測**：RTMPose 在側面視角可能失敗（眼部設為 None，跳過距離計算）
- **YOLO 閾值**：0.5 信心度可能遺漏部分遮擋的動物
- **CPU 推論**：Depth Pro 在 CPU 上每張影像約需 30-60 秒；正式環境建議使用 GPU
- **無相機校正**：Depth Pro 估計焦距，但真正的校正會更準確

## 執行測試

```bash
python -m pytest tests/ -v
```

## 專案結構

```
animal-metrology/
├── app/
│   ├── main.py                     # FastAPI 進入點
│   ├── config.py                   # Pydantic 設定
│   ├── models/schemas.py           # 回應結構（三層）
│   ├── routers/analyze.py          # API 端點
│   ├── services/
│   │   ├── coco_filter.py          # COCO 資料集過濾
│   │   ├── segmentation.py         # YOLO 實例分割
│   │   ├── eye_detection.py        # RTMPose 眼部關鍵點
│   │   ├── depth_estimation.py     # Depth Pro / DA V2
│   │   └── measurement.py          # 三層距離計算
│   └── utils/visualization.py      # 結果視覺化
├── tests/                          # 單元 + 整合測試
├── data/
│   ├── test_images/                # 3 張預選 COCO 影像
│   ├── test_annotations/           # 精簡版 COCO 標註 JSON
│   └── sample_results.csv          # 管線輸出 CSV
├── scripts/
│   └── run_demo.py                 # 一鍵展示
├── docs/architecture.png
├── Dockerfile, docker-compose.yml
├── requirements.txt, .env.example
└── README.md
```
