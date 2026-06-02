# Hand Tracking NPU 專案工作說明

以下依四個方向（3-6）說明本專案 (`hand_tracking_npu`) 的工作內容：將 Google MediaPipe 的兩階段手部追蹤管線移植到 Novatek NT9869x / NS02201 (NVTS) 平台，並把神經網路推論完全卸載到板上的 AI3 / NPU。

---

## 3. 系統架構圖 (Architecture)

### 3.1 硬體配置

```
+--------------------------------------------------------+
|              Novatek NT9869x / NS02201 SoC             |
|                                                        |
|   +------------+     +-----------+     +-----------+   |
|   |  Cortex-A73|     |    ISP    |     |   AI3 /   |   |
|   |   (Linux)  |<--->|    VPE    |<--->|    NPU    |   |
|   +------------+     +-----------+     +-----------+   |
|         |                  |                 |         |
|         | HDAL             | videocap/proc   | vendor_ |
|         |                  |                 | ai3_*   |
+---------|------------------|-----------------|---------+
          |                  |                 |
   +------+------+     +-----+------+    +-----+------+
   | IMX335 CMOS |     |   HDMI Out |    |  nvt_model |
   |  Sensor     |     |   1080p60  |    |   .bin     |
   | 2592x1944   |     +------------+    +------------+
   +-------------+
```

- **感測器**：Sony IMX335，原生 2592x1944@30 fps；60 fps 模式因板上 INCK 不足無法啟用，30 fps 即為硬體上限。
- **顯示**：HDMI 1080p60 輸出 (mode 16)。
- **運算分工**：Cortex-A73 處理幾何 / 解碼 / 追蹤；ISP+VPE 負責 RAW12 -> NV12 與中心裁切縮放 (2592x1944 -> 1920x1080)；AI3 NPU 專責神經網路推論。

### 3.2 軟體層級

```
+-------------------------------------------------------------+
|                Zig App  (薄殼啟動器, main.zig)              |
|        app <mode> <bundle>.task  ->  mp_task_open/run/close |
+-------------------------------------------------------------+
                          |
                          v  (C-ABI: include/mp_task.h)
+---------------------------------------------------------------+
|     MediaPipe CalculatorGraph (third_party/mediapipe v0.10.35)|
|     由 mp_glue/mp_task.cc 載入 graph.binarypb 並執行           |
|                                                                |
|  Custom Calculators (mp_glue/calculators/):                    |
|   - HDALFrameSourceCalculator   (擷取 + 1080p 預覽 + 推論分流) |
|   - ImageToTensorCalculator     (192x192 RGB 給 palm)          |
|   - Nv12RotatedCropToTensorCalc (224x224 旋轉手部裁切)          |
|   - NpuInferenceCalculator      (model-agnostic NPU 呼叫)       |
|   - OverlayHdmiSinkCalculator   (結果快取 + 手勢分類)           |
|   - LiveHdmiPreviewSinkCalc     (30 fps HDMI 推送與 HUD)        |
|   - 內建 SSD 解碼、NMS、RectTransformation、LandmarkProjection |
+---------------------------------------------------------------+
                          |
                          v  (C-ABI: include/nvts_board_abi.h)
+-------------------------------------------------------------+
|     Zig 板載機制 (app/src/*.zig, export fn nvts_*)          |
|   cam.zig       : HDAL videocap RAW12 + videoproc NV12       |
|   display.zig   : HDMI videoout                              |
|   overlay.zig   : NV12 Y/UV 雙平面骨架/HUD 著色              |
|   npu.zig       : vendor_ai3_* 推論封裝，manifest 驅動       |
|   npu_quant.zig : Affine dequant、CHW/HWC 轉換、@Vector/NEON |
|   nv12_sample.zig: 旋轉雙線性 NV12->RGB 取樣 (Cortex-A73 SIMD)|
+-------------------------------------------------------------+
                          |
                          v
            HDAL / vendor_ai3 SDK (Novatek 原廠 C ABI)
```

### 3.3 兩階段模型邏輯與資料流 (核心)

```
[Live 30 fps 路徑 - 每張影格]

   IMX335 2592x1944 RAW12
        |
        v  videoproc (中心裁切 + 縮放)
   NV12 1920x1080 ---------------------------+
        |                                    |
        | (FRAME, 全速 30 fps)               |
        |                                    |
        v                                    |
   LiveHdmiPreviewSink                       |
   (繪製快取的最新 overlay -> HDMI Push)     |
                                             |
   [Inference 路徑 - 每 33 ms 一次]          |
        +------------------------------------+
        |
        v  nvts_cam_frame_to_rgb (192x192, 直接 NV12->RGB)
   [Palm Detector NPU]   192x192 RGB / NCHW
   -> SSD 錨框解碼 + 加權 NMS
   -> 7 個 palm keypoints -> rotated palm_rect
   -> RectTransformation (CROP_SCALE=2.2, CROP_SHIFT=-40)
   -> rotated hand_rect (NormalizedRect)
        |
        v  nvts_nv12_sample_to_tensor (旋轉雙線性, NEON @Vector)
   [Hand Landmark NPU]   224x224 RGB / NCHW
   -> 21 個 (x,y,z) 在裁切空間
   -> presence + handedness
   -> LandmarkProjection 回投至原始 1920x1080 影格
   -> 規則式手勢分類 (gesture_rules.h: open/fist/point/peace/thumbs)
        |
        v  (NVTS_RESULT 結果包)
   OverlayHdmiSinkCalculator (快取最新 21 點骨架 + 手勢標籤)
        |
        +--> LiveHdmiPreviewSink 下一張 FRAME 取用此快取繪製
```

關鍵設計不變量：

1. 手部 landmark 模型 **永遠不跑全畫面**，而是由 palm keypoints 建構的「旋轉矩形」裁切。
2. 裁切 / 投影幾何全部繫於 **原始影格座標** (1920x1080)，不繫於 detector tensor。
3. NPU 只跑神經推論；幾何、解碼、追蹤、繪製全部留在 Host / Media block。
4. 模型輸出在進入追蹤或繪製前，先轉為強型別 struct (`PalmDetection`、`HandLandmarks`)。

### 3.4 模型轉換管線 (Off-line)

```
   palm_detection_full.tflite      hand_landmark_full.tflite
   hand_landmark_full.tflite             (官方 MediaPipe)
        |                                       |
        v  tf2onnx + 插入 NCHW 輸入 transpose   |
   deploy.onnx (planar RGB / NCHW)              |
        |                                       |
        v  Novaic Gen-tool (容器化 linux/amd64) |
   nvt_model.bin  +  gen_config.txt + quant_input_output.json
        |
        v  scripts/gen_backend_manifest.py
   nvt_backend.json  (input layout / dtype / scale / hw_layout)
        |
        v  scripts/check_task_contract.py  (對照 SDK input_bin oracle)
        v  scripts/build_hand_landmark_task.py
   hand_landmark.task  =  graph.binarypb
                       +  nvt_backend.json
                       +  nvt_model.bin (palm + hand)
                       +  provenance/  (自我審計)
```

`nvt_backend.json` 為 NPU 端「單一真相來源」，包含 dims、dtype、layout (CHW/HWC)、resize 政策 (stretch/letterbox)、channel order、normalize scale，避免板上需要硬編碼任何預處理參數。

---

## 4. 實作技術與環境

### 4.1 核心框架與函式庫

| 類別 | 技術 |
|------|------|
| 模型來源 | Google MediaPipe Hands (`palm_detection_full.tflite`, `hand_landmark_full.tflite`) |
| 推論框架 | **MediaPipe CalculatorGraph v0.10.35** (pinned submodule, commit `f8ef212`)，靜態封裝為 `libmediapipe_framework.a` + `libmp_stock_calculators.a` |
| NPU 後端 | Novatek **AI3 / vendor_ai3** (`vendor_ai3_net_proc`)，透過 Zig 客製化 `NpuInferenceCalculator` 取代 MediaPipe 預設 `InferenceCalculator` |
| 模型轉換 | Novatek **Novaic / NvtAI Gen-tool + Sim-tool** (CNN30 v12 配置)，容器化在 `linux/amd64` runner 內執行 |
| 板載 HAL | Novatek **HDAL** (videocap/videoproc/videoout)，鏡像於 `lib/libhdal.{so,a}` |
| 系統語言 | **Zig 0.16.0** (薄殼啟動器 + 板載 C-ABI 機制 + 純 host-testable 量化數學) |
| 玻璃層 | **C++20** (`mp_glue/*.cc`，含客製 Calculator 與 bundle loader) 使用 `zig c++` + **libc++** 靜態連結，編譯目標 `aarch64-linux-gnu.2.34`、Cortex-A73 |
| 模型轉換腳本 | Python (`uv run`)，依賴 `tensorflow`、`tf2onnx`、`numpy` |
| 序列化 | Protobuf (`graph.binarypb`)、abseil、glog/gflags |
| SIMD 最佳化 | Zig `@Vector` -> AArch64 NEON 4-wide (見 `npu_quant.dequant16Into`、`nv12_sample`) |

### 4.2 開發 / 建置環境

- **Nix Flake** (`flake.nix`)：提供 Zig 0.16.0 / uv / ffmpeg / podman / cmake / protobuf 的可重現 shell。
- **建置目標**：`aarch64-linux-gnu.2.34`，`-mcpu=cortex_a73`，預設 `-Dprefer-static-libs=true` (HDAL/SDK 靜態連結，glibc 動態)。
- **SDK Docker**：`nt9869x-sdk-docker/` (SSH submodule，內含 NS02201 SDK slim tarball 與離線重現的工具鏈)。
- **`zig build` 流程**：
  - `zig build` — 預設板端 app。
  - `zig build -Dsdk-root=<sdk>` — 啟用 vendor_ai3 NPU 路徑。
  - `zig build -Dmp-archive-dir=lib/mp/board` — 連結預先建好的 MediaPipe 靜態庫。
  - `zig build deploy` — 將剩餘共享庫重命名 DT_NEEDED 並打包成 SD-card payload。
  - `zig build test-parity` — Host 端對照 Novaic Sim-tool 的 dequant bit-exact 驗證。
- **Host 模擬**：`nvts_mp_harness` 連結相同 graph，但以 `mp_glue/host/nvts_board_host_stubs.cc` 提供 Sim-tool dump 重播，可在 macOS / Linux x86_64 上跑完整 graph，不需要實體板。

### 4.3 板載驗證工具

```
scripts/check_prereqs.sh          模型 + SDK + 轉換產物前置檢查
scripts/run_novaic_tool.sh        容器化 Gen-tool / Sim-tool wrapper
scripts/rebuild_novaic_inputs.sh  從原始 mp4 重建校正集 + onnx
scripts/prepare_p3_live_test.sh   單張影像 NPU 板測 payload
scripts/prepare_p4_live_test.sh   完整 live HDMI 板測 payload
scripts/check_task_contract.py    .task bundle 預處理契約閘門
app board-smoke {cam,npu,cam-npu,cam-hdmi,cam-npu-live}  繞過 graph 的純板端煙霧測試
```

---

## 5. 成果展示 (Results)

### 5.1 模型轉換正確性 (Novaic Sim-tool, 12 張 hand_waving review set)

```
Palm Detector (192x192, INT16 NPU 量化)
  avg score = 0.9491   min = 0.9111
  pass >= 0.75 : 12/12   pass >= 0.90 : 12/12

Hand Landmark (224x224 旋轉裁切, INT16 NPU 量化)
  avg presence = 0.9773   min = 0.9228
  pass >= 0.75 : 12/12   pass >= 0.90 : 12/12
```

視覺證據：`build/review/hand_waving_full_pipeline_2026-05-31/final/` 內的
`overlay_contact_sheet.jpg` 與 `hand_waving_overlay.mp4`，呈現偵測框、旋轉裁切矩形、與投影後 21 點骨架。

### 5.2 板載延遲與吞吐 (P4 Live, Cortex-A73 + AI3)

| 階段 | P4 初版 (1080p RGB) | P4 fused NV12 路徑 (現行) |
|------|---------------------|---------------------------|
| `nvts_cam_frame_to_rgb` 全畫面 1920x1080 | p50 **43.45 ms** / p95 47.84 ms | (移除) |
| Palm 路徑 NV12 -> 192x192 RGB | n/a | **約 1.4 ms** |
| Hand 路徑旋轉 224x224 取樣 | (走全畫面 RGB) | 與 palm 路徑融合，無全畫面材化 |
| Palm 推論 (`vendor_ai3_net_proc`) | ~4-5 ms | ~4-5 ms |
| Hand 推論 (`vendor_ai3_net_proc`) | ~4-5 ms | ~4-5 ms |
| Per-tick 總計算 | ~55 ms | **~10 ms** |
| Result cadence (100 ms 推論週期) | p50 277 ms | **p50 ~100 ms** |
| HDMI 顯示 | 60/30/10/1 fps 抖動 | **穩定 30 fps** (軟體 pacer >= 33.33 ms) |
| 30 fps / 33 ms frame budget | 未達標 | **達標** |

### 5.3 板載里程碑通過證據

```
P2B  board-smoke 全綠      (cam, npu, cam-npu, cam-hdmi)
P3   P3_IMAGE_EXIT:0       單張影像端到端 graph (palm + hand) 真機 AI3
P4 host   harness 完成      graph init + Sim-tool dumps + 結果包
P4 board  P4_LIVE_EXIT:0    `prepare_p4_live_test.sh` 持續 HDMI 骨架
```

NPU 板端輸出 dim 對齊預期：palm `36288 + 2016`、hand `63 + 1 + 1 + 63`。

### 5.4 額外功能

- **規則式手勢分類** (`gesture_rules.h`)：以 palm-axis 手指幾何在 host 端推論 `open / fist / point / peace / thumbs / unknown`，並以 HUD 標籤即時顯示，零 NPU 成本、零模型轉換。
- **HDMI HUD**：顯示推論 FPS、顯示 FPS (EMA)、NPU 延遲、追蹤狀態、手勢標籤、handedness。

---

## 6. 結論與未來展望

### 6.1 研究貢獻總結

1. **完整移植 MediaPipe Hands 至 Novatek NPU 平台**：保留 MediaPipe `CalculatorGraph` 的真實執行語意，只替換推論後端為 `vendor_ai3`，避免重新實作 SSD 解碼、NMS、Rect Transformation、Landmark Projection 等龐雜幾何邏輯。
2. **明確的軟硬體分工**：神經運算上 NPU、幾何運算留在 A73、影像搬移交給 ISP/VPE，並以 `nvts_*` Zig C-ABI 嚴格分隔，使 MediaPipe 與板載細節彼此獨立可測。
3. **自描述 `.task` bundle**：以 `nvt_backend.json` 作為 NPU 預處理單一真相來源，並透過 `check_task_contract.py` 在建置期對照 SDK `input_bin` oracle，根除 HWC/CHW 與 letterbox/stretch 兩類常見部署偏差。
4. **效能突破**：發現並消除全畫面 RGB 轉換瓶頸 (43 ms -> 1.4 ms)，將 per-tick 計算從 55 ms 降至 ~10 ms，在 30 fps / 33 ms 預算內穩定運行。
5. **可重現的轉換管線**：Nix + uv + Podman 容器化 Novaic 工具鏈，配合 12 張 hand-waving review set，使模型量化評分在不同主機上可重現 (avg presence 0.9773、min 0.9228)。

### 6.2 已知限制

- IMX335 60 fps 感測器模式因板上 INCK 不足無法啟用，30 fps 是當前硬體天花板。
- `P5LiteRoiReuseCalculator` (前次裁切矩形 reuse) 的初版 loopback/gate 設計會讓 `inference_video` 在 palm 偵測前就阻塞 overlay，目前已 compile-in 但停用，等待 P5 穩定化重新設計。
- 手勢分類為規則式幾何，對斜視角與遮擋的辨識率有限。
- 校正集仍是 12 張選擇性 review frames；非每張原始影格皆能通過 0.75 門檻。

### 6.3 未來展望

| 方向 | 內容 |
|------|------|
| **P5 完整版** | 多手追蹤 association、presence gate loopback 穩定化、palm suppression、ROI reuse 重設計，逼近上游 MediaPipe Hands 在 live 模式下的 tracking 行為。 |
| **學習型手勢分類** | 預留的 `GestureClassifierCalculator` 接點可接入第三顆小型分類網路，在 NPU 上跑 INT8/INT16 推論，取代當前的幾何規則。 |
| **NPU 性能變體** | `prepare_novaic_perf_variants.py` 已備好 `[performance/mode]=1` 與 `shrink_en=1` 兩個實驗變體，待 Gen-tool profile + Sim-tool 正確性 + 板端 `NVTS_NPU_STAGE` 驗證後晉升為正式設定。 |
| **更多場景** | 將 Novaic 校正集擴展至多光照、多膚色與雙手互動，提升量化後模型的泛化。 |
| **應用整合** | 在現有 HDMI overlay 之上開放 `nvts_npu_*` 與 landmark 結果包供應用層 (例如手語、AR、IoT 控制) 訂閱，並評估與板上 ISP 動態白平衡及 AE 同步以強化夜間表現。 |
| **多模型協同** | 探索同一 AI3 NPU 上同時掛載手部 + 人臉或姿態模型，並以 MediaPipe sub-graph 結構維持資料流可組合性。 |

整體而言，本專案完成了「真 MediaPipe graph + Novatek NPU 推論 + Zig 板載 C-ABI + 30 fps live HDMI」的端到端驗證；後續發展空間集中在更穩健的追蹤、可學習的高階語意 (手勢/意圖)、以及在不犧牲 30 fps 的前提下擴展到更豐富的多模型場景。


