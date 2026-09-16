# 🌙 YUPI 工作流（NSFW Ref2VA + FastH3 6步）— 說明

一條**獨立**的生成路徑，跟原本的 Turbo 生成（`build_workflow()` /
`dual_clock_multirate_api.json`）**零共用、零修改**。原本的模式、長片、排隊、
放大全部照舊。

**YUPI 現在是一個「輸入模式」**（跟 T2VA／I2VA／FL2VA／Ref2VA 一樣）：選它只會
切換模式，**不會馬上生成**；要等你上傳參考圖、輸入提示詞，再按「🚀 生成影片」。

YUPI 預設走 **6 步 FastH3 版**（原本 20 步版仍保留在 `yupi_nsfw_api.json`）：

| | 🌙 YUPI（預設） | （備用）20 步版 |
|---|---|---|
| 工作流檔 | `yupi_fast_api.json` | `yupi_nsfw_api.json` |
| LoRA 鏈 | UNET → AfterMidnight → **FastH3 6-step** | UNET → AfterMidnight |
| Steps | **6** | 20 |
| Sampler / Scheduler | euler / beta | euler / beta（相同）|
| 輸出前綴 | `MiniMaxH3/YUPI_FAST` | `MiniMaxH3/YUPI_NSFW` |
| 速度 | **約 2–2.8×**（6 步 vs 20 步）| 基準 |

## 它跟原本 Turbo 的差別

| 項目 | 原本 Turbo | YUPI |
|---|---|---|
| UNET | FL2VA | **Ref2VA**（`minimax_h3_ref2va_pruned_int8_convrot.safetensors`）|
| LoRA | `minimax_h3_fl2v_turbo_8step` | **AfterMidnight NSFW + FastH3 6-step** |
| Sampler | `res_multistep` | **euler**（LoRA 硬性要求，否則音訊會爆）|
| Scheduler | `simple` | **beta**（同上）|
| 素材 | 可文字/圖/首尾/參考 | **必須一張參考圖**（Ref2VA 錨定角色）|
| 解析度 | 1344×768 | **跟隨面板選擇**（預設值 1152×640；長片自動拆成短鏡頭）|
| Steps | 4–20 可選 | 固定 6（FastH3 蒸餾要求）|

## 準備（一次性）

1. **AfterMidnight NSFW LoRA** → 放到 `E:\Comfy\ComfyUI\ComfyUI\models\loras\`
   檔名：`AfterMidnight_ref2va_h3_sexytime_rank64-v1.2.safetensors`
   下載：https://huggingface.co/SexGod1979/AfterMidnight-MiniMax-H3-NSFW
   （備份：sasimi/ 、sisniha/ 、nutboyai/minimaxH3-NSFW）
   ⚠️ 別用假的 `Blackfrost-Research/MINIMAX-H3-NSFW`（hash 跟官方一樣＝貼牌）。

2. **（YUPI_FAST 用）FastH3 6-step LoRA** — 需要自己轉一次檔：
   - 下載 adapter（**只取 `dense-datafree/adapter_model.safetensors` ~1.4GB**）：
     https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA
   - 下載轉換腳本：
     https://github.com/NikoDemon80/ComfyUI-FastH3-Lora-Converter
   - 執行（CPU，5–10 秒）：
     ```
     python lora_convert_h3.py adapter_model.safetensors ComfyUI\models\loras\fasth3_6step.safetensors "<你的 Ref2VA UNET 路徑>"
     ```
     範例基底：`E:\Comfy\ComfyUI\ComfyUI\models\diffusion_models\minimax_h3_ref2va_pruned_int8_convrot.safetensors`
   - 產出檔名必須是 **`fasth3_6step.safetensors`**（或設環境變數
     `MINIMAX_YUPI_FAST_LORA` 改路徑），放進 `models\loras`，重啟 ComfyUI。

3. **參考圖** → 放到 `E:\Comfy\ComfyUI\ComfyUI\input\yupi_reference.png`
   （Bot 按鈕會用你暫存的圖；這個只是手動載入時的預設）

4. **Ref2VA UNET** 要已就位：
   `E:\Comfy\ComfyUI\ComfyUI\models\diffusion_models\minimax_h3_ref2va_pruned_int8_convrot.safetensors`

## 用 Bot 按鈕（跟其他模式一樣的流程）

面板 → 按 **🌙 YUPI工作流（6步）**（Ref2VA 按鈕下方那一列）。
**按下去只會切換模式，不會馬上生成**，接著：

1. 上傳一張**參考圖**（直接傳圖片即可，會存成參考素材）
2. 按「✍️ 輸入／更換提示詞」或 `/prompt` 貼上提示詞
3. 按 **🚀 生成影片** → 才開始生成

生成行為：
- 片長 ≤15 秒 → 直接生成一個鏡頭
- 選 60 秒等長片 → 自動依時間軸拆成短鏡頭、接續上一鏡尾幀，最後合併
- 沒參考圖就按生成 → Bot 會提醒你先上傳

## 解析度與步數

- **解析度跟隨面板選擇**（`res:{寬}x{高}` 按鈕）。可選階梯：
  448×256 / 512×288 / 608×352 / 736×416 / 864×480 / 960×544 / **1152×640** /
  1280×736 / 1344×768。
  10GB 顯存 + Ref2VA int8 建議 **864×480 ~ 960×544**；1152×640 是偏重的設定。
- **步數固定 6**（FastH3 蒸餾就是為 6 步調的，跑更多步反而劣化）。寫在工作流
  JSON 的 node 13。

## 長片接續：YUPI 現在也吃 Motion Context

YUPI 長片（>15 秒）現在跟一般長片一樣使用 **H3 Motion Context**：

| 鏡頭 | 做法 |
|---|---|
| 第 1 鏡 | AfterMidnight Ref2VA + 你上傳的參考圖（錨定角色）|
| 第 2 鏡起 | 接續上一鏡的 **AV latent + 尾幀**（`MiniMaxH3MotionContext`），參考圖改用上一鏡尾幀 |

- **切換按鈕**：主選單 →「⚙️ 生成參數」→「🔗 長片接續：Motion Context ✅ ／
  尾幀接續 ✅」（按下即切換，會記住設定）。環境變數
  `MINIMAX_H3_LONG_CONTINUITY` 只是**首次啟動的預設值**。
- 需要 4 個節點：`MiniMaxH3MotionContext` / `…Trim` / `…SaveLatent` / `…LoadLatent`
- 節點不存在或 layout 不相容 → 自動退回「尾幀接續」，並在 Telegram 告知
- YUPI 的 Motion Context 節點用 **id 31–36**（股票工作流用 15–20，但 YUPI_FAST 的
  node 15 已被 FastH3 LoRA 佔用，不能共用）
- 每鏡多生成 22 幀 context 頭部再裁掉（約 +0.92 秒），音訊窗口 24 幀

## 用 ComfyUI 手動載入（備援）

把 `yupi_nsfw_api.json` 或 `yupi_fast_api.json` 拖進 ComfyUI，改 node 14 的
LoadImage、填 node 6 的 prompt，Run 即可。（Bot 按鈕走同一份 JSON，只是自動
注入提示詞＋參考圖＋隨機 seed。）

## 驗證加速是否真的生效

跑完看 ComfyUI log：應該出現 **`208 patches attached`**（FastH3 有載入）。
若顯示 `0 patches attached`，代表 LoRA 沒生效（通常是 base 檔名不對或沒重啟）。

## 改動清單（對原 Bot 零侵入）

YUPI 圖（`yupi_nsfw_api.json` / `yupi_fast_api.json`）：
- 常數：`YUPI_API_TEMPLATE`、`YUPI_OUTPUT_PREFIX`、`YUPI_BUTTON`、`YUPI_TASK_TYPE`、
  `YUPI_FAST_API_TEMPLATE`、`YUPI_FAST_OUTPUT_PREFIX`、`YUPI_FAST_LORA_NAME`
  （可用環境變數 `MINIMAX_YUPI_FAST_LORA` 覆寫）
- 函式：`load_yupi_workflow(fast)`、`yupi_generation_config()`、
  `configure_yupi_workflow()`、`yupi_fast_lora_name()`
- 方法：`run_yupi_generation(chat_id, fast)`、`_yupi_worker()`、`run_yupi_segment()`
- `JobState` 新增 `yupi_fast: bool`（checkpoint 也會保存，長片續跑不會走錯版本）

YUPI 變成輸入模式（最新）：
- `INPUT_MODE_YUPI = "yupi"` 加入 `INPUT_MODES`
- 新函式 `is_ref2va_like()`：Ref2VA / YUPI 共用參考素材上傳流程
- `mode:yupi` callback：**只切換模式**（不再立即生成）
- `start_selected_generation()`：YUPI 模式 → `run_yupi_generation(fast=True)`
- 生成按鈕、`media_done`、選單按鈕、狀態列、照片上傳都認得 YUPI 模式
- 按鈕只剩一顆：**🌙 YUPI工作流（6步）**（`mode:yupi_fast` 已移除）
- **未改**：`build_workflow()`、`run_segment()`、`start_generation()`、
  `dual_clock_multirate_api.json`、任何現有模式/生成邏輯
- 備份：`MiniMax-H3-Telegram-Bot.py.bak_yupimode_20260909_133453`

## 注意

- LoRA 硬要求 **euler + beta**，工作流已設定，別改回 res_multistep/simple。
- FastH3 是 **preview release**，官方註明困難動作/極細節可能低於基線。
- **AfterMidnight + FastH3 疊加未經第三方驗證**：建議 A/B 比對一次（同 prompt、看
  NSFW 品質與音訊）；若音訊出問題，可改用 20 步版（`yupi_nsfw_api.json`）。
- YUPI 一定要參考圖，沒圖 Bot 會叫你補。
- 授權灰區：H3 Community License 禁止繞過護欄；這只是本地部署，風險自己評估。
