# MiniMax H3 Turbo Telegram 控制器

用 Telegram 控制本機 ComfyUI 跑 MiniMax H3 影片生成（配自架 llama.cpp 寫腳本），全程離線、素材唔出機。

- 🎬 短片／長片：最長 30 分鐘，自動分段生成＋合併（帶音訊）
- 🧩 四種輸入：文字、圖片首幀、首尾幀、參考素材（Ref2VA）
- ✨ 一句話生成腳本：本機 LLM 或 Command Code 雲端（DeepSeek v4.1）隨時切換（`/scriptllm`），可反覆修改到滿意
- 🔁 尾帧接力：完成一條片後一鍵用最後一兩帧做新參考圖，LLM 讀住上一段劇本續寫（同一人物、同一場景）
- 🔗 自動接力：一次過設定「每段時長清單（5 8 5 10）或總秒數（48）」＋想法，bot 自動生成短段 → 抽尾帧 → LLM 續寫 → 再生成，最後合併成一段（`/chain 5 8 5 10 想法`）
- 🎞 長片支援：Motion Context 音畫接續、檢查點續傳、OOM 自動降級、邊生成邊預覽
- 📚 故事排隊、歷史長片延續（`/extend`）
- 🖥 顯存自動管理：生成時自動開 ComfyUI，閒置自動關；LLM 同 ComfyUI 交替佔用，唔會互相搶爆顯存

## 快速開始

1. 在 `@BotFather` 產生新 Token，記下自己的 Chat ID
2. 雙擊 `Configure-MiniMax-H3-Telegram.cmd` 輸入兩者（Token 只存 Windows 使用者環境變數，唔會寫入資料夾）
3. 雙擊 `Start-MiniMax-H3-Telegram.cmd`（ComfyUI 可以先唔開）
4. Telegram 打 `/start` 或 `/menu`
5. （可選）`Install-MiniMax-H3-Telegram-Autostart.cmd` 設定登入後自動啟動

所有資料固定放 `E:\MiniMax-H3-Telegram\`：輸入、輸出、設定、參考圖、日誌、ComfyUI 狀態。
生成時若 ComfyUI 未運行，Bot 會自動以 `127.0.0.1:8191`、`--lowvram` 啟動並等 API 就緒；連續 5 分鐘無任務會自動關閉釋放顯存。

## 常用操作

- 面板按鈕為主：模式 → 提示詞／素材 → 「🚀 生成影片」
- `/make 60秒 想法`：一句話生成腳本（亦可用面板「✨」）；完成後可「✅ 採用並生成」「🤖 指令修改」「✏️ 手動編輯」等
- `/gen 864 480 12 15`：指令模式（先發參數，再貼提示詞）
- `/lang zh|en`：腳本語言（預設簡體中文）
- `/scriptllm local|cc`：切換腳本 LLM（本機 llama.cpp / Command Code 雲端）
- `/scripttemplate adult|general`：切換腳本模板（成人版 / 一般版〔非成人〕）
- 面板「📝 自訂指令」：加自己的風格規則，改完即時生效（檔案：`runtime\bot\script_prompt.txt`）
- `/help`：全部指令

## 尾帧接力（重點功能）

每條片生成完，Bot 會自動抽最後一兩帧傳回 Telegram，附上按鈕：

- ✅ **清空參考圖，用這兩帧做新參考圖** —— 之後用「✨ 一句話生成腳本」時，LLM 會收到【接續規則】＋【上一段劇本】，寫出同一人物、同一場景、由尾帧動作直接接落去的新一段
- ❌ 保留原本參考圖

清空參考圖、或重新上傳第一張素材（開新一套）＝結束接續，唔會錯誤延續。

## 模型（`models/`）

| 類別 | 檔案 | 用途 |
|---|---|---|
| UNET（預設） | `minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors` | fused 單檔（已內含加速），4 步＋SLA 稀疏注意力 |
| UNET（classic） | `minimax_h3_fl2va_pruned_int8_convrot` / `minimax_h3_ref2va_*` 等 | 舊路線，配 turbo LoRA 使用 |
| CLIP | `qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors` | 文字編碼 |
| VAE | `minimax_h3_video_vae_fp16` / `minimax_h3_audio_vae_fp32` | 畫面／音訊 |
| 其他 | 3D latent upscaler、SeedVR2（可選影片放大） | 兩段式放大／後期 |

## 節點需求

- 官方核心節點：ComfyUI-Turbo fork（H3 生成、Sampler、VAE、SaveVideo 等）
- 自訂節點：`ComfyUI-H3-Motion-Context`（長片接續）、`h3-av-latent-bridge`（AV latent 拆分）、`Comfyui_Minimax_h3_latent_Upscaler`（兩段式放大）
- 選用：`Comfyui-Sparse-SLA-Attention`（fused profile 會自動插入）

## 環境

GPU 20GB+（本機實測：2×RTX 3080 20GB，單卡生成）｜RAM 32GB+｜Windows 10/11｜ComfyUI-Turbo 0.31+｜Python 3.11+｜FFmpeg 6+｜llama.cpp 本機 LLM（`127.0.0.1:19092`）

## 檔案

- `MiniMax-H3-Telegram-Bot.py` —— Bot 本體（全部功能）
- `Configure / Start / Restart / Install-*-Autostart.cmd` —— 安裝、啟動、重啟
- `H3-中文短視頻提示詞指南.md` —— 提示詞寫作指引
- `YUPI_WORKFLOW_README.md` —— YUPI 工作流說明
- `雙卡調查報告.md` —— 雙 GPU 調查記錄

## 已知限制

- 首幀／音訊參考屬軟條件，長片接續可能有輕微漂移
- 高解析度（>0.4MP）在 20GB 卡上容易 OOM，Bot 會自動降級重試（較低檔位細節以實際生成為準）
- 超長片可能受 Telegram 檔案大小限制
- 舊版詳細 README 保留在 git 歷史（`git show 422a1ef:README.md`）
