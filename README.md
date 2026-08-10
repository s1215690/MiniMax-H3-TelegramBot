# MiniMax H3 Telegram Bot

這是一個在 Windows 上控制本機 ComfyUI MiniMax H3 Turbo 工作流的 Telegram Bot。
它可以從 Telegram 設定文字生視頻、圖片生視頻、片長、解析度、steps 和提示詞，生成完成後把 MP4 傳回 Telegram。

本專案只包含 Bot 和啟動腳本，不包含模型、VAE、ComfyUI、個人輸入圖片、輸出影片或 Telegram Token。

## 功能

- 文字生視頻（T2VA）和首幀圖片生視頻（I2VA）
- 5、10、12、15 秒短片，以及自訂總片長
- 長片自動解析時間軸、拆成 5–8 秒鏡頭、尾幀接力並用 FFmpeg 轉場合併
- 解析度、步數模式和提示詞按鈕（保留 4V/8A、4V/12A，另有真正 8V、10V 實驗模式）
- GPU 溫度、使用率和 VRAM 查詢按鈕
- 長片完成並傳回 Telegram 後自動關機，可在倒數期間取消
- ComfyUI 啟動、停止、重啟和生成進度查詢
- 長片生成中的中止、鏡頭間暫停和播放／繼續按鈕
- H3 原片先回傳，再用 Telegram 按鈕選擇 SeedVR2 1080p、2K 或保留原片
- SeedVR2 放大保留原片音訊；超過約 8 秒會自動啟用 temporal chunk，降低長片顯存壓力
- Turbo 工作流（固定使用 `--lowvram` 顯存管理）
- Windows 登入後自動啟動 Bot
- Token 只從 Windows 使用者環境變數讀取

## 先決條件

1. Windows 10/11、Python 3.10 或更新版本，以及可執行的 FFmpeg。
2. 已安裝 ComfyUI 和 MiniMax H3 T8 自訂節點：
   `MiniMaxH3AudioConditioningT8`、`MiniMaxH3DualClockSamplerT8` 等。
3. 已準備相容的 H3 模型、Qwen3-VL CLIP、Video VAE、Audio VAE 和 Turbo LoRA。
4. BotFather 建立的 Telegram Bot，以及自己的 Chat ID。

模型檔案名稱可以透過環境變數覆寫；請不要把模型檔案提交到這個 Repository。

## 安裝

```powershell
git clone https://github.com/s1215690/MiniMax-H3-TelegramBot.git
cd MiniMax-H3-TelegramBot
py -3 -m pip install -r requirements.txt
```

先設定 ComfyUI 路徑。最少需要設定以下變數；路徑請換成自己的安裝位置：

```powershell
[Environment]::SetEnvironmentVariable('MINIMAX_COMFY_DIR', 'D:\ComfyUI\ComfyUI-Turbo', 'User')
[Environment]::SetEnvironmentVariable('MINIMAX_COMFY_BASE_DIR', 'D:\ComfyUI\ComfyUI', 'User')
[Environment]::SetEnvironmentVariable('MINIMAX_COMFY_PYTHON', 'D:\ComfyUI\ComfyUI\.venv\Scripts\python.exe', 'User')
```

然後雙擊 `Configure-MiniMax-H3-Telegram.cmd`，輸入新的 Bot Token 和 Chat ID。
Token 輸入框會隱藏內容，Token 會保存到 Windows 使用者環境變數，不會寫入 Repository。

設定後請開一個新的終端機，再雙擊 `Start-MiniMax-H3-Telegram.cmd`。

## 環境變數

| 變數 | 用途 |
|---|---|
| `MINIMAX_TELEGRAM_BOT_TOKEN` | Telegram Bot Token，必須只放在本機環境變數 |
| `MINIMAX_TELEGRAM_CHAT_ID` | 允許使用 Bot 的 Chat ID |
| `MINIMAX_COMFY_DIR` | ComfyUI Turbo 執行目錄，內含 `main.py` |
| `MINIMAX_COMFY_BASE_DIR` | ComfyUI 模型和 custom nodes 的 base directory |
| `MINIMAX_COMFY_PYTHON` | 執行 ComfyUI 的 Python |
| `MINIMAX_T8_API_TEMPLATE` | T8 API 工作流 JSON；預設使用 `workflow/dual_clock_4step_api.json` |
| `MINIMAX_SEEDVR2_API_TEMPLATE` | SeedVR2 放大工作流 JSON；預設使用 `workflow/seedvr2_3b_int8_upscale_video_api.json` |
| `MINIMAX_COMFY_OUTPUT` | ComfyUI output 目錄 |
| `MINIMAX_COMFY_INPUT` | ComfyUI input 目錄 |
| `MINIMAX_COMFY_PORT` | ComfyUI API Port，預設 `8191` |
| `MINIMAX_FFMPEG` | FFmpeg 可執行檔路徑；不設定時使用 PATH 中的 `ffmpeg` |
| `MINIMAX_NVIDIA_SMI` | `nvidia-smi` 路徑；不設定時自動尋找 NVIDIA 標準安裝位置 |

模型檔名也可以覆寫：

```text
MINIMAX_VIDEO_VAE
MINIMAX_AUDIO_VAE
MINIMAX_CLIP
MINIMAX_UNET
MINIMAX_LORA
```

## Telegram 指令

```text
/start 或 /menu       開啟按鈕控制面板
/prompt                輸入提示詞
/image                 切換到圖片生視頻
/text                  切換到文字生視頻
/progress              查看生成進度
/pause                 暫停長片（目前分段完成後生效）
/resume 或 /play       繼續長片生成
/status                查看 Bot 狀態
/temperature           查看 GPU／CPU 溫度
/cancel_shutdown       取消已排程的自動關機
/comfy_status          查看 ComfyUI 狀態
/comfy_start           啟動 ComfyUI
/comfy_restart         重啟 ComfyUI
/comfy_stop            關閉 ComfyUI
/cancel                取消目前生成
```

步數按鈕中的 `V` 是影片採樣步數，`A` 是音訊採樣步數。`⚡ 4V / 8A` 和
`⚡ 4V / 12A` 是原本的 MultiRate Turbo 模式；`🧪 真 8 steps` 和
`🧪 真 10 steps` 才會真正把影片採樣提高到 8 或 10 步。真步數模式會明顯較慢，
而且仍然使用 4-step Turbo LoRA，屬於測試用途，不保證一定比 4V 模式清晰。

也可以使用指令直接設定一段短片：

```text
/gen 736 416 8 10
cinematic bright daylight scene with smooth camera movement and clear synchronized sound
```

總片長超過 15 秒時，Bot 不會再把整份長提示詞重複送給每次生成。它會先解析自然時間軸或 `SEGMENT N`，再把較長場景拆成最多 8 秒的短鏡頭。上一鏡最後畫面會成為下一鏡首幀；第二鏡開始也會把第一鏡音訊作為 `<Audio 1>` 參考，以 `reference_only` 延續音樂風格、節奏和環境聲。最後 FFmpeg 會加入 0.12 秒音畫交叉轉場並維持原定總片長。

建議直接使用自然時間軸。時間必須從 0 秒連續覆蓋到目前選擇的總片長；有缺口、重疊或只寫到 50 秒卻選擇 60 秒時，Bot 會在生成前指出錯誤：

```text
【60秒反詐騙短片】

開頭（0-5秒）：
黑底警示標題，沉重低音音樂開始。

第一幕（5-15秒）：
同一名女生在家看到可疑招聘廣告。
她猶豫後按下應聘按鈕。

第二幕（15-25秒）：
她拖着行李抵達機場，畫面逐漸轉為灰暗。

第三幕（25-40秒）：
接頭人收走護照和手機，鐵閘關上。

第四幕（40-50秒）：
她被迫坐在電腦前輸入詐騙訊息。

結尾（50-60秒）：
畫面轉黑，出現反詐騙警示。
```

原有 `GLOBAL`／`SEGMENT N` 格式仍然支援；每個最多 15 秒的 `SEGMENT` 也會再拆成短鏡頭：

```text
GLOBAL:
固定人物外貌、服裝、場景、光線和畫面風格。

SEGMENT 1:
只描述第 1 段最多 15 秒的動作。

SEGMENT 2:
只描述第 2 段最多 15 秒的延續動作。
```

純粹貼上一段沒有時間軸的長提示詞會被拒絕，避免模型把開頭重演多次。首幀和音訊參考仍屬模型條件而非硬性鎖定，因此長片仍可能有輕微人物或音樂漂移。現有 FL2VA INT8 模型不會強行套用 Ref2VA 人物參考，以免使用錯誤模型和增加 10GB 顯存負擔。

生成中的面板提供「⛔ 中止」「⏸ 暫停」和「▶️ 播放／繼續」。中止會打斷目前 ComfyUI 工作；暫停會在目前短鏡頭完成後生效，播放／繼續會生成下一鏡。因為 ComfyUI 不保存採樣中的中間狀態，單段短片不能安全地在採樣中途暫停。

生成期間按「📊 查看／刷新生成進度」，進度會直接顯示在同一個控制面板文字最底部；面板會原地更新，不會另外建立一條進度訊息。

面板上的「🌡 查看電腦溫度」會讀取 NVIDIA GPU 溫度、GPU 使用率和 VRAM；CPU 溫度只有在 Windows/主機板提供感測器時才會顯示。選擇超過 15 秒的長片後，可以開啟「🔌 長片完成後關機」；影片合併並成功傳回 Telegram 後，系統會在 60 秒後關機。倒數期間可以按「🛑 取消即將關機」，或輸入 `/cancel_shutdown`。

## 長片連貫與音訊接續

長片預設會嘗試使用 `ComfyUI-H3-Motion-Context`。它會把上一段的影片、尾端影像 context，以及上一段儲存的配對 AV latent 一起交給下一段，並裁走用來接續的 22 個影格；因此不只是重複提交同一條 prompt，也不只是把上一段音訊當作聲音參考。

請先在 ComfyUI 的 `custom_nodes` 安裝 [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)，然後重啟 ComfyUI。Bot 會透過 `/object_info` 自動檢查節點；節點不存在時會自動回退到穩定的「上一段尾幀＋音訊參考」模式。

如果要手動回退穩定模式，設定環境變數後重啟 Bot：

```powershell
[Environment]::SetEnvironmentVariable('MINIMAX_H3_LONG_CONTINUITY', 'stable', 'User')
```

長片中按「中止」時，Bot 會停止目前 ComfyUI 任務，並把已完成的分段先合併成 `_partial.mp4` 傳回 Telegram；未完成的分段不會被假裝成已完成。由於 H3 每個分段仍是獨立採樣，Motion Context 只能改善接續，不能保證跨很多段後人物、音樂和音效完全不漂移。

## 可選 SeedVR2 放大

H3 影片完成後會先把原片傳回 Telegram，接著顯示三個按鈕：`放大到 1080p`、`放大到 2K` 和 `保留原片`。只有按下放大按鈕才會提交另一個 ComfyUI 任務；原片不會被刪除。SeedVR2 工作流會從原片讀取影格、FPS 和音訊，再輸出 H.264 MP4。

這個功能使用 ComfyUI 0.31 或更新版本內建的 SeedVR2 節點，不需要另外安裝 SeedVR2 custom node。請把以下模型放到 ComfyUI 對應目錄：

```text
models/diffusion_models/seedvr2_3b_int8_convrot.safetensors
models/vae/seedvr2_ema_vae_fp16.safetensors
```

10GB 顯存建議先選 1080p；2K 需要更多顯存和時間，失敗時保留原片並重新選 1080p。放大期間可以用「中止」或 `/cancel` 打斷目前任務。

## 顯存配置

本專案固定使用 Turbo 工作流和 `--lowvram`。這只是 ComfyUI 的顯存管理方式，不會切換成其他模型；Telegram 面板不再提供極限顯存模式。

## 安全提醒

- 不要把 Bot Token、Chat ID、`.env`、日誌、資料庫、輸入圖片或輸出影片提交到 Git。
- 如果 Token 曾經貼到聊天、截圖或公開網站，請立即在 `@BotFather` 撤銷並重新建立。
- Bot 只接受設定的 Chat ID，ComfyUI 預設只監聽 `127.0.0.1`。

## 授權

本專案採用 MIT License。MiniMax H3 模型、ComfyUI 和自訂節點各自遵循其原作者授權條款。
