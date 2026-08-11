# MiniMax H3 Turbo Telegram 控制器

## 安裝一次

1. 到 Telegram 的 `@BotFather` 撤銷曾經外洩的舊 Token，再產生新 Token。
2. 雙擊 `Configure-MiniMax-H3-Telegram.cmd`，輸入新 Token 和自己的 Chat ID；不用直接開 `.ps1`。
3. 雙擊 `Start-MiniMax-H3-Telegram.cmd`，Bot 會先啟動；ComfyUI 可以先關閉。
4. Telegram 發 `/start` 或 `/menu`，按鈕可選文字生視頻或圖片生視頻；再選短片 5/10/12/15 秒、長片 30/60/120 秒、解析度和 steps。
5. 如要登入 Windows 後自動啟動 Bot，雙擊 `Install-MiniMax-H3-Telegram-Autostart.cmd`。

Token 只會存放在 Windows 使用者環境變數，不會寫入這個資料夾。

Bot 只接受設定好的 Chat ID。生成時如果 ComfyUI 未運行，Bot 會以目前 10GB 顯存設定自動啟動：
`127.0.0.1:8191`、Turbo 工作流和 `--lowvram`，並等待 API 就緒後才送出工作。

Bot 本身會保持運行；ComfyUI 連續 5 分鐘沒有 Bot 任務、而且 ComfyUI 佇列為空時會自動關閉，以釋放顯存。之後按面板的「▶️ 啟動 ComfyUI」或輸入 `/comfy_start` 即可重新啟動。可用環境變數 `MINIMAX_COMFY_IDLE_SHUTDOWN_SECONDS` 調整秒數，設為 `0` 可停用。

為避免佔用 C 槽，MiniMax 的輸入、輸出、設定、續接圖片、日誌和 ComfyUI 狀態固定放在 `E:\MiniMax-H3-Telegram`。

## Telegram 用法

## 按鈕模式

輸入 `/start` 或 `/menu` 後，直接按面板上的模型、模式、片長、解析度和 steps。解析度現在包括 1152×640、1280×736、1344×768 三個實驗檔位；在 10GB 顯存上較容易 OOM，建議先用 448×256 至 736×416。設定會保存到 Windows 使用者資料夾，下次啟動會讀回；提示詞和最近一張圖片也可以保留。

模型：

- `🧠 MiniMax H3 Turbo`：H3 FL2VA INT8 + Turbo 工作流，支援長片、Motion Context、恢復和音訊接續。

可以使用 `/model h3` 或 `/h3` 查看／切換目前模型；目前 Bot 只保留 MiniMax H3 Turbo。

圖片生視頻：按「🖼 圖片生視頻」，直接發一張圖片，再貼提示詞，最後按「生成影片」。也可以給圖片加 caption，caption 會直接當作提示詞。Bot 會把圖片上傳到 ComfyUI，使用 H3 Turbo 的 I2VA 首幀工作流。

文字生視頻：按「📝 文字生視頻」，直接輸入提示詞即可。

面板上的「🌡 查看電腦溫度」會讀取 NVIDIA GPU 溫度、GPU 使用率和 VRAM；CPU 溫度只有在 Windows/主機板提供感測器時才會顯示。選擇超過 15 秒的長片後，可以開啟「🔌 長片完成後關機」；影片合併並成功傳回 Telegram 後，系統會在 60 秒後關機。倒數期間可以按「🛑 取消即將關機」，或輸入 `/cancel_shutdown`。

長片會先解析提示詞時間軸，再將場景拆成最多 8 秒的短鏡頭，而不是固定把整篇提示詞重複送進四個 15 秒任務。每鏡完成後，Bot 會擷取最後畫面作為下一鏡的 I2VA 首幀，並延續上一鏡的音訊參考。最後 FFmpeg 會加入 0.12 秒音畫轉場並維持原定總片長。

建議使用自然時間軸，並由 0 秒連續寫到選擇的總片長：

```text
【60秒短片】

開頭（0-5秒）：
描述開場。

第一幕（5-15秒）：
描述下一個動作。

第二幕（15-25秒）：
描述延續動作。

第三幕（25-40秒）：
描述劇情轉折。

第四幕（40-50秒）：
描述結果。

結尾（50-60秒）：
描述收尾。
```

如果時間軸有缺口、重疊，或只寫到 50 秒卻選了 60 秒，Bot 會在生成前指出。純粹貼一段沒有時間軸的長提示詞會被拒絕，避免每個鏡頭重新演繹開頭。

原有 `GLOBAL`／`SEGMENT N` 格式仍然支援，每個 `SEGMENT` 也會再拆成最多 8 秒鏡頭：

```text
GLOBAL:
固定人物外貌、服裝、場景、光線和畫面風格。

SEGMENT 1:
只描述第 1 段最多 15 秒的動作。

SEGMENT 2:
只描述第 2 段最多 15 秒的延續動作。
```

H3 使用 FL2VA INT8 和 Turbo LoRA，不會強行加入需要 Ref2VA 模型的人物參考輸入。人物主要依靠全局描述和上一鏡尾幀接力保持。首幀和音訊參考仍屬模型條件而非硬性鎖定，因此可能有輕微漂移。

生成中的面板提供「⛔ 中止」「⏸ 暫停」和「▶️ 播放／繼續」。中止會打斷目前 ComfyUI 工作；暫停會在目前短鏡頭完成後生效，播放／繼續會生成下一鏡。因為 ComfyUI 不保存採樣中的中間狀態，單段短片不能安全地在採樣中途暫停。

長片現在會在 E 槽的 `E:\MiniMax-H3-Telegram\runtime\bot\long_checkpoints` 保存檢查點。每完成一鏡就會記錄已完成 MP4、下一鏡編號和 Motion Context latent；如果第 3 鏡因顯存不足失敗，按失敗訊息的「🔁 從第 3 鏡繼續」，或輸入 `/resume_long`，只會重試第 3 鏡，不會重做第 1、2 鏡。恢復前 Bot 會先要求 ComfyUI 釋放暫存顯存。

長片生成現在也會自動處理顯存不足：例如從 `0.4 MP` 開始時，第 3 鏡 OOM，Bot 會保留前兩鏡，將第 3 鏡改成 `0.3 MP` 重試；再失敗就依序降到 `0.2 MP`、`0.1 MP`，直到成功或已經沒有更低檔位。後續鏡頭會沿用成功的較低解析度，不會重新生成前面的鏡頭。最後合併時會把不同檔位統一成原本的影片尺寸；這是畫面尺寸統一，不等於 AI 放大，較低檔位的細節仍以實際生成結果為準。每次降級會在 Telegram 報告，完成資訊也會列出降級記錄。

Telegram Bot API 對 Bot 上傳影片有 50 MB 限制。Bot 現在會在生成完成後自動檢查檔案大小：小於安全上限就直接傳送；超過時先用 FFmpeg 壓縮，仍然過大就自動切成多段 MP4 逐段傳送。原始影片不會被覆蓋，壓縮和分段只會產生暫存檔，傳送後自動清理。

完整長片完成後也可以按面板的「📼 延續上一條長片」，輸入要新增的秒數和尾端提示詞。H3 會使用原片最後一鏡的影片和 AV latent 接續，再把原片與新增部分合併回傳。也可以使用 `/extend 30`，再貼上延續提示詞。只有由這個 Bot 保存過檢查點的完整長片能使用精確 Motion Context 延續；沒有檢查點的外部 MP4 不會被假裝成同樣的 latent 接續。

面板的「📚 歷史長片」會列出所有可用 checkpoint。選擇某個 ID 後可以查看詳情，再按「從這條影片延續新故事」；也可以使用 `/extend <ID> 30` 指定歷史長片。舊的完整 `long_<ID>` 輸出資料夾會在首次開啟歷史列表時自動匯入，會讀取原片的分辨率、分段、尾端影片和可用的 Motion Context latent。沒有完整合併片的舊資料夾不會被列入可延續列表。

面板的「🧾 故事排隊」可以一次加入多個獨立故事。按「加入故事」後貼上多段提示詞，用獨立一行的 `---` 分隔；Bot 會保存每個故事當時的片長、解析度、steps 和圖片輸入，前一個完成後自動開始下一個。可用 `/queue` 查看、`/queue_add` 加入、`/queue_start` 開始、`/queue_clear` 清空。排隊資料保存於 E 槽，Bot 重啟後仍會保留；如果上一條影片正在等待 SeedVR2 放大選擇，先按「保留原片」或完成放大，隊列才會接續。

生成期間按「📊 查看／刷新生成進度」，進度會直接顯示在同一個控制面板文字最底部；面板會原地更新，不會另外建立一條進度訊息。

面板上的「🔄 重啟 Bot」會先取消目前的生成（有的話），送出確認訊息後，Bot 會在幾秒內自動結束並重新啟動，不需要重新登入 Windows；重新啟動後請再按一次 `/start` 或 `/menu`。

## 指令模式（備用）

先發參數，再發提示詞：

```text
/gen 864 480 12 15
```

長片也可以用指令：

```text
/long 864 480 12 60
```

Bot 回覆等待提示詞後，直接貼一段或多段文字即可。也可以一則訊息完成：

```text
/gen 864 480 12 15
Bright photorealistic Japanese restaurant scene with two adult women eating dinner.
```

提示詞太長時，不要貼到 Telegram 輸入框；直接把提示詞另存為 `.txt` 或 `.text` 檔案後傳給 Bot。Bot 會讀取整個檔案、保留換行和 `GLOBAL`／`SEGMENT` 時間軸，然後保存成目前提示詞。建議使用 UTF-8 編碼，檔案上限為 512 KB；如果之前已上傳圖片，傳 TXT 不會清除圖片模式。

其他指令：

```text
/status
/cancel
/pause
/resume
/resume_long
/extend 30
/history
/extend long_5d8276f01375 30
/queue
/queue_add
/queue_start
/queue_clear
/image
/text
/long 864 480 12 60
/comfy_status
/comfy_start
/comfy_restart
/comfy_stop
/bot_restart
/temperature
/cancel_shutdown
/help
```

秒數會自動轉成 H3 的有效影格數；15 秒會使用 362 frames。完成後 Bot 會優先傳送含聲音的 `-audio.mp4`。
