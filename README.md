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

輸入 `/start` 或 `/menu` 後，模式、提示詞／素材、生成、任務控制、歷史和排隊按鈕都會直接顯示；片長、解析度和 steps 收納在「⚙️ 片長／解析度／steps」頁面，電腦溫度、ComfyUI 啟停／狀態和 Bot 重啟則收納在「🖥️ 電腦／ComfyUI／Bot」頁面；「📊 查看／刷新生成進度」固定放在主面板最底部。解析度現在包括 1152×640、1280×736、1344×768 三個實驗檔位；在 10GB 顯存上較容易 OOM，建議先用 448×256 至 736×416。設定會保存到 Windows 使用者資料夾，下次啟動會讀回；提示詞和最近一張圖片也可以保留。

Bot 不會自動固定或置頂任何 Telegram 訊息；控制面板使用輸入欄旁較短的常駐「🎛️ 面板」快捷按鈕。每次 Bot 發送狀態、錯誤或完成訊息時都會重新附上快捷鍵；新訊息很多時，按下這個按鈕會在聊天最底部重新發出控制面板。Bot 會把原生命令選單重設為預設狀態，斜線指令仍可手動輸入。Telegram 客戶端本身的原生選單位置不能由 Bot API 改成 Reply Keyboard 按鈕。

模型選擇：

- `🧠 MiniMax H3`：H3 FL2VA INT8 + LightX2V 正式 8 步 Turbo LoRA 工作流，支援長片、Motion Context、恢復和音訊接續；這是目前唯一支援的模型。LoRA strength 為 `1.0`，建議先用 `8 steps`。

也可以使用 `/model h3` 查看目前模型，`/h3` 是快捷指令。

圖片生視頻：按「🖼 圖片生視頻」，直接發一張圖片，再貼提示詞，最後按「生成影片」。也可以給圖片加 caption，caption 會直接當作提示詞。Bot 會把圖片上傳到 ComfyUI，使用 H3 Turbo 的 I2VA 首幀工作流。

文字生視頻：按「📝 文字生視頻」，直接輸入提示詞即可。

## ✨ 一句話生成腳本（本機 LLM，全程留在 Telegram）

不想自己寫長腳本時，按主面板的「✨ 一句話生成腳本（本機 LLM）」，或輸入 `/make`。
給一句話加上秒數，Bot 會用**你自己的本機 llama.cpp**（`127.0.0.1:19092`）生成完整、可直接生成 H3 的腳本：

```text
/make 60秒 下雨的東京街頭，一個女生錯過末班車
/make 30秒 貓在窗邊發呆，午後陽光
/make 2分鐘 賽博龐克機車追逐
```

也可以只打 `/make`，再依提示輸入想法。沒寫秒數就用目前面板上的片長。中文或英文的想法都可以，
輸出是英文動作描述（H3 對英文動態描述更穩定），約 20–60 秒完成。

生成後的草稿會直接顯示在 Telegram，並附上這些按鈕：

- **✅ 採用並生成影片** — 採用腳本、自動設定片長，並立刻開始生成
- **📥 只採用（回面板）** — 只採用腳本，回到面板等你按「🚀 生成影片」
- **🤖 指令修改（AI）** — 用一句話叫 AI 改，例如「運鏡再慢一點」
- **✏️ 手動編輯** — 直接貼上你要的完整內容
- **➕ 追加內容** — 在現有內容後面補一段
- **🔄 重新生成** — 同一個想法再產一版
- **↩️ 回到上一版** — 退回前一次修改
- **❌ 放棄** — 丟掉草稿

### 可以一直改下去

草稿不是一次性的。**要改幾次都可以**，每次修改都會自動保留上一版：

```text
第 1 版 → ✏️ 手動編輯 → 第 2 版 → 🤖 指令修改 → 第 3 版 → ➕ 追加 → 第 4 版 …
                                                             ↑ ↩️ 隨時退回
```

- **AI 修改會自我修正**：改完先用 Bot 自己的驗證器檢查，格式不對就帶著錯誤訊息要求重寫。
  **只有通過驗證的結果才會取代草稿**，所以改壞了也不會弄丟原本好的腳本。
- **手動編輯不擋你**：即使改到格式不通過也會保留（你可能正在分幾步修長時間軸），
  但面板會**明確標示格式未通過、直接生成會被拒絕**，並提示可退回上一版。
- **版本上限 30 版**，超過會自動淘汰最舊的。
- **換新想法會清空歷史**，避免退回到不相關的舊腳本。

每一版都會顯示「第 N 版」與即時格式檢查結果（通過時附上切出的鏡頭數）。


### 腳本語言（預設簡體中文）

生成的腳本**預設是簡體中文**。可隨時切換：

```text
/lang         查看目前語言
/lang zh      簡體中文（預設）
/lang en      English
```

或按面板上的「🌐 腳本語言」按鈕切換，設定會保存，重啟後仍有效。
也可用環境變數 `MINIMAX_SCRIPT_LANG=en` 改預設值。

切換語言會同時切換時間軸標題的寫法（簡中 `开头／结尾`，英文 `開頭／結尾`），
兩種寫法 Bot 都接受。場景中的可見文字（例如日本車站告示牌上的「終電」）會保留原語言，
這是官方指南的要求，不算混用。

> H3 的訓練語料以英文為主，因此英文的動態與運鏡描述通常最穩定；
> 簡體中文完全可用，實測 6 種片長（12–90 秒）皆一次通過驗證、0 個英文殘留、0 個繁體字。

### 🎬 長短鏡頭交替（不再固定秒數）

時間軸由 Bot 計算，**幕長採不規則節奏交替**，不會每幕都是 10 秒／20 秒：

```text
60 秒 → 4、9、6、12、7、5、10、7 秒（8 幕，長短差距 8 秒）
```

節奏取自一組預先驗證過的長度循環，保證：從 0 開始、首尾相接無缺口、
每幕都在 2–15 秒內、總和精確等於你指定的片長（2 秒到 30 分鐘都成立）。
想回到等長，程式呼叫端可用 `script_timeline_skeleton(total, varied=False)`。

### 📐 鏡頭語言（已內建）

系統提示已加入運鏡規範與生活寫實規則，因此每次生成的腳本都會：

- 運鏡寫成自然散文（`鏡頭以小幅度緩慢推向她手裡的杯子`），不堆標籤
- 完整運鏡＝**類型＋幅度＋速度**，只在有意義時才寫幅度與速度
- **強制多樣化**：同一幕不得沿用上一幕的運鏡，整片至少混用三種景別、兩種角度、三種運動
- 預設**真實生活流**：手機拍攝感、手持微晃、自然光寫出來源與方向、具體可指認的場景細節、
  自然表演、環境音為主；避免電影級調色與廣告感
- 禁止抽象情緒詞（溫馨、治癒、震撼）與未來式敘述

完整方法論、官方 12 種運鏡對照表、三種可複用結構模板與關鍵詞庫見
`H3-中文短視頻提示詞指南.md`。

### 📝 自訂指令（直接在 Telegram 改，存檔即生效）

想加入自己的風格規則，不必碰程式碼，也不必開檔案總管。按面板的「📝 自訂指令」，
或輸入 `/prompt_file`，就會看到目前內容與四個按鈕：

| 按鈕 | 作用 |
|---|---|
| ✏️ 編輯（整段取代） | 直接把完整內容打在 Telegram 裡送出 |
| ➕ 追加一條規則 | 新的規則接在現有內容後面，不覆蓋 |
| 🗑 清空 | 清掉所有自訂指令 |
| ↩️ 還原上一版 | 改錯了就退回前一次內容（每次儲存都會保留上一版） |

送出後 Bot 會回報「✅ 已儲存…立即生效」，並附上**之後每次生成實際會送出的內容**，
所以你當下就能確認寫進去了。

**改完立即生效，永遠不需要重啟 Bot** —— 這個檔案在每次生成時重新讀取。

規則會被附加到系統提示的後段，因此可以調整風格，但**不會**覆蓋結構與時間軸規則
（那些由 Bot 計算，改不動也不該改）。建議一次只加 1–3 條，方便判斷哪一條造成變化。

也可以用文字編輯器直接改這個檔案，效果完全相同：

```text
E:\MiniMax-H3-Telegram\runtime\bot\script_prompt.txt
```

- 以 `#` 開頭的行是註解，不會送給模型（用 TG 編輯時不寫註解也沒關係）
- 空白輸入會被忽略，不會清掉既有規則
- 超過 4000 字元會截斷並回報，不會弄壞生成
- 檔案不存在或內容全為註解 → 等同沒有自訂指令，不會報錯

環境變數：`MINIMAX_SCRIPT_PROMPT_FILE` 可改路徑、`MINIMAX_SCRIPT_PROMPT_MAX_CHARS` 可改上限。

實測：寫入「不得出現任何人類對白；配樂必須是單一獨奏大提琴」後生成，產出內容
明確寫出「她……沒有說話」，配樂全部為獨奏大提琴，且一次通過格式驗證。

### 為什麼產出的格式一定可用

Bot 生成後會**用自己同一套解析器**驗證（`build_long_video_plan`）。若模型寫出缺口、重疊、
或單幕超過 15 秒，Bot 會把**確切的驗證錯誤**回饋給模型要求它重寫，最多重試 3 次
（`MINIMAX_SCRIPT_GEN_ATTEMPTS`）。因此「顯示給你看的草稿」與「Bot 真正能生成的腳本」是同一件事，
不會出現看起來沒問題、按下生成卻被拒絕的情況。三次都失敗就明確報錯，不會無限重試。

腳本會依目前模式自動調整規則：T2VA 寫完整時間軸；I2VA 只寫動作、不重述圖片外觀；
FL2VA 只寫首尾幀之間的中間過程；Ref2VA 只鎖人物外貌、場景明寫在腳本內。
短片（≤15 秒）則產出單段散文，不套時間軸。

生成腳本需要本機 LLM 在線。如果它沒開，Bot 會自動啟動並等待就緒；
真正開始生成影片時 Bot 仍會照常先關閉 LLM 釋放顯存，所以兩者不需要同時佔用顯存。

### 🛡 顯存保護：不會在 ComfyUI 佔用時硬啟 LLM

ComfyUI 跑完**不會自動釋放模型**。在這台 2×20GB 的機器上，約 35GB 的 llama-server
根本塞不下剩下的空間，硬啟動不會得到可用的 LLM，只會 OOM。

因此 `start_llama_process()` 會先檢查：**ComfyUI 在線且可用顯存不足時直接拒絕**，
並回報實際數字與解法：

```text
ComfyUI 正在佔用顯存，現在啟動本機 LLM 會直接 OOM。

可用顯存：21,006 MB（共 40,960 MB）
啟動 LLM 約需：34,000 MB
GPU0 剩 19,018MB、GPU1 剩 1,988MB

ComfyUI 跑完不會自動釋放模型，請先擇一：
  • 按面板的「🛑 關閉 ComfyUI」釋放顯存，再重試
  • 或先在 ComfyUI 裡卸載模型（Free model and node cache）
```

這道檢查放在 `start_llama_process()` **內部**，所以**所有入口一次覆蓋**：
面板按鈕、`/llm_start`、`/llm_restart`、腳本生成器。而 `restart_llm_after_job()`
會先停 ComfyUI 再呼叫它，不受影響。

判定用 **nvidia-smi 的全卡快照**，不是 ComfyUI 自己的回報 —— 因為 Bot 只把單一
GPU 曝露給 ComfyUI（`CUDA_VISIBLE_DEVICES`），它的 `/system_stats` 看不到另一張卡。

安全設計：ComfyUI 離線 → 放行；ComfyUI 開著但閒置 → 放行；探測不到顯存 → 放行
（不因探測失敗誤擋）；需求上限自動夾在總顯存的 85%（設定錯誤不會讓 LLM 永久無法啟動）。

環境變數：`MINIMAX_LLM_START_GUARD=0` 停用、`MINIMAX_LLM_START_MIN_FREE_MB`
調整所需餘量（預設 34000）。

可用環境變數：`MINIMAX_SCRIPT_GEN=0` 停用此功能、`MINIMAX_SCRIPT_GEN_TIMEOUT`（預設 600 秒）、
`MINIMAX_SCRIPT_GEN_TEMP`（預設 0.85）、`MINIMAX_SCRIPT_GEN_ATTEMPTS`（預設 3）、
`MINIMAX_SCRIPT_GEN_MAX_TOKENS`（預設 6000）、`MINIMAX_SCRIPT_GEN_RETRY_TOKENS`（預設 16000）。

### 時間軸由 Bot 計算，不交給模型

長片的每一行時間標題（`開頭（0-10秒）：` 等）都是 Bot **先算好**再要求模型照抄的，
模型只負責填每一幕的內容。片長會被切成等長、連續、且每幕不超過 15 秒的段落
（例如 60 秒＝6 幕各 10 秒；45 秒＝5 幕各 9 秒）。

這樣做是因為先前用「範例」教模型寫時間軸時，範例本身前後矛盾（前面的幕已經到 55 秒，
結尾範例卻寫 50-60 秒），模型會照抄出重疊的時間軸。改由 Bot 計算後，時間軸在模型動筆前
就已經是正確的，這類錯誤從根本上消失。

本機 LLM 的思考（reasoning）與正文共用同一個 token 預算。若思考把預算用光、正文為空，
Bot 會自動以更大的預算重試一次；仍失敗才報錯並說明原因。**不建議關閉思考**：
實測關閉後模型會盲目照抄標題、產出格式錯誤的腳本，思考對遵守格式是必要的。


面板上的「🌡 查看電腦溫度」會讀取 NVIDIA GPU 溫度、GPU 使用率和 VRAM；CPU 溫度只有在 Windows/主機板提供感測器時才會顯示。選擇超過 15 秒的長片後，可以開啟「🔌 長片完成後關機」；影片合併並成功傳回 Telegram 後，系統會在 60 秒後關機。倒數期間可以按「🛑 取消即將關機」，或輸入 `/cancel_shutdown`。

長片會先解析提示詞時間軸，再將場景拆成最多 8 秒的短鏡頭，而不是固定把整篇提示詞重複送進四個 15 秒任務。每鏡完成後，Bot 會擷取最後畫面作為下一鏡的 I2VA 首幀；H3 會延續音訊參考。最後 FFmpeg 會加入 0.12 秒音畫轉場並維持原定總片長。

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

原有 `GLOBAL`／`SEGMENT N` 格式仍然支援。Bot 會讀取所有連續的 `SEGMENT 1`、`SEGMENT 2`、`SEGMENT 3`……，不再按固定 15 秒限制段落數；每個 `SEGMENT` 會再拆成最多 8 秒鏡頭。總片長會平均分配到你提供的 SEGMENT 數量，因此 120 秒可以寫 8 段、12 段或更多段，但每段仍須至少 2 秒、最多 15 秒：

```text
GLOBAL:
固定人物外貌、服裝、場景、光線和畫面風格。

SEGMENT 1:
只描述第 1 段最多 15 秒的動作。

SEGMENT 2:
只描述第 2 段最多 15 秒的延續動作。
```

SEGMENT 編號必須由 1 開始並連續遞增；例如寫到 `SEGMENT 12` 時，`SEGMENT 1` 至 `SEGMENT 12` 都要存在。影片總長仍受目前 30 分鐘上限、硬碟空間、生成時間和 Telegram 檔案大小限制。

H3 仍然使用原本的 FL2VA INT8，不會強行加入需要 Ref2VA 模型的人物參考輸入。人物主要依靠全局描述和上一鏡尾幀接力保持。首幀和音訊參考仍屬模型條件而非硬性鎖定，因此可能有輕微漂移。

生成中的面板提供「⛔ 中止」「⏸ 暫停」和「▶️ 播放／繼續」。中止會打斷目前 ComfyUI 工作；暫停會在目前短鏡頭完成後生效，播放／繼續會生成下一鏡。因為 ComfyUI 不保存採樣中的中間狀態，單段短片不能安全地在採樣中途暫停。

長片生成中還提供「🎬 預覽已完成片段」按鈕（或輸入 `/preview`）：Bot 會把目前已完成的鏡頭合成成一段影片，立即傳回 Telegram 給你預覽，而長片生成**不會中斷**，其餘鏡頭會繼續生成。例如生成 2 分鐘長片、目前做到第 4 鏡時，按下預覽會收到前 3 鏡的合併影片；也可以連續預覽，每次都會包含最新完成的鏡頭。目前鏡頭仍在生成中時，該鏡不會被納入本次預覽。

長片現在會在 E 槽的 `E:\MiniMax-H3-Telegram\runtime\bot\long_checkpoints` 保存檢查點。每完成一鏡就會記錄已完成 MP4、下一鏡編號和 Motion Context latent；如果第 3 鏡因顯存不足失敗，按失敗訊息的「🔁 從第 3 鏡繼續」，或輸入 `/resume_long`，只會重試第 3 鏡，不會重做第 1、2 鏡。恢復前 Bot 會先要求 ComfyUI 釋放暫存顯存。

長片生成現在也會自動處理顯存不足：例如從 `0.4 MP` 開始時，第 3 鏡 OOM，Bot 會保留前兩鏡，將第 3 鏡改成 `0.3 MP` 重試；再失敗就依序降到 `0.2 MP`、`0.1 MP`，直到成功或已經沒有更低檔位。後續鏡頭會沿用成功的較低解析度，不會重新生成前面的鏡頭。最後合併時會把不同檔位統一成原本的影片尺寸；這是畫面尺寸統一，不等於 AI 放大，較低檔位的細節仍以實際生成結果為準。每次降級會在 Telegram 報告，完成資訊也會列出降級記錄。

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
/preview
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
/prompt_help
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

秒數會自動轉成 H3 的有效影格數；15 秒會使用 362 frames。完成後 Bot 會傳送含聲音的影片（官方 SaveVideo 節點把音訊直接併入同一個 MP4）。

## 模型與節點

### 模型（`models/` 資料夾）

| 類別 | 檔案 | 大小 | 用途 |
|---|---|---|---|
| UNET（主模型） | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.5 GB | T2VA / I2VA / FL2VA 生成 |
| UNET（Ref2VA） | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 19.5 GB | Ref2VA 參考素材生成 |
| UNET（Hybrid） | `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors` | 19.5 GB | Ref2VA 預設（FL2VA 品質 + Ref2VA 條件） |
| CLIP（文字編碼） | `qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors` | ~32 GB | 提示詞 → conditioning |
| LoRA（Turbo） | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 1.8 GB | 8 步加速（official v1.0） |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | — | 影片 latent ↔ 像素 |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | — | 音訊 latent ↔ 波形 |
| Latent Upscaler | `minimax_h3_latent_upscaler_3d_fp16.safetensors` | — | 兩段式半解析度 → 全解析度 |
| SeedVR2（放大） | `seedvr2_3b_int8_convrot.safetensors` | — | 可選影片放大（1080p/2K） |

### ComfyUI 節點

#### 官方核心節點（ComfyUI-Turbo 內建）

| 節點 | 用途 |
|---|---|
| `VAELoader` × 2 | 載入 video / audio VAE |
| `CLIPLoader` | 載入 Qwen3VL 文字編碼器 |
| `UNETLoader` | 載入 H3 diffusion 模型 |
| `LoraLoaderModelOnly` | 載入 8-step Turbo LoRA（strength 1.0） |
| `MiniMaxH3ImageToVideo` | T2VA / I2VA / FL2VA 生成（含 first/last frame） |
| `MiniMaxH3ReferenceToVideo` | Ref2VA 生成（ref_images / ref_videos / ref_audios） |
| `KSamplerSelect` (`res_multistep`) | 多步降噪採樣器 |
| `BasicScheduler` (`simple`) | 噪聲調度（denoise 1.0 或 0.5） |
| `BasicGuider` | 模型 + conditioning → guider |
| `SamplerCustomAdvanced` | 實際執行採樣 |
| `RandomNoise` | 隨機種子 |
| `VAEDecode` | Video latent → 像素幀 |
| `VAEDecodeAudio` | Audio latent → 音訊波形 |
| `CreateVideo` | 合併影片 + 音訊 → 影片物件（24 fps） |
| `SaveVideo` | 輸出同步 MP4（H.264 + AAC） |
| `LoadImage` / `LoadVideo` / `LoadAudio` | 載入用戶上傳的素材 |
| `GetVideoComponents` | 拆出影片幀 + 內嵌音訊 |

#### 自訂節點（`custom_nodes/`）

| 套件 | 節點 | 用途 |
|---|---|---|
| **ComfyUI-H3-Motion-Context** (v0.3.1) | `MiniMaxH3MotionContext` | 長片接續：把上鏡 latent pin 到新鏡頭部 |
| | `MiniMaxH3MotionContextTrim` | 裁掉 pinned 頭部（避免重複幀/音訊） |
| | `MiniMaxH3MotionContextSaveLatent` | 保存本鏡 AV latent 給下一鏡 |
| | `MiniMaxH3MotionContextLoadLatent` | 載入上鏡 AV latent |
| **h3-av-latent-bridge** | `H3AVLatentSeparate` | 拆 AV NestedTensor → video latent + audio latent |
| | `H3AVLatentJoin` | 合併 video latent + audio latent → AV NestedTensor |
| **Comfyui_Minimax_h3_latent_Upscaler** | `MinimaxH3LatentUpscaler3D` | 3D latent 空間放大（無 VAE round-trip） |

### 工作流結構

```
[VAE/CLIP/UNET/LoRA loaders] → [生成節點] → [Sampler chain]
    → [VAEDecode + VAEDecodeAudio] → [CreateVideo] → [SaveVideo]

長片：+ [MotionContext Load/Process/Trim/Save] 鏈
兩段式：+ [LatentSeparate → Upscaler → LatentJoin → Stage2 Sampler]
```

### 環境要求

| 項目 | 最低 | 建議 |
|---|---|---|
| GPU | NVIDIA 20 GB VRAM（RTX 3080/3090） | 24 GB（RTX 4090）可跑更高解析度 |
| RAM | 32 GB | 64 GB |
| ComfyUI | ComfyUI-Turbo fork 0.31.0+（`--base-directory` 模式） | — |
| Python | 3.11+ | 3.13 |
| PyTorch | 2.12+ cu130 | — |
| FFmpeg | 6.0+（含 libx264 + AAC） | — |
| 作業系統 | Windows 10/11 | — |

### ComfyUI 啟動參數

```
python main.py \
  --base-directory <models+custom_nodes 目錄> \
  --listen 127.0.0.1 --port 8191 \
  --lowvram --use-sage-attention \
  --output-directory <輸出> --input-directory <輸入> \
  --disable-auto-launch
```

環境變數 `CUDA_VISIBLE_DEVICES` 自動選取顯存最充裕的 GPU。

## 已知限制（2026-09-04 官方節點迁移後）

- **drive_audio fallback**：長片在 H3 Motion Context 節點不可用時，會改用「上一鏡尾幀 + `<Audio 1>`」的尾幀接續；在官方核心節點下若沒有 Motion Context，`<Audio 1>` tag 沒有對應的參考音訊，該鏡會退化為原生音訊（不承接上一鏡的音訊）。Motion Context 預設可用，此情況罕見；若實際遇到再處理（候選方案：fallback 時把上一鏡音訊當 ref_audio 餵 R2V，但目前長片與參考素材互斥）。
- **T8 包已移除**：`E:\Comfy\ComfyUI\ComfyUI\custom_nodes\minimax-h3-audio-T8`（288 節點，Bot 已 0 引用）已删除。備份在本資料夾 `minimax-h3-audio-T8_backup_20260904.zip`（13.2 MB）；若要還原，解壓回 `custom_nodes\` 即可，ComfyUI 重啟後節點重新註冊。
