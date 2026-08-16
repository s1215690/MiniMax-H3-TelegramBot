# MiniMax H3 Telegram Bot — 提示詞寫作模板

> 結合官方 MiniMax-H3 Prompt Guide（chronological description、四要素、自然運鏡散文）與本 Bot 的實際運作方式（T2V / I2V / FL2VA / Ref2VA、時間軸長片、native 音畫聯合生成、Motion Context 接續）整理。

---

## 一、通用原則（所有模式適用）

1. **按時間順序寫** — 官方指南的核心要求是「詳細的、按時間順序的描述」。從第一秒寫到最後一秒，動作要有先後，不要只堆形容詞。
2. **四要素齊全**：
   - **主體**：外貌、年齡、髮型、服裝、配飾 — 要「可見」的特徵，不是背景故事
   - **場景**：地點、時間、光線方向、天氣、環境細節
   - **動作**：具體動態（怎麼動、動多快、表情如何變化）。避免「跳舞」這種模糊詞 → 寫「她舉起左手轉了半圈，裙擺揚起」
   - **運鏡**：寫成自然散文（"the camera slowly pushes in"）。H3 不接受 `Camera direction:` 這種標籤格式
3. **聲音一起寫** — Bot 使用 native 音畫聯合生成（H3 同時生成畫面與聲音）。音樂類型與節奏、環境音、對白、音效都要描述，而且要和畫面放在同一條時間線上。
4. **動作描述建議用英文** — H3 訓練語料以英文為主，英文的動態/運鏡描述更穩定。中文可行，但建議至少動作句用英文。
5. **不要手寫官方媒體標籤** — `<Picture N>`、`<Video N>`、`<Audio N>` 由 Bot/ComfyUI 自動編號。手寫標籤在嚴格模式（strict prompt tags）下會被檢查，素材對不上會直接報錯。
6. **不要寫「接下來會發生」** — H3 只生成當下的動作；寫未來事件會讓畫面搶跑或混亂。長片時把未來劇情放進對應時間段。

---

## 二、短片：文字生視頻（T2V）

**結構**：`[主體] + [場景] + [按秒推進的動作] + [運鏡] + [聲音]`

**範例（15 秒）：**

```
An elderly Japanese sushi chef in a white uniform and headband stands behind
a wooden counter in a small warmly-lit sushi bar at night. He picks up a fresh
salmon fillet, holds it up to the camera with a proud smile, then places it on
the cutting board and slices it into even pieces with a long knife, the blade
glinting under the warm light. The camera starts on his face in a close-up,
then slowly pulls back to a medium shot showing the whole counter. Soft jazz
plays quietly in the background, mixed with the rhythmic sound of the knife
tapping the board and the low murmur of two guests.
```

---

## 三、短片：圖片生視頻（I2V）

發一張圖片，**caption 直接當提示詞**。不要重複描述圖片裡已看得見的靜態外觀，重點寫：

1. 這個人／物**接下來做什麼動作**
2. 鏡頭怎麼動
3. 發出什麼聲音

**範例 caption：**

```
The woman in the image slowly turns her head to look at the camera and smiles,
raising the coffee cup for a small toast. The camera gently zooms in on her
face. Soft morning cafe ambience with light chatter; the clink of her cup is
clear and close.
```

---

## 四、短片：FL2VA（首幀 + 尾幀）

已上傳首幀和尾幀，提示詞只寫**中間過程**：從首幀的狀態如何演變成尾幀的狀態。開頭與結尾姿勢不用重複描述（幀已鎖定）。

**範例：**

```
Starting from the first frame, the dancer slowly lifts her arms and spins once
to the left, her dress flowing outward; she completes the turn and settles into
the exact pose of the last frame, facing the window. The camera stays still at
eye level. A slow waltz plays, with footsteps and the rustle of fabric audible.
```

---

## 五、短片：Ref2VA（參考素材）

素材是「參考」不是「重播」。寫清楚：借什麼（外貌／風格／動作／聲音），然後做什麼**新**動作。素材外觀不用整段重寫，一句「keeping the exact appearance of the reference」即可。

**範例（參考 1 張人物圖 + 1 段跳舞影片）：**

```
Keeping the exact appearance of the reference image, the character walks
across a rooftop at sunset and stops at the edge, looking out over the city.
The camera follows from behind, then swings around to a front medium shot.
Wind sounds, distant traffic, and a slow electronic beat fade in.
```

---

## 六、長片：自然時間軸（30／60／120 秒）★ 推薦

**格式規則（Bot 解析器硬性要求）：**

- 標題格式：`場景名（開始秒-結束秒）：`，例如 `開頭（0-5秒）：`（全／半形括號、冒號皆可，分隔符可用 `-`、`～`、`至` 等）
- 必須從 **0 秒**開始、**連續**寫到所選總片長 — 有缺口、重疊、或只寫到 50 秒卻選 60 秒，Bot 會拒絕
- 每一幕 **2–15 秒**（Bot 會自動把每幕拆成 ≤8 秒的鏡頭）
- 第一個時間標題**之前**的文字 = **全局設定**（人物／場景／風格／音樂），每個鏡頭都會自動附上 → 人物外貌只需要寫一次
- 每一幕的動作只描述該幕，結尾自然過渡到下一幕

**範例（60 秒）：**

```text
【60秒短片】

人物與風格（第一個時間標題前的文字 = GLOBAL，每鏡自動附上）：
A woman in her 30s with short black hair, a beige trench coat and a red scarf
is the only main character throughout. Setting: an old European train station
on a rainy evening, wet platforms reflecting warm station lights. Visual style:
cinematic, muted colors with warm highlights. Music: a slow melancholic piano
piece plays through the whole film.

開頭（0-5秒）：
She stands under the station clock holding a small suitcase, looking down the
empty platform. Rain falls steadily. The camera slowly pushes in from a wide
shot.

第一幕（5-15秒）：
A train arrives with a long whistle, steam and light flooding the platform.
She watches the doors open but nobody gets off. She takes a step forward, then
hesitates.

第二幕（15-25秒）：
She boards the train and walks down the empty aisle, glancing at each seat.
Her footsteps echo. The camera follows her from behind.

第三幕（25-40秒）：
She finds a seat by the window. Through the wet glass she sees a man with an
umbrella on the platform, looking at her. She presses her hand against the
glass.

第四幕（40-50秒）：
The train starts moving. The man on the platform slowly raises his hand. She
sits back, closes her eyes and smiles faintly.

結尾（50-60秒）：
The train accelerates into the rain. The camera lingers on the empty platform
and the man's umbrella, then fades to black. The piano piece finishes with a
single soft note.
```

---

## 七、長片：GLOBAL / SEGMENT 格式（舊格式，仍支援）

```text
GLOBAL:
人物外貌、服裝、場景、光線、畫面風格、音樂 — 每個 SEGMENT 都會自動附上。

SEGMENT 1:
第 1 段動作（2–15 秒）。

SEGMENT 2:
第 2 段動作。
```

**規則：**

- `SEGMENT` 編號必須從 **1** 開始**連續遞增**（寫到 `SEGMENT 5` 就必須有 1–5）
- 每段 **2–15 秒**；總片長平均分配到段數（120 秒可寫 8 段、12 段……）
- 每段只寫當下動作，不要寫其他段內容
- 最後一段之後可用 `---` 分隔線追加共享風格描述（會被當 GLOBAL）

**範例（30 秒，3 段）：**

```text
GLOBAL:
A young woman with long silver hair, black leather jacket and motorcycle
helmet under her arm. Setting: a neon-lit city street at night after rain.
Style: cyberpunk cinematic, teal and magenta lights. Music: low pulsing synth.

SEGMENT 1:
She walks out of a small ramen shop, pulling on one glove, and looks up at the
neon signs. The camera cranes down from the signs to her face.

SEGMENT 2:
She mounts a black motorcycle, the engine starts with a deep roar. She taps the
fuel gauge twice. The camera circles around the bike.

SEGMENT 3:
She rides off down the wet street, neon reflections streaking across her
helmet, and disappears around a corner. The synth pulse fades out.
```

---

## 八、長片延續（/extend 或面板「📼 延續上一條長片」）

輸入要新增的秒數 + 尾端提示詞。**不需要重複人物設定**（checkpoint 已保存），只寫新劇情，開頭一句與原片結尾銜接即可。

```text
/extend 30
The same woman steps off the train onto the station platform, the piano theme
resumes softly as she walks toward the exit.
```

---

## 九、聲音描述速查（native 音畫聯合生成）

| 想要 | 寫法 |
|---|---|
| 音樂 | 曲風 + 樂器 + 節奏 + 情緒（"a slow melancholic piano piece"） |
| 環境音 | 地點 + 聲音源（"rain falling", "distant traffic"） |
| 對白 | 直接寫句子，H3 會生成說話與唇形（"She says: 'Wait for me.'"） |
| 音效 | 動作 + 聲音（"the clink of her cup"） |
| 全片統一音樂 | 寫進 GLOBAL（時間軸第一個標題前／GLOBAL 段落） |
| 無聲 | 明寫 "The scene is completely silent."（否則可能隨機加音樂） |

長片時每幕只需一句聲音描述（音樂已在 GLOBAL 定義），Motion Context 會自動接續音訊。

---

## 十、常見錯誤

| 錯誤 | 後果 | 正確做法 |
|---|---|---|
| 時間軸只寫到 50 秒但選了 60 秒 | Bot 直接拒絕 | 覆蓋 0–60 全程 |
| 一幕超過 15 秒 | 拆段不自然 | 拆成多幕 |
| 時間軸有缺口或重疊（3-8、7-12） | 拒絕 | 首尾相接（3-8、8-15） |
| 把人物外貌寫進每一幕 | 佔字數、易漂移 | 寫在 GLOBAL，只寫一次 |
| 寫「她接下來會見到男主角」 | 畫面搶跑 | 只寫當幕發生的事 |
| 只用形容詞（beautiful、epic） | 畫面空洞 | 寫具體可見的細節與動作 |
| 手寫 `<Picture 1>` 等標籤 | 嚴格模式報錯 | 素材交給 Bot 面板上傳，標籤自動管理 |
| 忽略聲音描述 | 音訊隨機 | 每幕至少一句聲音 |
| I2V 把整張圖片外觀重寫一遍 | 與圖衝突、字數浪費 | 只寫動作、運鏡、聲音 |
| 長片 SEGMENT 跳號（1、3） | 拒絕 | 從 1 連續編號 |
