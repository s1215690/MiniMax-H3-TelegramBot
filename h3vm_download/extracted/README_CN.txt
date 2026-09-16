H3 VRAM Master v0.21.1｜中文安装与使用说明

【发布内容】
1. ComfyUI-H3-VRAM-Master/
   正式插件目录，复制到 ComfyUI/custom_nodes/。

2. 01_H3VM_Core_Reference_Workflow.json
   当前正式参考工作流。已经接好 H3VM Core 与 H3VM Video VAE。

【正式产品边界】
H3VM 只提供两颗基础设施节点：
- H3VM Core｜模型执行引擎
- H3VM Video VAE｜双卡加速解码

H3VM 不接管 Prompt、Seed、分辨率、时长、Sampler、Sigmas、实际采样步数。
原工作流仍由原节点控制。

【接入方式】
原 Model Loader -> H3VM Core -> 原工作流后续 MODEL 输入

Video VAE 部分：
LATENT + 原 Video VAE -> H3VM Video VAE -> IMAGE
Core 的“模式”和“VAE 策略”输出分别连接到 H3VM Video VAE。

【采样步数提示】
Core 的“采样步数提示”只用于 H3VM 预取 / 末步质量调度，不是运行边界，也不会修改实际采样步数。
真实采样开始 / 结束由 Sampler 生命周期自动识别；提示值填错不会让旧 mailbox / predictor history 跨到下一条任务。
建议仍尽量与常用 steps 对齐。随包参考工作流采用标准 BasicScheduler=20，因此提示值预设为 20。

【双卡 VAE 加速】
开关位于 H3VM Core 统一面板。
- 开启：双卡 Video VAE 时间块解码。
- 关闭：回原生单卡 VAE 解码。
- Core 为单卡模式时：强制单卡 VAE。
如果两张显卡性能差距很大，双卡 VAE 可能没有正收益，可直接关闭。

双卡 VAE 时，Video VAE 节点里的“副卡 VAE 文件”必须选择与上游 VAE Loader 相同的 MiniMax H3 Video VAE 权重。v0.21.1 会在可比较时做轻量签名校验，明显不一致会直接拒绝。

【默认设置】
- 界面语言：中文
- 多卡支持：开启
- 双卡 VAE 加速：开启
- 运行模式：双卡极速
- 加速算法：频谱极速
- 副卡参与度：100%
- 采样步数提示：20（标准值；仅提示，非运行边界）

【兼容基线】
本包按 ComfyUI v0.35.0 当前接口完成静态 / API 检查。旧版 ComfyUI 如果缺少多卡 clone 或 wrapper 接口，请先升级。

【重要】
只安装 H3VM、但工作流不使用 H3VM 节点时，H3VM runtime 不会激活，不应修改标准 ComfyUI Loader / CLIP / VAE 行为。

文本编码：UTF-8；中文说明文件保留 BOM 以兼容旧版 Windows 记事本。ZIP 内文件名全部使用 ASCII，避免 Windows/7-Zip/WinRAR 中文文件名乱码。
