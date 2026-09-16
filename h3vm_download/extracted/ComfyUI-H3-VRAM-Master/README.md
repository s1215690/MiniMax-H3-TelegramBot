# ComfyUI-H3-VRAM-Master v0.21.1

正式版聚焦 **H3VM Core + H3VM Video VAE** 两颗基础设施节点。

## 接入方式

主模型：

`原 Model Loader -> H3VM Core -> 原工作流后续节点`

Video VAE：

`LATENT + 原 Video VAE + Core 的 mode / vae_policy -> H3VM Video VAE -> IMAGE`

H3VM 不接管 Prompt、Seed、分辨率、视频时长、Sampler、Sigmas、实际采样步数、CLIP/文本编码器或原工作流 LoRA/Conditioning。

## 统一控制面板

**双卡 VAE 加速开关现在位于 H3VM Core 面板。**

- 开启：Video VAE 节点按 Core 输出策略使用双卡时间块解码。
- 关闭：Video VAE 节点直接回原生单卡 `vae.decode()`。
- Core 关闭“多卡支持”时，VAE 自动强制单卡。
- 双卡性能差距很大时，建议关闭双卡 VAE 加速；主模型仍可继续使用 H3VM 双卡模式。

Video VAE 节点不再重复提供开关，只接收 Core 的 `vae_policy`。

### 采样步数提示不是运行边界

Core 的“采样步数提示”只用于 H3VM 的预取 / 末步质量调度，**不会修改 Sampler 的实际 steps，也不再负责判断一轮采样何时结束**。真实采样开始 / 结束由 ComfyUI `OUTER_SAMPLE` 生命周期自动识别，所以提示值即使与实际 steps 不一致，也不会把上一条任务的 mailbox / predictor history 延续到下一条任务。

建议仍把提示值设置为常用实际 steps，以获得更准确的末步质量调度。随包参考工作流按 MiniMax H3 标准配置使用 `BasicScheduler = 20 steps`，Core 的提示值同步为 `20`。

### Video VAE 同源保护

双卡 Video VAE 需要两张卡使用同一套 MiniMax H3 Video VAE 权重。`vae` 输入来自上游 VAE Loader；`vae_name` 是副卡要加载的文件，因此两者必须指向同一权重。

v0.21.1 会在权重可比较时做轻量签名校验；发现明显不一致会直接报错，而不是让两张卡各解一半不同 VAE 后再拼接。

## 公共模式

- **双卡极速**：速度优先；默认频谱极速。
- **双卡扩容**：显存容量优先。
- **双卡后台**：保守双卡协同。
- 关闭“多卡支持”即单卡。

副卡参与度提供 100% / 75% / 50% / 25% / 自定义。

## 中英文界面

Core 与 Video VAE 节点均提供 `界面语言 / Language`。翻译层只修改显示标签和节点标题，不重排 widget、不修改后端值、不覆盖序列化。

## 启动隔离

安装插件但不执行 H3VM 节点时，H3VM runtime 不激活，也不会替换标准 ComfyUI CLIP/VAE Loader。

## ComfyUI 兼容基线

本发布包按 **ComfyUI v0.35.0** 当前接口完成静态 / API 兼容检查。H3VM 依赖多卡 clone、ModelPatcher wrapper 等较新的 ComfyUI 能力；旧版本如果缺少相关接口，请先更新 ComfyUI。

硬件运行表现仍取决于显卡组合、驱动、模型与工作流。若平台本身出现 compiler / VRAM 回归，建议先用关闭 compiler 作为诊断对照，而不是把它当作 H3VM 的永久必需参数。

## 安装

将 `ComfyUI-H3-VRAM-Master` 放入 `ComfyUI/custom_nodes/` 后重启。

附带正式参考工作流：`01_H3VM_Core_Reference_Workflow.json`。
