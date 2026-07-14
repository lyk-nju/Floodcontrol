# 02 离线数据处理与加载

状态：`STRICT4_SCHEMA_IMPLEMENTED / NATIVE_SOURCE_REQUIRED`

## 本文只回答什么

- 原生SMPL/AMASS rotations怎样retarget并转换为新版full-clip representation。
- full-root recovery、body extraction、全局 yaw augmentation、OriginEpoch 与 window 的采样顺序。
- tokenizer/VAE 训练样本和 LDF 训练样本为何需要不同的裁剪/缓存策略。
- deterministic body code、root/body statistics、cache manifest 和 Dataset/DataLoader 接口。
- frame/token mask、文本片段和四帧边界如何对齐。

## 本文不回答什么

- 网络层和 attention 结构。
- 推理时 persistent noisy state 怎样 rebase。
- loss、optimizer 和训练步数。

## 已发现的关键风险

旧HumanML263在预处理时丢弃了原生rotations，不能无损构造body265。新版数据入口因此要求native local rotations、root translation、parents和offsets；缺失时明确fail-fast，不从joint positions运行IK。

新版候选不变量是：

```text
native SMPL/AMASS local rotations + root translation
    -> retargeted HumanML22 skeleton
    -> full-sequence FK
    -> extract root5 + body265
    -> optional session-level yaw augmentation
    -> choose OriginEpoch e
    -> choose window start s, with e <= s
    -> translation-only rebase
    -> crop/pack/normalize
```

上述schema、full-sequence-before-crop顺序和strict4 artifact dataset已经实现并通过合成native-rotation集成测试；真实数据质量仍需在原生数据路径就绪后验收。

## 需要分别设计的数据产品

```text
FullClipMotionStore
  root_pose_full
  body_features_full
  texts/segments
  frame_length
  representation_version

BodyCodeStore
  deterministic full-prefix body codes
  token masks/lengths
  tokenizer checkpoint + stats identity

HybridWindowSample
  epoch origin metadata
  history/generation/future slices
  explicit root patches
  body code slices
  observations and masks
```

## 当前待讨论问题

- 全局 yaw augmentation 的概率/分布，以及 cold-start initial heading condition 如何同步变换。
- epoch origin `e` 与 window start `s` 的联合采样分布。
- VAE训练clip在1–10秒范围内按完整四帧patch裁剪；crop中间位置必须携带真实preceding frame。
- LDF target 是否必须来自 full-prefix encode 后缓存的 deterministic code。
- root statistics 在哪一种 epoch/window sampling policy 下统计。
- BABEL 多文本段落和 future observation 如何跨 crop 保留绝对 frame index。
- 尾部不足四帧在artifact构建时显式丢弃；batch padding只允许完整四帧patch并携带frame/token mask。
- 新旧 Dataset 类、cache 目录和 manifest 如何隔离，确保 baseline 可复现。

## 冻结条件

- 同一完整 motion token 无论被哪个 LDF window 采到，都得到同一个 body target code。
- full recover 后的 slice 不会被 window-local recovery 重置 yaw。
- `e=s` 和 `e<s` 都有测试样本。
- 所有随机变换发生在物理空间且同步作用于对应的世界约束。
- 数据 cache 能拒绝 tokenizer、stats、token protocol 或 representation version 不匹配。
