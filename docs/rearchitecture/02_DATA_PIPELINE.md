# 02 离线数据处理与加载

状态：`HUMANML_AND_BABEL_ARTIFACTS_READY / MULTI_STATS_READY`

## 本文只回答什么

- HumanML3D/BABEL的HumanML-style 263D怎样恢复并转换为新版full-clip representation。
- full-root recovery、body extraction、全局 yaw augmentation、OriginEpoch 与 window 的采样顺序。
- tokenizer/VAE 训练样本和 LDF 训练样本为何需要不同的裁剪/缓存策略。
- online deterministic body encoding、root/body/latent statistics、TXT split metadata、artifact contract和Dataset/DataLoader接口。
- frame/token mask、文本片段和四帧边界如何对齐。

## 本文不回答什么

- 网络层和 attention 结构。
- 推理时 persistent noisy state 怎样 rebase。
- loss、optimizer 和训练步数。

## 已发现的关键风险

HumanML263中的rotation来自其官方预处理阶段的IK，而非原生AMASS pose；这会限制rotation监督的精度，但不阻止构造结构一致的body265。第一版明确接受该来源，并在artifact metadata中记录`humanml3d-263-ik-v1`，不得把它描述为原生SMPL监督。

新版候选不变量是：

```text
HumanML3D root deltas + RIC positions + IK local rotations
    -> recover root position/full yaw and global joint positions
    -> compose 22 global rotations with the HumanML hierarchy
    -> recompute backward global velocity in m/s
    -> extract root5 + body265
    -> training-time uniform global yaw augmentation
    -> choose OriginEpoch e
    -> choose window start s, with e <= s
    -> translation-only rebase
    -> crop/pack/normalize
```

上述schema、full-sequence-before-crop顺序和唯一`humanml265`转换器已经实现。转换方法的恢复、旋转组合、backward velocity与contact规则直接记录在转换函数注释中，不拆分多套feature子版本。`HumanML3DDataset`消费`HumanML3D_motion`，`BABELDataset`消费`BABEL_motion`；两者保留独立source identity，但输出同一个root5/body265 batch contract。`MultiDataset`只做同构Dataset的concat，不再兼容旧feature/token/traj键集合。训练期每次采样独立均匀yaw；root、heading、body positions、global rotations、global velocities和previous-root boundary同步旋转，contacts与validity不变。统计工具使用四个quarter-turn quadrature点，精确匹配均匀yaw增强下一阶和逐维二阶矩，并将实际train artifact manifest hash写入statistics；训练入口发现source hash、converter或FPS变化后会在优化前fail-fast。

当前可执行数据构建入口使用模块形式，避免依赖隐式`PYTHONPATH`：

```bash
python -m tools.preprocess_humanml3d \
  --source-root ${RAW_DATA}/HumanML3D \
  --output ${RAW_DATA}/HumanML3D_motion \
  --splits train val test \
  --workers 8

python -m tools.compute_vae_stats \
  --train-meta-paths ${RAW_DATA}/HumanML3D_motion/train.txt \
  --output ${RAW_DATA}/HumanML3D_motion/motion_stats.npz

python -m tools.preprocess_babel \
  --source-root ${RAW_DATA}/BABEL_streamed \
  --output ${RAW_DATA}/BABEL_motion \
  --workers 8

python -m tools.compute_vae_stats \
  --train-meta-paths \
    ${RAW_DATA}/HumanML3D_motion/train.txt \
    ${RAW_DATA}/BABEL_motion/train.txt \
  --output ${RAW_DATA}/HumanML3D_BABEL_motion_stats.npz

python -m tools.export_vae_tokenizer \
  --checkpoint ${VAE_RUN}/step_299999.ckpt \
  --motion-stats ${RAW_DATA}/HumanML3D_motion/motion_stats.npz \
  --output ${VAE_RUN}/tokenizer_ema.pt

# tools.pretokenize_body_latents仅保留为诊断/可选加速工具；
# 正式LDF训练由冻结EMA encoder在线产生deterministic mu。
```

`configs/vae.yaml`保留HumanML-only基线，`configs/vae_multi.yaml`使用HumanML3D+BABEL联合statistics。BABEL源目录没有正式test split，因此构建器默认只发布train/val；空的`test_processed.txt`和调试用`test_min_processed.txt`不会被伪装成正式test。

## 需要分别设计的数据产品

```text
FullClipMotionStore
  root_pose_full
  body_features_full
  texts/segments
  frame_length
  representation_version

OnlineBodyEncodingBatch
  physical body crop + encoder warm-up context
  frame/token masks and active-token slice
  frozen EMA tokenizer + latent-statistics identity

HybridWindowSample
  epoch origin metadata
  history/generation/future slices
  explicit root patches
  online deterministic mu slices
  observations and masks
```

## 当前待讨论问题

- 全局 yaw augmentation 冻结为train split每次采样`Uniform(0,2π)`；cold-start heading、previous-root boundary和全部world-space body feature同步变换。
- epoch origin `e` 与 window start `s` 的联合采样分布。
- VAE训练clip在1–10秒范围内按完整四帧patch裁剪；crop中间位置必须携带真实preceding frame。
- LDF target固定由冻结EMA encoder在线产生deterministic `mu`；active crop前必须携带完整encoder感受野，encode后丢弃warm-up context输出。
- root statistics 在哪一种 epoch/window sampling policy 下统计。
- LDF阶段的BABEL多文本段落和future observation如何跨crop保留absolute timeline index；这不属于body VAE Dataset职责。
- 尾部不足四帧在artifact构建时显式丢弃；batch padding只允许完整四帧patch并携带frame/token mask。
- 新旧Dataset类、artifact目录和split TXT如何隔离，确保baseline可复现。

## 冻结条件

- 同一完整motion token无论被哪个LDF window采到，都通过相同历史context得到同一个deterministic `mu`。
- full recover 后的 slice 不会被 window-local recovery 重置 yaw。
- `e=s` 和 `e<s` 都有测试样本。
- 所有随机变换发生在物理空间且同步作用于对应的世界约束。
- 在线encoder bridge能拒绝tokenizer、stats、context protocol或representation version不匹配。
