# 02 离线数据处理与加载

状态：`FULL_CLIP_DATASETS_AND_TASK_COLLATORS_READY / LDF_TRAINER_PENDING`

## 本文只回答什么

- HumanML3D/BABEL的HumanML-style 263D怎样恢复并转换为新版full-clip representation。
- full-root recovery、body extraction、全局 yaw augmentation 与 task window 的采样顺序。
- tokenizer/VAE 训练样本和 LDF 训练样本为何需要不同的裁剪/缓存策略。
- online deterministic body encoding、root/body/latent statistics、TXT split metadata和Dataset/DataLoader接口。
- frame/token mask、文本片段和四帧边界如何对齐。

## 本文不回答什么

- 网络层和 attention 结构。
- 推理时 persistent noisy state 怎样 rebase。
- loss、optimizer 和训练步数。

## 已发现的关键风险

HumanML263中的rotation来自其官方预处理阶段的IK，而非原生AMASS pose；这会限制rotation监督的精度，但不阻止构造结构一致的body265。第一版明确接受该来源，不得把它描述为原生SMPL监督；这一事实记录在转换代码和文档中，不再复制到每个NPZ的runtime metadata。

新版候选不变量是：

```text
HumanML3D root deltas + RIC positions + IK local rotations
    -> recover root position/full yaw and global joint positions
    -> compose 22 global rotations with the HumanML hierarchy
    -> recompute backward global velocity in m/s
    -> extract root5 + body265
    -> training-time uniform global yaw augmentation
    -> task collator chooses a four-frame-aligned window
    -> translation-only rebase
    -> crop/pack/normalize
```

上述schema和full-sequence-before-crop顺序已经实现。263D与265D的完整通道定义、root恢复、rotation层级组合和唯一转换入口集中在离线`tools/convert_motion_263_to_265.py`；`tools/build_motion_artifact.py`只原子写入`root_motion/body_motion/body_feature_valid_mask`三个字段。旧NPZ中的version、hash或FPS metadata会被Dataset忽略。

`HumanML3DDataset`与`BABELDataset`分别解析自己的split和`caption#tokens#start#end`文本，且不互相继承；统一返回完整未裁剪sample：`dataset/name/root_motion/body_motion/body_feature_valid_mask/text_data`。处理后的dataset root自包含split、`artifacts/`和`texts/`；需要文本的任务使用相对目录`text_path: texts`，纯VAE训练可显式设置`text_path: null`，避免读取未消费的caption。motion NPZ只保存三组tensor，文本保持独立TXT，不把字符串复制进每个NPZ。公开source identity固定为`HumanML3D`和`BABEL`，不从数据目录名推导，因此目录移动或重命名不会改变batch metadata。`MultiDataset`只实例化并concat子Dataset，不拥有collate。

任务相关处理下沉到training data层：`utils/training/vae/data.py`负责VAE随机/确定性crop、translation rebase、同步yaw、previous-root和padding；`utils/training/ldf/data.py`负责active window、固定`encoder_context_tokens × 4`左历史、cold-start左零填充、context mask以及相对token文本。LDF同区间多caption在train随机选一个、validation取第一个。当前LDF data contract已实现，但OriginEpoch、future observation、history/generation noise和trainer接线仍未实现。

statistics和VAE reconstruction evaluation直接消费Dataset full sample，不读取artifact manifest。physical statistics仍用四个quarter-turn quadrature点匹配均匀yaw的一阶和逐维二阶矩；latent statistics改用显式seed的普通随机生成器，不再用sample hash决定yaw。

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

python -m tools.compute_vae_latent_stats \
  --config configs/vae.yaml \
  --checkpoint ${VAE_RUN}/step_299999.ckpt \
  --motion-stats ${RAW_DATA}/HumanML3D_motion/motion_stats.npz \
  --train-meta-paths ${RAW_DATA}/HumanML3D_motion/train.txt \
  --output ${VAE_RUN}/latent_stats.npz

# 正式LDF训练直接从checkpoint加载EMA encoder并在线产生deterministic mu。
```

`configs/vae.yaml`保留HumanML-only基线，`configs/vae_multi.yaml`使用HumanML3D+BABEL联合statistics。BABEL源目录没有正式test split，因此构建器默认只发布train/val；空的`test_processed.txt`和调试用`test_min_processed.txt`不会被伪装成正式test。

## 为什么不保留 `datasets/generate.py`

旧版`GenerateDataset`只在test split中读取一组写死的prompt与人工时长，再创建全零`feature/token`占位数组。它没有加载真实motion，也没有提供可作为监督的数据真实性，因此不属于HumanML3D/BABEL/Multi这一层的Dataset职责。

在当前root5/body265协议下，Dataset sample表示一个已经存在的完整动作事实：它必须有真实`root_motion/body_motion/body_feature_valid_mask`。纯文本生成请求只有caption、目标时长和未来可能加入的约束，不能用全零动作伪装成ground truth，否则statistics、VAE reconstruction和LDF mask都会把占位值误当数据。后续需要批量生成固定prompt时，应在具体eval或inference任务中构造generation request/timeline；它可以复用文本区间格式，但不进入训练Dataset，也不需要通用Dataset collate。当前代码没有`GenerateDataset`调用，migration guard明确要求`datasets/generate.py`保持删除。

## 需要分别设计的数据产品

```text
FullClipMotionStore
  root_motion [F,5]
  body_motion [F,265]
  body_feature_valid_mask [F,265]
  text_data [{text,tokens,start_frame,end_frame}]

VAETrainingBatch
  aligned physical crop + previous root
  synchronized yaw / translation rebase
  frame/feature masks and batch padding

LDFDataBatch
  active root/body window + previous root
  fixed left body encoder context and masks
  relative token text intervals
```

## 当前待讨论问题

- 全局 yaw augmentation 冻结为train split每次采样`Uniform(0,2π)`；cold-start heading、previous-root boundary和全部world-space body feature同步变换。
- VAE训练clip在1–10秒范围内按完整四帧patch裁剪；crop中间位置必须携带真实preceding frame。
- LDF target固定由冻结EMA encoder在线产生deterministic `mu`；active crop前必须携带完整encoder感受野，encode后丢弃warm-up context输出。
- root statistics 在哪一种 epoch/window sampling policy 下统计。
- LDF的OriginEpoch、future observation与history/generation noise如何在当前简单window contract之上扩展。
- 尾部不足四帧在artifact构建时显式丢弃；batch padding只允许完整四帧patch并携带frame/token mask。

## 冻结条件

- 同一完整motion token无论被哪个LDF window采到，都通过相同历史context得到同一个deterministic `mu`。
- full recover 后的 slice 不会被 window-local recovery 重置 yaw。
- 所有随机变换发生在物理空间且同步作用于对应的世界约束。
- online EMA encoder与HybridMotion trainer bridge接入后，必须保持context mask与active slice一致。
