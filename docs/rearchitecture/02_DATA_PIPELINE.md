# 02 离线数据处理与加载

状态：`FULL_CLIP_DATASETS_AND_LDF_SPAN_READY / LDF_TRAINER_PENDING`

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

`HumanML3DDataset`与`BABELDataset`分别解析自己的split和`caption#tokens#start#end`文本，且不互相继承；统一返回完整未裁剪sample：`dataset/name/root_motion/body_motion/body_feature_valid_mask/text_data`。处理后的dataset root自包含split、`all.txt`、`artifacts/`和`texts/`；`all.txt`严格等于所有正式处理后split的唯一ID并集，只定义数据集级文本库存，不参与train/val/test采样。需要文本的任务使用相对目录`text_path: texts`，纯VAE训练可显式设置`text_path: null`，避免读取未消费的caption。motion NPZ只保存三组tensor，文本保持独立TXT，不把字符串复制进每个NPZ。公开source identity固定为`HumanML3D`和`BABEL`，不从数据目录名推导，因此目录移动或重命名不会改变batch metadata。`MultiDataset`只实例化并concat子Dataset，不拥有collate。

任务相关处理下沉到training data层：`utils/training/vae/data.py`负责VAE crop、translation rebase、同步yaw、previous-root和padding；LDF的`MinimumFrameDataset`只排除不足一个5-token active band的样本。`LDFSpanCollator`为每个sample保留自然长度并最多裁40 tokens/160 frames：短动作不缩短，长动作随机裁一个8秒parent；batch只在右侧padding并输出`span_token_count[B]`。它同时携带真实VAE左context、previous root、source坐标和token prompt timeline，但不决定H/active/frontier，也不做translation anchor、noise或self-forcing。Lightning plan随后在每个sample内独立采样H；只有`source_start_token=0`的真实序列前缀允许`H=0`，中间parent必须`H>=1`。HumanML3D从覆盖parent的caption备选中选择一条并重复到每个token；BABEL任意frame半开区间按四帧token最大重叠编译。训练crop、caption和yaw由sampler提供的epoch/sample seed决定；resume时恢复bucket epoch与已消费batch游标。

statistics和VAE reconstruction evaluation直接消费Dataset full sample，不读取artifact manifest。physical statistics仍用四个quarter-turn quadrature点匹配均匀yaw的一阶和逐维二阶矩；latent statistics改用显式seed的普通随机生成器，不再用sample hash决定yaw。

`tools/compute_ldf_root_stats.py`复用同一scaled-ARDY分布：自然parent最多40 tokens/160 frames；真实序列前缀均匀采样`H∈[0,N-5]`，中间parent均匀采样`H∈[1,N-5]`。`H>0`取第`4H-1`帧作为anchor，合法`H=0`取序列第0帧。产物仍只保存`root_mean/root_std [5]`。HumanML-only与HumanML+BABEL训练统一复用这份HumanML canonical root statistics。

当前可执行数据构建入口使用模块形式，避免依赖隐式`PYTHONPATH`：

### 一体化训练资产准备

迁移服务器或从空白派生目录重建全部训练资产时，首选统一入口
`tools.prepare_training_assets`。输入边界与FloodDiffusion保持一致：HumanML3D
必须已有`new_joint_vecs/texts/train.txt/val.txt/test.txt`，BABEL必须是已有
`motions/texts/train_processed.txt/val_processed.txt`的`BABEL_streamed`，而不是
官方BABEL JSON与AMASS原始文件。

VAE训练前阶段生成两套root5/body265数据、HumanML与联合physical statistics、
canonical HumanML root statistics以及两份UMT5表：

```bash
python -m tools.prepare_training_assets pre-vae \
  --raw-data-root /path/to/raw_data \
  --deps-root /path/to/deps \
  --workers 16 \
  --t5-devices 0,1,2
```

VAE训练完成或已有checkpoint迁入后，使用EMA encoder扫描完整HumanML train
split并把`latent_stats.npz`写到checkpoint同目录：

```bash
python -m tools.prepare_training_assets post-vae \
  --raw-data-root /path/to/raw_data \
  --deps-root /path/to/deps \
  --vae-checkpoint /path/to/vae_run/last.ckpt \
  --latent-device cuda:0
```

checkpoint从一开始就存在时可以直接运行全部阶段；`verify`只执行最终资产与
两份LDF配置的fail-fast检查：

```bash
python -m tools.prepare_training_assets all \
  --raw-data-root /path/to/raw_data \
  --deps-root /path/to/deps \
  --vae-checkpoint /path/to/vae_run/last.ckpt \
  --workers 16 \
  --t5-devices 0,1,2 \
  --latent-device cuda:0

python -m tools.prepare_training_assets verify \
  --raw-data-root /path/to/raw_data \
  --deps-root /path/to/deps \
  --vae-checkpoint /path/to/vae_run/last.ckpt
```

工具通过CLI覆盖`dirs.raw_data/dirs.deps`及最终VAE checkpoint，不修改正式训练
YAML中的绝对路径。每个昂贵阶段完成后都会验证产物并原子提交，同时更新
`raw_data/training_assets.json`。默认会复用当前路径下通过schema验证的现有产物；
模型配置或统计策略发生变化时必须显式传入`--force`，以重算statistics、T5和
latent statistics。motion转换器本身按root5/body265 schema断点续跑；本次迁移
假设两台服务器的HumanML3D/BABEL源数据相同。`--skip-t5`只用于无UMT5依赖的motion/
statistics准备或测试，此时不会宣称LDF配置已经完整ready。联合VAE statistics
在脚本内部通过真正的`MultiDataset`计算，不把BABEL split伪装成HumanML3D。

当前LDF仍由冻结EMA encoder在线产生deterministic body latent，因此该工具不会
生成逐样本latent cache、旧`TOKENS_*`或`z_mean/z_std`。

### 独立阶段入口

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
  --config configs/vae_multi.yaml \
  --override dirs.raw_data=${RAW_DATA} \
  --output ${RAW_DATA}/HumanML3D_BABEL_motion_stats.npz

python -m tools.compute_vae_latent_stats \
  --config configs/vae.yaml \
  --checkpoint ${VAE_RUN}/step_299999.ckpt \
  --motion-stats ${RAW_DATA}/HumanML3D_motion/motion_stats.npz \
  --train-meta-paths ${RAW_DATA}/HumanML3D_motion/train.txt \
  --output ${VAE_RUN}/latent_stats.npz

python -m tools.compute_ldf_root_stats \
  --train-meta-paths ${RAW_DATA}/HumanML3D_motion/train.txt \
  --output ${RAW_DATA}/HumanML3D_motion/root_stats.npz \
  --min-frames 20 \
  --max-frames 160 \
  --windows-per-sample 1 \
  --active-tokens 5 \
  --seed 1234

python -m tools.pretokenize_t5_text \
  --config configs/ldf_s5.yaml \
  --output ${RAW_DATA}/HumanML3D_motion/t5_text_embeddings.pt

python -m tools.pretokenize_t5_text \
  --config configs/ldf_multi.yaml \
  --output ${RAW_DATA}/HumanML3D_BABEL_t5_text_embeddings.pt

# 正式LDF训练直接从checkpoint加载EMA encoder并在线产生deterministic mu。
```

两份LDF配置通过`data.text_meta_paths`显式指向各数据集的`all.txt`；T5工具只读取这些完整库存，不再从某次实验的train/val/test/probe字段猜测文本范围。文本表在离线写出时对encoder/tokenizer身份、caption和embedding内容生成一次`content_id`。训练加载只检查metadata与shape，具体tensor在第一次lookup时才检查finite，避免启动阶段扫描整个mmap文件；LDF checkpoint保存`content_id`，即使路径不变而表内容被替换，resume也会失败。

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

LDFSpanBatch
  per-sample natural parent window capped at 40 tokens / 160 frames + right padding
  span_token_count [B] + independent source crop index
  all available real left context within the encoder receptive field
  true-cold/continuation boundary + previous root validity
  per-token prompt timeline; no beta/noise/rollout fields
```

`text_data`只包含与motion具有正长度覆盖的half-open区间。HumanML的`0#0`仍表示整段caption；其余零时长、反向、或裁剪到motion边界后为空的源annotation在Dataset解析时丢弃，不能交换端点或扩张成假区间。BABEL使用相同的正覆盖规则。LDF collator继续把非正区间视为上游合同错误。

## 当前待讨论问题

- 全局 yaw augmentation 冻结为train split每次采样`Uniform(0,2π)`；cold-start heading、previous-root boundary和全部world-space body feature同步变换。
- VAE训练clip在1–10秒范围内按完整四帧patch裁剪；crop中间位置必须携带真实preceding frame。
- LDF target固定由冻结EMA encoder在线产生deterministic `mu`；active crop前携带感受野内全部真实可用历史，encode后按逐样本offset丢弃warm-up context输出。只有真实序列起点附近允许少于完整感受野，且不插入假历史token。
- 正式root statistics需要使用已实现的固定anchor sampler重新生成并写入训练资产。
- LDF condition阶段已经从同一absolute XZ计划编译span内current observation与带timeline position的future goal，并独立执行constraint dropout。
- 尾部不足四帧在artifact构建时显式丢弃；batch padding只允许完整四帧patch并携带frame/token mask。

## 冻结条件

- 同一完整motion token通过其感受野内全部真实历史得到唯一deterministic `mu`；序列起点不足的历史由encoder自身的causal边界处理，而不是由collator显式补token。
- full recover 后的 slice 不会被 window-local recovery 重置 yaw。
- 所有随机变换发生在物理空间且同步作用于对应的世界约束。
- online EMA encoder与HybridMotion trainer bridge必须保持context mask与active slice一致；当前实现已由真实checkpoint/data smoke验证。
