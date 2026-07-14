# 02 Body VAE 与显式动作表示

状态：`MODEL_DATA_AND_STATS_READY / GPU_TRAINING_PENDING`

## 目标

新版动作状态由explicit root与latent body组成。VAE只编码body，root保持为LDF可直接生成和约束的结构化变量。该设计参考ARDY的hybrid representation、Patch4 tokenizer和local-root-conditioned decoder，但保留FloodDiffusion的因果卷积工程基础。

## 物理表示

```text
root_motion [B,F,5]
  root xyz                                3
  cos/sin global heading                 2

body_motion [B,F,265]
  non-root planar-root-relative positions 21*3 = 63
  global joint rotation 6D                22*6 = 132
  backward global joint velocities        22*3 = 66
  foot contacts                                  4
```

positions只从x/z减去planar root，y保留世界高度。rotations和velocities保持全局朝向，因此随机yaw增强必须同步旋转root、positions、rotations和velocities。

## Token与因果合同

- `FRAMES_PER_TOKEN=4`，模型输入帧数必须整除4。
- encoder在进入网络前将连续四帧reshape为一个patch；不存在首帧特殊token。
- encoder/decoder均在token轴使用causal convolution，token t不读取t+1。
- decoder每次读取一个128D latent token和一个`[4,4]` local-root patch，严格输出四帧。
- decoder cache由调用方的`VAEDecoderState`持有，不允许module-global cache或`first_chunk`开关。

## Local root

```text
local_root_motion [B,T,4,4]
  [yaw_rate, current-heading-local vx, current-heading-local vz, root_y]
```

差分使用当前帧减前一帧。裁剪位于原序列中间时必须携带真实previous root；cold start的首帧yaw/velocity无效并置零，root height仍有效。该处有意不同于ARDY公开代码的forward/world-axis velocity，以满足persistent decoder在token边界不读取未来帧的要求。

## 统计所有权

```text
global root statistics   [5]
local root statistics    [4]
body continuous stats    [261]
latent mu statistics     [128]
```

contacts保持0/1，不参与z-score。前三组只从train split计算；latent stats只在VAE冻结后用deterministic posterior mu计算。数据划分沿用FloodNet协议：`train.txt/val.txt/test.txt`每行仅保存sample id，artifact NPZ保存contract与source hash，统计和latent metadata记录split TXT、skeleton、motion stats及checkpoint hash，禁止identity stats进入真实模型。

train Dataset在物理空间施加独立`Uniform(0,2π)`全局yaw，因此第一帧yaw不会固定为零。该变换同步作用于root xz/heading、root-relative positions、global rotations、global velocities和previous-root boundary；contacts保持不变，current-heading-local root velocity在变换前后保持不变。physical statistics使用`0/90/180/270`度四点quadrature；由于所有受影响通道都对yaw的cos/sin线性，该方法精确匹配连续均匀yaw的一阶与逐维二阶矩，同时保持统计结果确定性。

ARDY使用一个physical stats集合的root/local/body切片，并为tokenizer latent保存额外统计；Floodcontrol将这些所有权显式拆开，但设计意图同构。

## VAE与loss

encoder输出`mu/logvar [B,T,128]`。VAE训练通过reparameterization采样，LDF预编码固定使用mu。

首版训练目标冻结为`L_total = L_recon + 0.01 L_skate + 1e-5 L_KL`，不使用KL warmup。其中`L_recon`由三个分别归一化、分别masked-mean的continuous SmoothL1 block（position、rotation、velocity）和contact BCE相加，避免132D rotation仅凭维数支配重建目标。geodesic、FK-to-GT、direct/FK position consistency和backward velocity consistency不进入首版正式配置，只保留为后续独立消融能力。

正式训练从每段动作中随机裁剪20–200帧（1–10秒、四帧对齐）的片段，共训练300k optimizer steps。优化采用FloodDiffusion验证过的AdamW `2e-4`与constant-after-warmup schedule：前1k steps线性升至基础学习率，之后保持恒定，便于在300k后按验证曲线直接续训。使用三卡DDP，每卡batch size 32，global batch size为96；运行时通过`CUDA_VISIBLE_DEVICES=2,3,4`绑定物理GPU，Lightning配置只声明`devices: 3`。

## LDF接口补充

body保存global rotations，因此pure-noise cold start还需要绝对heading。Body Stage从唯一clean root派生首个有效帧的`[cos(yaw), sin(yaw)]` heading condition；该值与local root一起在stage boundary detach，不读取raw constraints，也不把Body loss传回Root Stage。

## 当前阻塞

第一版正式数据源冻结为现有HumanML3D 263D。预处理恢复root轨迹和global joint positions，将其中21个IK-derived local rotation6d按HumanML22层级组合成22个global rotations，并从恢复后的位置重新计算backward global velocity；旧263D的heading-local forward velocity不直接复用。artifact显式记录`humanml3d-263-ik-v1`，说明rotation监督不是原生AMASS pose。本地完整`HumanML3D_motion` artifacts和train-split statistics已经生成并通过真实batch装配；Web与真实LDF训练继续fail-fast直到VAE checkpoint和latent artifacts就绪。
