# 02 Strict-4 Body VAE 与显式动作表示

状态：`MODEL_CORE_IMPLEMENTED / NATIVE_ROTATION_DATA_REQUIRED`

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

contacts保持0/1，不参与z-score。前三组只从strict4 train split计算；latent stats只在VAE冻结后用deterministic posterior mu计算。artifact记录contract、manifest、skeleton、motion stats和checkpoint hash，禁止identity stats进入真实模型。

ARDY使用一个physical stats集合的root/local/body切片，并为tokenizer latent保存额外统计；Floodcontrol将这些所有权显式拆开，但设计意图同构。

## VAE与loss

encoder输出`mu/logvar [B,T,128]`。VAE训练通过reparameterization采样，LDF预编码固定使用mu。

首版loss为三个独立归一化continuous block的SmoothL1、contact BCE、权重0.01的predicted-contact skating loss，以及warmup至`1e-4`的KL。geodesic、FK-to-GT、direct/FK position consistency和backward velocity consistency均为独立可选项且默认权重为零；其中FK相关项没有版本化HumanML22 parents/offsets时会明确报错。

## LDF接口补充

body保存global rotations，因此pure-noise cold start还需要绝对heading。Body Stage从唯一clean root派生首个有效帧的`[cos(yaw), sin(yaw)]` heading condition；该值与local root一起在stage boundary detach，不读取raw constraints，也不把Body loss传回Root Stage。

## 当前阻塞

当前工作区只有legacy HumanML joint vectors/positions，没有ARDY式保留的原生SMPL rotations。真实预处理和训练必须提供retargeted native rotations manifest；不允许通过IK或旧263D临时恢复。模型核心、schema、统计工具和合成测试可独立验证，Web与真实LDF训练继续fail-fast直到VAE checkpoint和latent artifacts就绪。
